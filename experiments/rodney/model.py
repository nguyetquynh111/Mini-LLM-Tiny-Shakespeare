"""Byte-level GPT: configs + architecture. Nothing else lives here.

Every tensor op carries a shape comment in (B, T, C) terms:

    B  = batch size
    T  = time / sequence length, at most block_size
    C  = n_embd, the residual stream width
    nh = n_head
    hs = head size = C // nh          (so nh * hs == C exactly)
    V  = vocab size = 256, locked

Deliberately NOT used, per the brief: nn.Transformer, nn.TransformerEncoderLayer,
nn.MultiheadAttention, F.scaled_dot_product_attention. Attention is written from
matmuls, a torch.tril mask, and a masked softmax.

Run `python model.py` for a parameter breakdown and an end-to-end shape trace.
"""

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class GPTConfig:
    """Everything needed to rebuild a model. Checkpoints embed one of these, so
    evaluate.py never has to guess an architecture to load weights into."""

    block_size: int  # context length T_max, in BYTES
    n_layer: int
    n_head: int
    n_embd: int  # C; must be divisible by n_head
    dropout: float = 0.1
    bias: bool = True
    vocab_size: int = 256  # locked: byte-level, see train.py

    def __post_init__(self) -> None:
        # This is the constraint that forces Model B to n_embd=384 rather than
        # 256: the residual stream is split evenly across heads, so C must be a
        # multiple of nh. 256/6 is not an integer; 384/6 = 64 is.
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head}); "
            f"heads split the residual stream evenly."
        )

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head


# A TA reproduces either model with a single flag: --model A  or  --model B
CONFIGS: dict[str, GPTConfig] = {
    # Baseline. ~0.47M params.
    "A": GPTConfig(block_size=64, n_layer=2, n_head=4, n_embd=128, dropout=0.1),
    # Scaled. ~10.9M params. 6 heads x 64 = 384.
    #
    # Why block_size jumps 64 -> 256 specifically:
    # A byte token carries at most 8 bits, and in English prose only ~1-1.5 bits
    # of actual information -- far less than a word- or subword-level token,
    # which packs several characters (and a morpheme's worth of meaning) into one
    # slot. Shakespeare averages ~4.5 bytes per word, so:
    #
    #     A: 64 bytes  ~= 14 words  ~= one line of verse, and not a whole one.
    #     B: 256 bytes ~= 57 words  ~= four lines -- enough to hold a speaker
    #                                   name, close a clause, keep a rhyme in view.
    #
    # Same architecture, but only B has a window long enough for the structure it
    # is being asked to model. The price is quadratic: the attention matrix goes
    # from (T=64)^2 = 4,096 entries per head to (T=256)^2 = 65,536, a 16x jump.
    # That is the single biggest driver of B's memory and step time -- not depth.
    "B": GPTConfig(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.1),
}

# --------------------------------------------------------------------------
# Observation for the write-up: 191 dead output logits
# --------------------------------------------------------------------------
#
# tinyshakespeare is pure ASCII and uses only 65 distinct byte values. The vocab
# is nonetheless locked at 256 (a deliberate design choice: a byte-level model is
# defined by covering all 256 byte values, and pruning would make it corpus-specific
# and no longer able to ingest arbitrary UTF-8). The consequence:
#
#   * lm_head is (C -> 256). The 191 columns for never-occurring bytes are trained
#     ONLY as negatives -- they receive gradient exclusively through the softmax
#     denominator, pushing them down, never up. They learn "never me," nothing more.
#   * The token embedding table wte is (256, C). Its 191 unused rows receive NO
#     gradient at all: nn.Embedding only backprops into rows that were indexed, and
#     those byte values never appear in any batch. They keep their N(0, 0.02) init
#     values for the entire run, untouched.
#
# Wasted capacity, in parameters:
#     Model A: 191 * 128 (wte) + 191 * 128 (lm_head) + 191 (bias) = 49,087  (10.4% of 0.47M)
#     Model B: 191 * 384 (wte) + 191 * 384 (lm_head) + 191 (bias) = 146,879 ( 1.3% of 10.9M)
#
# So the waste is real but shrinks with scale, and it is almost entirely embedding
# table, not compute: the dead rows cost memory, not FLOPs, because the model never
# looks them up. The dead lm_head columns DO cost FLOPs (every forward pass computes
# all 256 logits) but that is a rounding error next to attention.
#
# The interesting downstream fact -- see evaluate.py -- is that a model which learns
# nothing but "these 191 bytes never occur" already scores ln(65) = 4.17 nats,
# versus the ln(256) = 5.545 it starts at. Roughly a quarter of the drop off the
# initial loss is that trivial discovery, not language.

# --------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, written out.

    q, k, v are three separate nn.Linear layers rather than one fused (C -> 3C)
    projection. Fusing is marginally faster; keeping them separate is the version
    you can derive on a whiteboard, and the arithmetic is identical.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_size = cfg.head_size

        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)  # C -> C
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)  # C -> C
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)  # C -> C
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)  # C -> C

        self.attn_dropout = nn.Dropout(cfg.dropout)  # on the attention weights
        self.resid_dropout = nn.Dropout(cfg.dropout)  # on the block output

        # The causal mask. Lower-triangular ones: mask[i, j] == 1 iff j <= i,
        # i.e. query position i may attend to key position j only if j is at or
        # before i. Registered as a buffer, not a Parameter: it is constant, gets
        # no gradient, but must follow the module across .to(device) and must be
        # saved/loaded with the state_dict.
        #
        # Shape (1, 1, block_size, block_size): the two leading singleton dims
        # broadcast across B and nh, so one mask serves every batch item and head.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # (B, T, C)
        nh, hs = self.n_head, self.head_size  # nh * hs == C

        # --- project to queries, keys, values -------------------------------
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, C)
        v = self.v_proj(x)  # (B, T, C)

        # --- split C into nh heads of width hs ------------------------------
        # This is the step to be able to draw from memory. The C-dim residual
        # stream is *reinterpreted* as nh independent subspaces of width hs. No
        # arithmetic happens here -- view() is a reshape, transpose() a stride
        # permutation. Both are free; they exist so the matmuls below treat the
        # head axis as a batch axis and run all nh heads in parallel.
        #
        #   (B, T, C) --view--> (B, T, nh, hs) --transpose(1,2)--> (B, nh, T, hs)
        #
        # After the transpose the last two dims are (T, hs): a per-head sequence
        # of hs-dim vectors, which is exactly what a matmul wants.
        q = q.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(1, 2)  # (B, nh, T, hs)

        # --- attention scores ------------------------------------------------
        # Every query dots with every key: (T, hs) @ (hs, T) -> (T, T), batched
        # over B and nh. att[b, h, i, j] = how much query i wants key j.
        att = q @ k.transpose(-2, -1)  # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)

        # Scale by 1/sqrt(hs). Each score is a sum of hs products of ~unit-variance
        # terms, so it has variance ~hs and std ~sqrt(hs). Un-scaled, with hs=64 the
        # scores would be O(8) and the softmax would saturate at init: one key gets
        # ~all the mass, the rest get ~0, and the gradient through softmax vanishes.
        # Dividing by sqrt(hs) restores unit variance and keeps the softmax in its
        # responsive range. This is the entire reason the term exists.
        att = att * (hs**-0.5)  # (B, nh, T, T)

        # --- causal mask ------------------------------------------------------
        # Slice to :T so the same buffer works when T < block_size (generation
        # starts with a 1-token context and grows). Positions where the mask is 0
        # are strictly-future keys; set them to -inf BEFORE the softmax so they
        # receive exactly zero probability after it (exp(-inf) == 0).
        #
        # No row is ever entirely -inf: row i always keeps at least j=i (the
        # diagonal), so every softmax row has finite mass and there is no NaN.
        # A model that could see j > i would trivially "predict" the next byte by
        # reading it, and val loss would collapse to ~0 -- the classic silent bug.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # (B, nh, T, T)

        # --- masked softmax ---------------------------------------------------
        # dim=-1 normalizes over KEYS: each query's attention over the keys sums
        # to 1. Normalizing over dim=-2 instead would be a real and subtle bug --
        # the rows would no longer be probability distributions over what to read.
        att = F.softmax(att, dim=-1)  # (B, nh, T, T), rows sum to 1
        att = self.attn_dropout(att)  # (B, nh, T, T)

        # --- gather values ----------------------------------------------------
        # Weighted average of value vectors: (T, T) @ (T, hs) -> (T, hs).
        y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        # --- re-assemble the heads --------------------------------------------
        # Exactly the inverse of the split. transpose() returns a NON-contiguous
        # view (its strides no longer match row-major order), and view() requires
        # contiguous memory -- hence .contiguous(), which forces the actual copy.
        # This is why the idiom is .transpose(1,2).contiguous().view(...) and not
        # just .view(...): without it, torch raises. Concatenating the heads back
        # into C is what lets out_proj mix information ACROSS heads; until this
        # point every head has been entirely independent.
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, nh, T, hs) -> (B, T, C)

        y = self.resid_dropout(self.out_proj(y))  # (B, T, C)
        return y  # (B, T, C)


# --------------------------------------------------------------------------
# MLP and Block
# --------------------------------------------------------------------------


class MLP(nn.Module):
    """Position-wise feed-forward: C -> 4C -> C.

    Applied independently to every position (no mixing across T -- attention is
    the only thing in the model that moves information between positions). The
    4x expansion is the standard transformer ratio: attention decides WHAT to
    read, the MLP is where the capacity to actually process it lives. It holds
    roughly 2/3 of the model's parameters.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)  # C -> 4C
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)  # 4C -> C
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)  # (B, T, C) -> (B, T, 4C)
        x = F.gelu(x)  # (B, T, 4C), elementwise -- shape unchanged
        x = self.proj(x)  # (B, T, 4C) -> (B, T, C)
        x = self.dropout(x)  # (B, T, C)
        return x  # (B, T, C)


class Block(nn.Module):
    """One transformer block: pre-LN, two residual sub-layers.

        x = x + attn(ln1(x))
        x = x + mlp (ln2(x))

    Pre-LN (normalize the INPUT of each sub-layer) rather than post-LN (normalize
    after the residual add). Two reasons, both of which matter here:

      1. The residual path stays an unmodified identity from input to output --
         a clean gradient highway straight to the embeddings. Post-LN puts a
         LayerNorm on that path, and deep post-LN transformers need a careful
         warmup schedule just to survive the first few hundred steps.
      2. It is what makes the ln(256) initial-loss check meaningful: with pre-LN
         and a small-init output projection, the block is near-identity at init,
         so the logits arrive at lm_head essentially untouched and come out ~0.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)  # normalizes over C
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)  # normalizes over C
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm normalizes over the LAST dim (C) only -- per position, per
        # batch item, independently. It never mixes across T or B, so it cannot
        # leak future information into the present. (BatchNorm, which normalizes
        # across the batch, would be a causality hazard here.)
        x = x + self.attn(self.ln1(x))  # (B, T, C)
        x = x + self.mlp(self.ln2(x))  # (B, T, C)
        return x  # (B, T, C)


# --------------------------------------------------------------------------
# GPT
# --------------------------------------------------------------------------


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # (V, C) token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)  # (T_max, C) position embeddings
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=cfg.bias)  # C -> V

        self.apply(self._init_weights)

        # Scaled init for the residual output projections. Each of the n_layer
        # blocks adds TWO contributions into the residual stream (attn + mlp), and
        # the variance of a sum of independent terms adds. Left alone, activation
        # variance grows linearly with depth and the deep model starts out badly
        # conditioned. Scaling those projections by 1/sqrt(2 * n_layer) keeps the
        # stream's variance roughly constant with depth -- so Model B (6 layers) is
        # as well-behaved at step 0 as Model A (2 layers).
        # Match ONLY the two projections that write into the residual stream:
        # attn.out_proj and mlp.proj. Note the trap -- a naive endswith("proj.weight")
        # would also catch q_proj / k_proj / v_proj, which live INSIDE the block and
        # must keep the plain 0.02 init. Shrinking those would weaken attention at
        # init for no reason, and it would fail silently.
        n_scaled = 0
        for name, p in self.named_parameters():
            if name.endswith("attn.out_proj.weight") or name.endswith("mlp.proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layer) ** 0.5)
                n_scaled += 1
        assert n_scaled == 2 * cfg.n_layer, (
            f"expected to scale 2 projections per block ({2 * cfg.n_layer}), got {n_scaled}"
        )

        # Width-aware init for the output head.
        #
        # A logit is a sum of C terms w_i * h_i. ln_f guarantees h has unit
        # variance, so with a FIXED std_w the logit std comes out as
        #
        #     std(logit) = std_w * sqrt(C)
        #
        # -- it grows with the width of the model. Measured, at std_w = 0.02:
        #
        #     Model A (C=128): predicted 0.2263, measured 0.2273
        #     Model B (C=384): predicted 0.3919, measured 0.3907
        #
        # Bigger logits mean a softmax further from uniform at init, which means an
        # initial loss ABOVE ln(256): B started at 5.668 rather than 5.545. That is
        # not a bug -- it is what a width-independent init constant costs you -- but
        # it makes "initial loss == ln(256)" a width-dependent test, which would flag
        # any sufficiently wide model as broken. Since that check is our primary
        # correctness gate, we want it sharp.
        #
        # So scale the head's init by 1/sqrt(C), cancelling the sqrt(C) growth and
        # holding the logit scale constant across widths. Both models now start at
        # ln(256) to three decimals, and the gate tolerance can be 0.01 instead of 0.10.
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02 / math.sqrt(cfg.n_embd))
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

    def _init_weights(self, module: nn.Module) -> None:
        # N(0, 0.02) throughout -- small, so that at init the logits are ~0, the
        # softmax is ~uniform over 256, and cross-entropy is ~ln(256) = 5.545.
        # That is not a coincidence to be checked after the fact; it is the
        # property the init is CHOSEN to have, and sanity.py asserts it. If the
        # measured initial loss is much higher, the init is too large (saturated,
        # confidently wrong logits); if it is much lower, something is leaking.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # nn.LayerNorm defaults (weight=1, bias=0) are already what we want.

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx (B, T) int64 -> logits (B, T, V), and loss if targets given."""
        B, T = idx.shape  # (B, T)
        assert T <= self.cfg.block_size, (
            f"sequence length {T} exceeds block_size {self.cfg.block_size}; "
            f"the position embedding table has no row for position {T - 1}."
        )

        pos = torch.arange(T, device=idx.device)  # (T,)

        tok = self.wte(idx)  # (B, T) -> (B, T, C)   one embedding row per byte
        pos = self.wpe(pos)  # (T,)   -> (T, C)      one row per position

        # Broadcast: (B, T, C) + (T, C) -> (B, T, C). The position embedding is
        # the ONLY thing that tells the model about order -- attention itself is
        # permutation-equivariant. Strip this line out and the model becomes a
        # bag of bytes: it can still learn byte frequencies, but not that 'q' is
        # followed by 'u'. Adding (not concatenating) is what lets the residual
        # stream carry "which byte" and "which position" in one C-dim vector.
        x = self.drop(tok + pos)  # (B, T, C)

        for block in self.blocks:
            x = block(x)  # (B, T, C) -> (B, T, C), shape invariant all the way down

        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, C) -> (B, T, V)

        loss = None
        if targets is not None:
            # F.cross_entropy wants (N, V) logits against (N,) integer targets, so
            # flatten batch and time into one axis of N = B*T independent
            # predictions. The model makes a prediction at EVERY position, not
            # just the last one -- that is what makes training efficient.
            #
            # .view(-1, V) and .view(-1) must flatten in the SAME order (they do:
            # both are row-major over (B, T)), so logits[b*T + t] lines up with
            # targets[b*T + t]. Get this wrong -- e.g. flatten one transposed --
            # and every prediction is scored against another batch item's target.
            # The loss still decreases. It just decreases toward nonsense.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, V)
                targets.view(-1),  # (B*T,)
            )  # scalar: mean NLL in nats, averaged over all B*T positions

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. idx (B, T0) -> (B, T0 + max_new_tokens).

        Sampling, not argmax: greedy decoding on a byte-level model collapses into
        a loop almost immediately (it is the degenerate-repetition failure that
        evaluate.py measures rather than asserts).
        """
        assert temperature > 0, "temperature must be > 0 (it divides the logits)"

        # Dropout must be off, or every sampled byte is drawn from a randomly
        # perturbed model. Save and restore the mode so the caller's model is
        # unchanged on the way out.
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to the last block_size tokens. The position embedding table
            # only HAS block_size rows, so a longer context has nowhere to sit.
            # This is the model's hard memory limit: past this, the earliest
            # bytes are simply gone. It is the concrete reason A (64 bytes) cannot
            # hold a speaker name across four lines and B (256) can.
            idx_cond = idx[:, -self.cfg.block_size :]  # (B, T<=block_size)

            logits, _ = self(idx_cond)  # (B, T, V)

            # Only the LAST position's logits predict the next byte. The other T-1
            # predictions are re-computed every step and thrown away -- that is the
            # O(T^2) waste a KV-cache exists to remove. We keep it simple here.
            logits = logits[:, -1, :]  # (B, T, V) -> (B, V)

            # Temperature divides the logits BEFORE the softmax. <1 sharpens the
            # distribution (more confident, more repetitive); >1 flattens it (more
            # surprising, more misspellings). It is a monotone rescale, so it never
            # changes the RANKING of candidates, only how much mass the leader keeps.
            logits = logits / temperature  # (B, V)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                # Keep the k highest logits, set the rest to -inf so softmax gives
                # them exactly zero mass. This is what stops the long tail of ~250
                # implausible bytes from collectively stealing probability: each is
                # individually near-zero, but summed they are not, and without top-k
                # the model occasionally emits a byte it considers absurd.
                v, _ = torch.topk(logits, k)  # (B, k), sorted descending
                threshold = v[:, [-1]]  # (B, 1) the k-th largest, per row
                logits = logits.masked_fill(logits < threshold, float("-inf"))  # (B, V)

            probs = F.softmax(logits, dim=-1)  # (B, V), sums to 1
            next_tok = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_tok), dim=1)  # (B, T) -> (B, T+1)

        self.train(was_training)
        return idx  # (B, T0 + max_new_tokens)


# --------------------------------------------------------------------------
# Shape trace / parameter breakdown:  python model.py
# --------------------------------------------------------------------------


def _param_table(cfg: GPTConfig, model: GPT) -> None:
    C, V, L = cfg.n_embd, cfg.vocab_size, cfg.n_layer
    wte = V * C
    wpe = cfg.block_size * C
    attn = L * 4 * (C * C + C)
    mlp = L * ((C * 4 * C + 4 * C) + (4 * C * C + C))
    lns = L * 2 * 2 * C + 2 * C
    head = C * V + V
    print(f"  token embedding  wte  ({V}, {C})".ljust(42) + f"{wte:>12,}")
    print(f"  pos embedding    wpe  ({cfg.block_size}, {C})".ljust(42) + f"{wpe:>12,}")
    print(f"  attention        x{L}".ljust(42) + f"{attn:>12,}")
    print(f"  mlp              x{L}".ljust(42) + f"{mlp:>12,}")
    print(f"  layernorms".ljust(42) + f"{lns:>12,}")
    print(f"  lm_head          ({C}, {V})".ljust(42) + f"{head:>12,}")
    print("  " + "-" * 40)
    total = model.num_params()
    print("  TOTAL".ljust(42) + f"{total:>12,}")
    assert wte + wpe + attn + mlp + lns + head == total, "hand count != torch count"
    dead = 191 * C * 2 + 191
    print(f"  (of which dead-byte capacity: {dead:,} = {dead/total:.1%})")


def main() -> None:
    torch.manual_seed(1337)
    for name, cfg in CONFIGS.items():
        model = GPT(cfg)
        print(f"\n{'=' * 62}\nModel {name}: {cfg}\n{'=' * 62}")
        print(f"head_size = n_embd / n_head = {cfg.n_embd} / {cfg.n_head} = {cfg.head_size}\n")
        _param_table(cfg, model)

        B, T = 4, cfg.block_size
        idx = torch.randint(0, cfg.vocab_size, (B, T))
        targets = torch.randint(0, cfg.vocab_size, (B, T))
        logits, loss = model(idx, targets)
        print(f"\n  forward (B={B}, T={T}):")
        print(f"    idx      {tuple(idx.shape)}")
        print(f"    logits   {tuple(logits.shape)}   (B, T, V)")
        print(f"    flattened for loss: ({B * T}, {cfg.vocab_size}) vs ({B * T},)")
        print(f"    loss     {loss.item():.4f}   (untrained, random targets)")
        print(f"    attention matrix per head: (B, nh, T, T) = "
              f"{(B, cfg.n_head, T, T)} = {B * cfg.n_head * T * T:,} entries")

        out = model.generate(idx[:1, :1], max_new_tokens=5, temperature=1.0, top_k=40)
        print(f"    generate (1,1) + 5 tokens -> {tuple(out.shape)}")


if __name__ == "__main__":
    main()
