"""Training infrastructure: tokenizer, data pipeline, and the training loop.

This file owns everything that is *not* the network itself. model.py contains
only the architecture, so that the two can be defended separately.

    python train.py --inspect-data          self-test the data pipeline
    python train.py --model A               train Model A
    python train.py --model B               train Model B
"""

import argparse
import csv
import math
import pathlib
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from model import CONFIGS, GPT, GPTConfig

# --------------------------------------------------------------------------
# Reproducibility and device
# --------------------------------------------------------------------------

DATA_PATH = pathlib.Path(__file__).parent / "data" / "tinyshakespeare.txt"

# The vocabulary is locked at 256 and is not learned from the data. This is the
# defining property of a byte-level tokenizer: every possible byte is a token,
# whether or not it appears in the corpus. tinyshakespeare is pure ASCII, so
# roughly 190 of the 256 output logits will be trained to predict "never" and
# the rest of the softmax mass concentrates on ~65 byte values.
VOCAB_SIZE = 256


def set_seed(seed: int) -> None:
    """Seed every RNG that can affect a run.

    torch.manual_seed covers CPU and (since torch 1.8) all CUDA devices. We also
    seed python's `random` and numpy because they are used for index shuffling in
    other places, and a run is only reproducible if *every* source of randomness
    is pinned -- weight init, dropout masks, and batch sampling alike.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(override: str | None = None) -> torch.device:
    """cuda -> mps -> cpu, in that order of preference.

    `override` forces a device, which exists so we can prove the CPU path actually
    runs rather than assuming it does.
    """
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Byte-level tokenizer
# --------------------------------------------------------------------------
#
# There is no training, no merge table, and no vocabulary file. A byte-level
# tokenizer is a total function from text to integers in [0, 255] and back:
#
#     "To be"  ->  [84, 111, 32, 98, 101]
#
# Properties worth being able to state out loud:
#
#   * It never emits an <UNK>. Any byte sequence whatsoever is representable,
#     because the token set *is* the set of byte values.
#   * It is lossless in the encode direction: bytes(encode(s)) == s.encode().
#   * It is NOT lossless in the decode direction for arbitrary token sequences.
#     A model can emit a byte sequence that is not valid UTF-8 (e.g. a lone
#     continuation byte 0x80-0xBF with no lead byte). That is why decode uses
#     errors="replace": a partially-generated multi-byte character becomes U+FFFD
#     rather than raising. On pure-ASCII tinyshakespeare this is rare, but it is
#     a real failure mode of byte-level generation and the reason the decoder
#     must be defensive.
#   * The cost of all this: one token = one byte = roughly one character of
#     English. Sequences are ~4x longer than a subword tokenizer would produce,
#     which is exactly why Model B needs a 4x longer context window to see a
#     comparable amount of actual text.


def encode(text: str) -> list[int]:
    """str -> list of ints in [0, 255]. UTF-8 bytes, nothing more."""
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    """list of ints in [0, 255] -> str. Invalid UTF-8 becomes U+FFFD."""
    return bytes(tokens).decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# Data: load, split, batch
# --------------------------------------------------------------------------


def load_data(
    path: pathlib.Path = DATA_PATH, val_frac: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the corpus and split it 90/10 into train and val.

    The split is a single contiguous cut, taken BEFORE any batching happens.
    That ordering is the whole point: if we batched first and split the batches,
    a window starting at byte 1_003_800 would span the boundary and put the same
    text in both sets, and val loss would be measuring memorization. Cutting the
    token stream once means no training window can ever contain a val byte.

    A contiguous cut (rather than an interleaved or random-chunk split) also
    means val is a held-out *continuous stretch of plays* -- a genuinely unseen
    region of the corpus, not scattered sentences whose neighbours were trained on.

    Returns two 1-D int64 tensors:
        train  (1003854,)   90%
        val    ( 111540,)   10%
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:  python data/download.py"
        )
    text = path.read_text(encoding="utf-8")
    tokens = encode(text)  # list[int], length == number of BYTES, not characters

    # int64 because nn.Embedding requires a Long index tensor. The data itself
    # would fit in uint8; we pay 8x the memory (9 MB instead of 1 MB) to avoid a
    # cast on every batch. At this corpus size that is free.
    data = torch.tensor(tokens, dtype=torch.long)  # (N,)

    n_train = int(len(data) * (1.0 - val_frac))
    train_data = data[:n_train]  # (N_train,)
    val_data = data[n_train:]  # (N_val,)
    return train_data, val_data


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (context, target) pairs from a token stream.

    We draw `batch_size` random start positions and take a window of
    `block_size` tokens from each. The targets are the same windows shifted
    right by exactly one position, because the model is trained to predict the
    next byte at *every* position in the window simultaneously:

        x[b, t]  is the byte at stream position i+t
        y[b, t]  is the byte at stream position i+t+1   <- the answer for x[:, :t+1]

    So a single window of length T yields T supervised predictions, not one.
    This is why the loss is averaged over B*T positions rather than B.

    The upper bound on the start index is len(data) - block_size - 1: we need
    block_size tokens for x AND one more token beyond them for the final target.
    Off-by-one here is the single most common way to corrupt a language model,
    and it fails silently -- the loss still decreases, it just decreases toward
    the wrong function.

    Passing an explicit `generator` lets the caller pin batch selection. The
    eval loop uses a fixed-seed generator so that it scores the SAME batches at
    every eval step; otherwise the val curve would jitter from batch-sampling
    noise and the train/val divergence step would move run to run.

    Returns x (B, T) and y (B, T), both int64, both on `device`.
    """
    high = len(data) - block_size - 1
    ix = torch.randint(high, (batch_size,), generator=generator)  # (B,)

    x = torch.stack([data[i : i + block_size] for i in ix])  # (B, T)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # (B, T)

    if device.type == "cuda":
        # pin_memory + non_blocking overlaps the host->device copy with compute.
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# --------------------------------------------------------------------------
# Data pipeline self-test
# --------------------------------------------------------------------------


def inspect_data() -> None:
    """Print everything needed to verify the pipeline by eye."""
    set_seed(1337)
    device = get_device()
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu:    {torch.cuda.get_device_name(0)}")

    train_data, val_data = load_data()
    total = len(train_data) + len(val_data)
    print(f"\ncorpus: {total:,} bytes = {total:,} tokens (byte-level: 1 byte = 1 token)")
    print(f"  train: {len(train_data):,}  ({len(train_data)/total:.1%})")
    print(f"  val:   {len(val_data):,}  ({len(val_data)/total:.1%})")
    print(f"  vocab: {VOCAB_SIZE} (locked)")
    print(f"  distinct byte values actually used: {len(set(train_data.tolist()))}")

    # Round-trip: the tokenizer must be exactly invertible on real text.
    sample = "To be, or not to be, that is the question:\n"
    toks = encode(sample)
    assert decode(toks) == sample, "tokenizer round-trip failed"
    print(f"\nround-trip OK")
    print(f"  text:   {sample!r}")
    print(f"  tokens: {toks[:12]} ... ({len(toks)} tokens for {len(sample)} chars)")

    # Non-ASCII demonstration: 1 character != 1 token.
    multi = "café"
    print(f"  {multi!r} -> {encode(multi)}  ({len(multi)} chars, {len(encode(multi))} tokens)")
    print("    ^ 'é' is two bytes, so the model must learn to emit BOTH or the")
    print("      decoder produces U+FFFD. This is why decode() uses errors='replace'.")

    # A batch, at Model A's shape.
    B, T = 4, 8  # small enough to print
    x, y = get_batch(train_data, block_size=T, batch_size=B, device=device)
    print(f"\nbatch at (B={B}, T={T}):  x {tuple(x.shape)} {x.dtype}   y {tuple(y.shape)} {y.dtype}")
    print(f"  x[0] = {x[0].tolist()}  -> {decode(x[0].tolist())!r}")
    print(f"  y[0] = {y[0].tolist()}  -> {decode(y[0].tolist())!r}")
    print("  y is x shifted left by one: y[0][:-1] == x[0][1:]")
    assert torch.equal(y[0][:-1], x[0][1:]), "target alignment is wrong"
    print("  target alignment OK")

    # The leakage check that matters: no training window can reach into val.
    print(f"\nsplit integrity: train ends at byte {len(train_data):,}, val begins there.")
    print("  Max start index for a training window is len(train) - T - 1, so the")
    print("  furthest byte any training window can touch is the last train byte.")


# --------------------------------------------------------------------------
# Training hyperparameters
# --------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Stated, not silent. Every value here has a reason attached.

    These are stored in the checkpoint alongside the GPTConfig, so a run is fully
    described by its checkpoint -- no lab-notebook archaeology required.
    """

    model: str  # "A" or "B"
    steps: int
    batch_size: int
    learning_rate: float

    # AdamW. betas=(0.9, 0.95) rather than torch's default (0.9, 0.999): the
    # shorter second-moment memory is standard for transformers because gradient
    # scale changes quickly early in training, and 0.999 averages over ~1000 steps,
    # which is a large fraction of a run this short.
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Linear warmup then cosine decay to 10% of peak.
    #   warmup: Adam's second-moment estimate v is initialized at 0 and is
    #     near-meaningless for the first few dozen steps. Taking full-size steps
    #     while the preconditioner is garbage is how runs blow up in the first 50
    #     steps. 100 steps of warmup is cheap insurance.
    #   cosine: large steps early to travel, small steps late to settle. Decaying
    #     to a 10% floor rather than 0 keeps the model still learning at the end
    #     instead of freezing.
    warmup_steps: int = 100
    min_lr_frac: float = 0.1

    eval_interval: int = 100  # resolution of the train/val divergence analysis

    # eval_iters = 50, and this number was MEASURED, not inherited from folklore.
    #
    # The loss estimate is a Monte-Carlo mean over `eval_iters` batches, so its
    # standard error is s/sqrt(n) where s is the per-batch loss std. Measured on a
    # TRAINED Model A checkpoint: s = 0.026 (train), 0.027 (val).
    #
    #   n=25  -> SE 0.0052     n=100 -> SE 0.0026
    #   n=50  -> SE 0.0037     n=200 -> SE 0.0018
    #
    # The quantity that actually has to be resolved is the GAP (val - train), a
    # difference of two means, so its error is sqrt(SE_train^2 + SE_val^2):
    # at n=50 that is 0.005 nats, i.e. 10x smaller than the 0.05-nat threshold used
    # to declare divergence onset. n=50 is therefore ample, and n=200 buys a
    # sharper number than the analysis can use, at 4x the cost -- it would have
    # made evaluation MORE expensive than training for Model B (11.7 min vs 9.8).
    #
    # Critically, s must be measured on a trained model. On an UNTRAINED one it is
    # 0.003 -- ~8x smaller -- because a uniform model scores every batch almost
    # identically; batch-to-batch difficulty only becomes visible once the model
    # has learned something. Calibrating at init would have under-provisioned by
    # two orders of magnitude.
    #
    # What the fixed-seed generator does and does NOT buy (stated carefully, because
    # the tempting overclaim is wrong):
    #
    #   IT DOES remove step-to-step jitter from batch reshuffling. The same 50
    #   batches are scored at every eval, so a change in the curve between step k
    #   and step k+100 reflects the MODEL changing, not the sample changing. That is
    #   what makes the divergence-onset step robust: the val - train gap is a
    #   comparison of the same two fixed subsamples throughout the run, so it is not
    #   contaminated by reshuffling.
    #
    #   IT DOES NOT make the curve "noise-free" or reduce the sampling bias to a
    #   constant offset. The bias of a fixed 50-batch subsample is SLOWLY VARYING,
    #   not fixed: which batches are relatively hard depends on what the model has
    #   learned, and that changes over the run. So the offset between our estimate
    #   and the true population loss drifts.
    #
    # Hence n still matters for the LEVEL, and the headline loss/perplexity numbers
    # do not come from here at all -- they come from a deterministic sweep over the
    # entire val split in evaluate.py, which has no sampling error whatsoever.
    eval_iters: int = 50
    seed: int = 1337

    @property
    def tokens_per_step(self) -> int:
        """The number that makes A and B comparable -- and the reason the CSV
        logs cumulative tokens. A: 64*64 = 4,096. B: 32*256 = 8,192. At step k,
        B has consumed TWICE the text A has. Plotting loss against step alone
        silently compares two models on different diets."""
        return self.batch_size * CONFIGS[self.model].block_size


# Defaults chosen per model. B gets half the batch size because its context is 4x
# longer: the attention matrix is (B, nh, T, T), so its memory grows with B*T^2.
# Halving B against a 4x T still leaves attention 8x larger -- see the measured
# VRAM number from sanity.py. Per the brief, if memory is tight we cut batch size
# further and never touch block_size: the 256-byte window is the point of Model B.
DEFAULTS: dict[str, dict] = {
    # lr 1e-3: a 0.47M-param model is robust to a large step and would waste the
    # run at 3e-4.
    "A": dict(steps=5000, batch_size=64, learning_rate=1e-3),
    # lr 3e-4: the stable band for a ~10M transformer. 1e-3 at this depth tends to
    # spike the loss out of warmup.
    "B": dict(steps=5000, batch_size=32, learning_rate=3e-4),
}


def get_lr(step: int, tc: TrainConfig) -> float:
    """Linear warmup -> cosine decay -> 10% floor."""
    if step < tc.warmup_steps:
        return tc.learning_rate * (step + 1) / tc.warmup_steps
    if step >= tc.steps:
        return tc.learning_rate * tc.min_lr_frac
    progress = (step - tc.warmup_steps) / max(1, tc.steps - tc.warmup_steps)  # 0 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
    return tc.learning_rate * (tc.min_lr_frac + (1 - tc.min_lr_frac) * coeff)


def configure_optimizer(model: GPT, tc: TrainConfig) -> torch.optim.AdamW:
    """AdamW with two parameter groups: decay and no-decay.

    Decay: the 2-D weight matrices (Linear layers). Shrinking them toward zero is
    a genuine capacity constraint and it is what weight decay is for.

    NO decay: biases, LayerNorm gains, AND the embedding tables. The embeddings
    are the subtle one, and it matters for a byte-level model specifically:

        AdamW's decay is DECOUPLED -- it applies  p -= lr * wd * p  to every
        parameter in the group on every step, independent of the gradient. The
        token embedding wte is (256, C), but only 65 of those rows are ever
        indexed by real data; nn.Embedding's backward produces a dense gradient
        whose other 191 rows are exactly ZERO. Those rows are therefore NOT
        skipped by the optimizer (their grad is a zero tensor, not None) -- so if
        wte were in the decay group, weight decay would quietly shrink the 191
        dead byte-embeddings toward zero over the run, purely as an optimizer
        artifact, with no gradient signal ever touching them.

    Excluding embeddings from decay keeps the claim in model.py exactly true: the
    dead rows are untouched from init to the end of training, and if you inspect
    them afterwards they still hold their N(0, 0.02) draw. That is a much cleaner
    thing to say in a write-up than "they shrank a bit, for reasons unrelated to
    learning."
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and not name.startswith(("wte", "wpe")):
            decay.append(p)
        else:
            no_decay.append(p)

    groups = [
        {"params": decay, "weight_decay": tc.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=tc.learning_rate, betas=(tc.beta1, tc.beta2))


# --------------------------------------------------------------------------
# Loss estimation -- the number my evaluation component is built on
# --------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss(
    model: GPT,
    splits: dict[str, torch.Tensor],
    tc: TrainConfig,
    device: torch.device,
    eval_iters: int | None = None,
) -> dict[str, float]:
    """Mean loss on train and val, measured identically for both.

    Three decisions here, all of which exist to keep the train/val GAP honest --
    the gap is the headline finding, so anything that biases one side and not the
    other would fabricate the result:

    1. model.eval() -- DROPOUT OFF for both. The live minibatch loss from the
       optimizer step is measured with dropout ON, which inflates it: the model is
       being scored while randomly lobotomized. If we logged that as train_loss and
       compared it to a val_loss measured with dropout off, the "gap" would be
       partly a regularization artifact, and at dropout=0.1 that artifact runs in
       the WRONG DIRECTION -- it makes train look worse, masking real overfitting
       and pushing the apparent divergence step later than the truth. So train_loss
       here is a fresh no_grad pass over held-out train batches, not the training
       loss the optimizer saw.

    2. torch.no_grad() -- no graph, so eval costs a fraction of the memory and
       cannot perturb the parameters.

    3. A FIXED-SEED generator -- the same eval_iters batches are drawn at every
       eval step, for every model. Re-randomizing each time would add sampling
       noise to the curve, and since we are looking for the step where a ~0.05-nat
       gap opens, a noisy estimate would move that step around from run to run.
       Fixed batches make the curve smooth and the onset step reproducible.
       (The batches ARE random -- just the same random ones every time.)
    """
    iters = eval_iters if eval_iters is not None else tc.eval_iters
    cfg = CONFIGS[tc.model]

    was_training = model.training
    model.eval()

    out: dict[str, float] = {}
    for split_name, data in splits.items():
        # Re-seeded per split, per call -> identical batches every single eval.
        gen = torch.Generator().manual_seed(tc.seed + 1)
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = get_batch(data, cfg.block_size, tc.batch_size, device, generator=gen)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split_name] = losses.mean().item()
        out[f"{split_name}_std"] = losses.std().item()  # feeds eval_iters calibration

    model.train(was_training)
    return out


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------


def train(
    tc: TrainConfig,
    out_dir: pathlib.Path,
    quiet: bool = False,
    device_override: str | None = None,
) -> pathlib.Path:
    set_seed(tc.seed)
    device = get_device(device_override)
    cfg = CONFIGS[tc.model]

    # TF32 matmuls on Ampere: same fp32 storage, reduced-precision multiply.
    # ~2x faster with no meaningful accuracy cost at this scale. It is a numeric
    # setting, not a prebuilt transformer abstraction.
    torch.set_float32_matmul_precision("high")

    train_data, val_data = load_data()
    splits = {"train": train_data, "val": val_data}

    model = GPT(cfg).to(device)
    optimizer = configure_optimizer(model, tc)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "log.csv"
    ckpt_path = out_dir / "ckpt.pt"

    print(f"\n=== training Model {tc.model} ===")
    print(f"device        : {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    print(f"params        : {model.num_params():,}")
    print(f"config        : {cfg}")
    print(f"steps         : {tc.steps}   batch {tc.batch_size} x block {cfg.block_size}")
    print(f"tokens/step   : {tc.tokens_per_step:,}")
    print(f"total tokens  : {tc.tokens_per_step * tc.steps:,} "
          f"({tc.tokens_per_step * tc.steps / len(train_data):.1f} epochs over the train split)")
    print(f"lr            : {tc.learning_rate:g} (warmup {tc.warmup_steps}, cosine -> {tc.min_lr_frac:.0%})")
    print()

    # CSV columns fixed from step 0. tokens_seen is here from the start because
    # adding it retroactively would mean re-running everything.
    f = csv_path.open("w", newline="")
    writer = csv.writer(f)
    writer.writerow(["step", "tokens_seen", "train_loss", "val_loss", "lr", "wall_time"])

    best_val = float("inf")
    t0 = time.time()

    for step in range(tc.steps + 1):  # +1 so the final step is also evaluated
        lr = get_lr(step, tc)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # --- evaluate ------------------------------------------------------
        if step % tc.eval_interval == 0 or step == tc.steps:
            losses = estimate_loss(model, splits, tc, device)
            tokens_seen = step * tc.tokens_per_step
            wall = time.time() - t0
            writer.writerow([
                step, tokens_seen,
                f"{losses['train']:.6f}", f"{losses['val']:.6f}",
                f"{lr:.6e}", f"{wall:.2f}",
            ])
            f.flush()

            if not quiet:
                gap = losses["val"] - losses["train"]
                print(
                    f"step {step:>5} | tok {tokens_seen:>10,} | "
                    f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                    f"gap {gap:+.4f} | lr {lr:.2e} | {wall:6.1f}s"
                )

            # Checkpoint on best val -- this is the early-stopping point, and for
            # an overfitting model it is NOT the final step. Keeping both matters:
            # the best-val weights are the ones worth generating from, while the
            # final weights are the ones that show what overfitting did.
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        # The config is EMBEDDED, so evaluate.py reconstructs the
                        # architecture from the checkpoint and never has to guess.
                        "model_cfg": asdict(cfg),
                        "train_cfg": asdict(tc),
                        "model_state": model.state_dict(),
                        "step": step,
                        "tokens_seen": tokens_seen,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                    },
                    ckpt_path,
                )

        if step == tc.steps:
            break

        # --- one optimizer step ---------------------------------------------
        x, y = get_batch(train_data, cfg.block_size, tc.batch_size, device)  # (B,T),(B,T)
        _, loss = model(x, y)  # scalar

        # set_to_none=True frees the grad tensors instead of zeroing them: slightly
        # faster, and it means a parameter that receives no gradient has grad=None
        # rather than a stale zero.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Clip the GLOBAL grad norm (all params as one vector) to 1.0. Language
        # models produce occasional huge-gradient batches; without the clip one bad
        # batch can knock the weights somewhere the run never recovers from. It
        # rescales, never changes direction.
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()

    f.close()
    total = time.time() - t0
    print(f"\ndone in {total/60:.1f} min. best val {best_val:.4f} -> {ckpt_path}")
    print(f"log -> {csv_path}")
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-data", action="store_true", help="self-test the data pipeline")
    parser.add_argument("--model", choices=["A", "B"], help="which config to train")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=50)  # measured; see TrainConfig
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="force a device (cuda/cpu/mps); default = auto-select")
    args = parser.parse_args()

    if args.inspect_data:
        inspect_data()
        return
    if args.model is None:
        parser.print_help()
        return

    d = DEFAULTS[args.model]
    tc = TrainConfig(
        model=args.model,
        steps=args.steps if args.steps is not None else d["steps"],
        batch_size=args.batch_size if args.batch_size is not None else d["batch_size"],
        learning_rate=args.lr if args.lr is not None else d["learning_rate"],
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        seed=args.seed,
    )
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path("runs") / f"model_{args.model}"
    train(tc, out_dir, device_override=args.device)


if __name__ == "__main__":
    main()
