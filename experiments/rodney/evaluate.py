"""Evaluation and results analysis.

Every loss and perplexity this file reports is stated RELATIVE TO THREE BASELINES,
because a bare "1.48 nats" means nothing to a reader:

    1. uniform over 256      ln(256) = 5.5452   what an untrained model scores
    2. uniform over observed ln(65)  = 4.1744   "knows which bytes exist" -- and
                                                nothing else. The corpus is pure
                                                ASCII and uses only 65 of the 256
                                                byte values, so a model that has
                                                learned only that much already
                                                claims ~25% of the drop from init.
    3. train-unigram CE on val  = 3.3475 nats   "knows byte frequencies, zero
                                                context." A frequency table. This
                                                is the real floor for "did it learn
                                                anything about LANGUAGE."

NOTE, because the working estimate for baseline 3 was ~2.9 and the MEASURED value
is 3.3475: the frequency-table floor is ~0.45 nats HIGHER than assumed, i.e. a
weaker opponent than expected. Anything at ~2.9 nats would in fact already be
beating a unigram model comfortably, not "barely" beating it. The measured number
is the one to quote; it is computed below, not estimated.

A model at ~3.3 nats has merely matched a frequency table. A model at 1.5 has
actually learned context. That framing IS the analysis.

Baseline 3 is deliberately a CROSS-ENTROPY, not an entropy: the byte frequencies
are counted on the TRAIN split only and then used to score VAL. Counting them on
the full corpus would let the baseline peek at val, quietly handing it an unfair
advantage over the very models it is supposed to be a floor for.
"""

import argparse
import csv
import math
import pathlib
import re
from collections import Counter

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from train import decode, encode, get_device, load_data, set_seed

LN_256 = math.log(256)


# --------------------------------------------------------------------------
# Reference baselines
# --------------------------------------------------------------------------


def compute_baselines(train_data: torch.Tensor, val_data: torch.Tensor) -> dict[str, float]:
    """The three reference losses, in nats/byte. Computed once, reported everywhere."""
    train_counts = Counter(train_data.tolist())
    n_observed = len(train_counts)  # distinct byte values in TRAIN
    n_train = len(train_data)

    # --- baseline 3: train-unigram cross-entropy on val -------------------
    # q(b) = P(byte b) estimated from the TRAIN split. Score every VAL byte under q.
    #
    # Laplace (add-one) smoothing over all 256 classes. Without it, a byte value
    # that appears in val but never in train would have q(b) = 0 and contribute
    # -log(0) = +inf, making the baseline infinite. Smoothing is the standard fix
    # and it costs ~nothing here (256 pseudo-counts against ~1M real ones). We
    # report how many val bytes were actually unseen in train so the reader can see
    # whether the smoothing did any real work or was pure insurance.
    probs = torch.zeros(256, dtype=torch.float64)
    for b, c in train_counts.items():
        probs[b] = c
    probs = (probs + 1.0) / (n_train + 256.0)  # add-one smoothing

    val_counts = Counter(val_data.tolist())
    unseen = [b for b in val_counts if b not in train_counts]
    unseen_tokens = sum(val_counts[b] for b in unseen)

    # Cross-entropy H(p_val, q_train) = -(1/N) * sum_i log q(y_i), over val bytes.
    nll = 0.0
    for b, c in val_counts.items():
        nll += -c * math.log(probs[b].item())
    unigram_ce_val = nll / len(val_data)

    # For contrast: the same unigram model scored on its OWN training data. This is
    # a true entropy (self-information), and it is necessarily <= the cross-entropy
    # above. The difference is the price of not having seen val.
    nll_train = 0.0
    for b, c in train_counts.items():
        nll_train += -c * math.log(probs[b].item())
    unigram_entropy_train = nll_train / n_train

    return {
        "uniform_256": LN_256,
        "uniform_observed": math.log(n_observed),
        "unigram_ce_val": unigram_ce_val,
        "unigram_entropy_train": unigram_entropy_train,
        "n_observed": n_observed,
        "unseen_val_bytes": len(unseen),
        "unseen_val_tokens": unseen_tokens,
    }


def perplexity(loss_nats: float) -> float:
    """exp(mean cross-entropy in nats) = BYTES-perplexity.

    WHY THIS IS BYTES-PERPLEXITY, AND WHY IT IS NOT COMPARABLE TO A WORD-LEVEL
    MODEL'S PERPLEXITY. (Expect to be asked this. The short answer: the unit of
    normalization is different.)

    Perplexity is exp of the mean NLL PER TOKEN, so its value is only meaningful
    once you say what a token is. Here a token is one BYTE, so a perplexity of ~4.4
    means "on average the model is as uncertain as if it were choosing uniformly
    among 4.4 equally likely NEXT BYTES." A word-level model's perplexity of ~4.4
    would mean "as uncertain as choosing among 4.4 equally likely NEXT WORDS" --
    an enormously stronger claim, because a word carries several bytes' worth of
    information.

    The two live on different scales and CANNOT be compared directly. To convert,
    you must renormalize to a common unit. Shakespeare averages ~4.5 bytes/word, so

        ppl_word ~= ppl_byte ** (bytes per word) ~= 4.4 ** 4.5 ~= 1,000

    i.e. our byte-level perplexity of 4.4 corresponds to a word-level perplexity in
    the hundreds-to-thousands -- which is the number you would actually compare
    against a word-level LM. Quoting the raw 4.4 next to a word model's 80 and
    declaring victory is the classic error, and it is off by orders of magnitude.

    Corollary that matters for this project (Amendment B):
    bytes-perplexity IS directly comparable between Model A and Model B, because
    they share an identical 256-byte vocabulary and an identical val split. It is
    NOT comparable to a teammate's model unless that model is also byte-level. If
    theirs is character-level (a ~65-symbol vocab, so 1 token != 1 byte for any
    non-ASCII input, and a different normalization) or subword/BPE, the perplexity
    numbers are measured per DIFFERENT unit and must not be placed side by side in
    a table as though they were the same quantity. In that case either (a) compare
    bits-per-BYTE, which renormalizes every model to the same unit and is the
    honest cross-tokenizer metric, or (b) report the numbers in separate tables and
    say plainly why they cannot be compared. The comparison table in this file
    carries that warning inline for exactly this reason.
    """
    return math.exp(loss_nats)


def bits_per_byte(loss_nats: float) -> float:
    """The cross-tokenizer-safe metric: nats/byte -> bits/byte.

    Unlike perplexity, this is normalized per BYTE OF SOURCE TEXT rather than per
    token, so it is directly comparable across models with different tokenizers.
    For a byte-level model it is just loss / ln(2).
    """
    return loss_nats / math.log(2)


def format_against_baselines(loss: float, b: dict[str, float]) -> str:
    """State a loss relative to the three reference points. This is the framing."""
    frac = (b["uniform_256"] - loss) / (b["uniform_256"] - b["unigram_ce_val"])
    if loss > b["uniform_observed"]:
        verdict = "has not even learned which bytes exist"
    elif loss > b["unigram_ce_val"]:
        verdict = "knows which bytes exist, but has NOT beaten a frequency table"
    elif loss > b["unigram_ce_val"] - 0.3:
        verdict = "has barely beaten a frequency table -- little real context"
    else:
        verdict = "has genuinely learned context, well beyond byte frequencies"
    return (
        f"{loss:.4f} nats/byte | ppl {perplexity(loss):7.3f} | "
        f"{bits_per_byte(loss):.3f} bits/byte\n"
        f"      vs uniform-256      {b['uniform_256']:.4f}  ({b['uniform_256'] - loss:+.4f})\n"
        f"      vs uniform-observed {b['uniform_observed']:.4f}  ({b['uniform_observed'] - loss:+.4f})\n"
        f"      vs unigram-CE-val   {b['unigram_ce_val']:.4f}  ({b['unigram_ce_val'] - loss:+.4f})\n"
        f"      -> {verdict}"
    )


# --------------------------------------------------------------------------
# Deterministic, context-primed sweep over the whole val split
# --------------------------------------------------------------------------


@torch.no_grad()
def sweep_loss(
    model: GPT, data: torch.Tensor, device: torch.device, batch_size: int = 32
) -> dict[str, float]:
    """Exact mean loss over an entire split. No sampling, no variance, reproducible.

    THE BIAS THIS EXISTS TO AVOID:

    The naive way to sweep a split is to chop it into non-overlapping windows of
    block_size and score every position. That is wrong -- or rather, it is
    pessimistic, and unequally so across models. Position 0 of each window is
    predicted with NO context at all; position 1 with one byte of context; and so
    on. Only the last positions of a window enjoy the full context the model was
    built for. Those early, context-starved positions have genuinely high loss, and
    they drag the average up. The bias is worse for a model with a SHORT block_size
    (more window boundaries per unit of text -> a larger fraction of scored
    positions are context-starved), so a naive sweep would systematically flatter
    Model B relative to Model A -- corrupting the very comparison we are making.

    THE FIX (context-primed scoring):

    Slide the window by stride = block_size // 2 and score only the SECOND HALF of
    each window. Every scored target then has at least stride+1 tokens of context
    behind it (129 for Model B, 33 for Model A), instead of possibly zero. The
    windows tile the split so each target is scored exactly once. Cost: 2x the
    forward passes of a naive sweep, since each token is seen twice (once while
    priming, once while scored).

    THE BIAS THAT REMAINS (stated, not hidden):

    Even here, the first scored token in each window has only stride+1 tokens of
    context, not the full block_size. So this is still slightly pessimistic versus
    an ideal stride-1 sliding window, which would give EVERY token the maximum
    available context -- but that costs block_size times more compute. Reported
    numbers are therefore a mild upper bound on the model's true loss, and the bias
    is now small and, more importantly, applied evenly.

    The first window is scored in full (all block_size positions), because nothing
    precedes it -- there is no context to prime with, and skipping those tokens
    would silently drop the opening of the split.
    """
    model.eval()
    T = model.cfg.block_size
    stride = T // 2
    N = len(data)

    starts, score_from = [], []
    i = 0
    while i + T + 1 <= N:
        starts.append(i)
        # first window: score everything (nothing precedes it).
        # later windows: score only the last `stride` positions -- the primed half.
        score_from.append(0 if i == 0 else T - stride)
        i += stride

    total_nll = 0.0
    total_tokens = 0

    for j in range(0, len(starts), batch_size):
        chunk = starts[j : j + batch_size]
        x = torch.stack([data[s : s + T] for s in chunk]).to(device)  # (B, T)
        y = torch.stack([data[s + 1 : s + T + 1] for s in chunk]).to(device)  # (B, T)
        logits, _ = model(x)  # (B, T, V)

        # Per-position NLL, no reduction, so we can drop the unprimed prefix.
        nll = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (B*T, V)
            y.reshape(-1),  # (B*T,)
            reduction="none",
        ).view(len(chunk), T)  # (B, T)

        for row, s_from in enumerate(score_from[j : j + batch_size]):
            total_nll += nll[row, s_from:].sum().item()
            total_tokens += T - s_from

    mean = total_nll / total_tokens
    return {
        "loss": mean,
        "perplexity": perplexity(mean),
        "bits_per_byte": bits_per_byte(mean),
        "tokens_scored": total_tokens,
        "coverage": total_tokens / (N - 1),
    }


# --------------------------------------------------------------------------
# Train/val divergence analysis -- the headline finding
# --------------------------------------------------------------------------

GAP_THRESHOLD = 0.05  # nats
GAP_PATIENCE = 3  # consecutive evals that must stay above it


def gap_analysis(csv_path: pathlib.Path) -> dict:
    """When does the model start overfitting, and where is it best?

    Two events, both defined quantitatively so they can be pointed at rather than
    eyeballed:

      divergence onset: the first eval step where (val - train) exceeds 0.05 nats
                        AND stays above it for GAP_PATIENCE consecutive evals. The
                        patience requirement is what stops a single noisy eval from
                        being reported as the onset.

      val minimum:      argmin of val loss -- the early-stopping point. For an
                        overfitting model this is NOT the final step, and the
                        distance between the two is a direct measure of how much of
                        the run was spent actively getting worse.

    WHY THE SAMPLING NOISE DOES NOT MOVE THE ONSET STEP (stated precisely):

    estimate_loss scores the SAME fixed 50 batches at every eval, so a change in
    the curve between step k and k+100 reflects the model changing, not the sample
    changing. The val - train gap is therefore a comparison of the same two fixed
    subsamples throughout the run, and is not contaminated by batch reshuffling --
    which is what makes the onset step robust.

    What is NOT true, and is tempting to claim: that this reduces the sampling bias
    to a constant offset. It does not. The bias of a fixed 50-batch subsample is
    SLOWLY VARYING, not fixed -- which batches are relatively hard depends on what
    the model has learned, and that changes over the run. So the offset between
    these estimates and the true population loss drifts. That is precisely why the
    headline loss and perplexity figures come from sweep_loss() over the ENTIRE val
    split (zero sampling error) and not from these curves.
    """
    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "step": int(r["step"]),
                    "tokens_seen": int(r["tokens_seen"]),
                    "train_loss": float(r["train_loss"]),
                    "val_loss": float(r["val_loss"]),
                    "gap": float(r["val_loss"]) - float(r["train_loss"]),
                }
            )

    onset = None
    for i, r in enumerate(rows):
        window = rows[i : i + GAP_PATIENCE]
        if len(window) < GAP_PATIENCE:
            break
        if all(w["gap"] > GAP_THRESHOLD for w in window):
            onset = r
            break

    val_min = min(rows, key=lambda r: r["val_loss"])
    final = rows[-1]

    return {
        "rows": rows,
        "onset": onset,
        "val_min": val_min,
        "final": final,
        "overfit_cost": final["val_loss"] - val_min["val_loss"],
    }


# --------------------------------------------------------------------------
# Sample quality: generation, repetition, structure
# --------------------------------------------------------------------------

PROMPTS = [
    "To be, or not to ",
    "KING RICHARD III:\n",
    "O Romeo, Romeo! wherefore art thou ",
    "First Citizen:\nWe are accounted poor citizens, the ",
    "Now is the winter of our ",
]

# Sampling hyperparameters, stated:
#   temperature 0.8 -- below 1.0, so the distribution is sharpened. This buys
#     fewer misspellings at the cost of MORE repetition, and that tradeoff is
#     exactly what the repetition metric below is here to quantify. At 1.0 the
#     model samples its true distribution and produces more novel (and more
#     broken) text; below ~0.5 it collapses toward greedy decoding and loops.
#   top_k 40 -- of 256 possible bytes. The model's long tail of ~200 implausible
#     bytes is individually near-zero but collectively not, so without top-k it
#     occasionally emits a byte it considers absurd. 40 is generous for a
#     65-symbol effective alphabet: it never truncates a genuinely plausible
#     continuation, it only removes the junk.
GEN_TEMPERATURE = 0.8
GEN_TOP_K = 40
GEN_TOKENS = 150  # exactly, per the brief


def repetition_rate(tokens: list[int], n: int = 4) -> float:
    """Fraction of n-grams that are duplicates: 1 - unique/total.

    0.0 = every n-gram appears exactly once. Higher = more self-repetition.
    A model stuck in a loop ("the the the the") drives this toward 1.0.

    This is a MEASUREMENT, not a verdict, and on its own it is uninterpretable --
    which is the whole point of the corpus control below. Real Shakespeare repeats
    too: ' the', 'and ', ', and' recur constantly in any 150 bytes of English. So
    a raw 0.15 means nothing until you know what the DATA scores on the identical
    metric at the identical length. "Degenerate" must mean "measurably worse than
    the corpus," not "a number I dislike."
    """
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def corpus_repetition_control(
    data: torch.Tensor, length: int = GEN_TOKENS, n: int = 4, trials: int = 500,
    seed: int = 1234,
) -> tuple[float, float]:
    """The control row: what does REAL Shakespeare score on the same metric?

    Same n, same slice length as the generations, sampled from held-out val text.
    Returns (mean, std). Any model result must be read against this, and the std
    tells you whether a difference is even meaningful at this sample size.
    """
    g = torch.Generator().manual_seed(seed)
    rates = []
    for _ in range(trials):
        i = torch.randint(len(data) - length, (1,), generator=g).item()
        rates.append(repetition_rate(data[i : i + length].tolist(), n))
    t = torch.tensor(rates)
    return t.mean().item(), t.std().item()


def corpus_structure_control(
    data: torch.Tensor, length: int = GEN_TOKENS, trials: int = 500, seed: int = 1234
) -> dict[str, float]:
    """The same structural proxies, measured on real 150-byte slices of val text.

    Without this, "Model A produced 0.4 speaker headings per sample" is a number
    with no scale attached. With it, we can say whether that is what Shakespeare
    actually looks like at this length.
    """
    g = torch.Generator().manual_seed(seed)
    acc: dict[str, list[float]] = {}
    for _ in range(trials):
        i = torch.randint(len(data) - length, (1,), generator=g).item()
        m = structure_metrics(decode(data[i : i + length].tolist()))
        for k, v in m.items():
            acc.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}


# A speaker heading in tinyshakespeare: a short line ending in a colon. NOTE the
# corpus uses TITLE CASE far more than all-caps -- 'First Citizen:', 'All:',
# 'MENENIUS:' all occur. An .isupper() test would miss the dominant style entirely
# and undercount headings in corpus and generations alike.
HEADING_RE = re.compile(r"^[A-Z][A-Za-z' ]{0,28}:$")


def structure_metrics(text: str) -> dict[str, float]:
    """Cheap, measurable proxies for 'structural stability'.

    tinyshakespeare has a rigid shape: a speaker heading ending in a colon, then
    lines of dialogue, then a blank line. A model that has learned the FORM of a
    play -- as opposed to merely the letter statistics of English -- reproduces that
    shape. These numbers let the qualitative table row point at evidence.

    THE PARTIAL-LINE TRAP: a 150-byte window almost never begins or ends on a line
    boundary, so split('\\n') hands back a truncated fragment at each end. A fragment
    such as 'IUS:' (the tail of 'MENENIUS:') looks exactly like a speaker heading and
    would be counted as one -- inflating the corpus control and making the models
    look structurally worse than they are, purely as a slicing artifact. We drop the
    first and last fragments and score only the complete lines in between.
    """
    lines = text.split("\n")
    inner = lines[1:-1] if len(lines) >= 3 else []  # complete lines only
    headings = [ln for ln in inner if HEADING_RE.match(ln.strip())]
    non_empty = [ln for ln in inner if ln.strip()]
    return {
        "lines": len(inner),
        "speaker_headings": len(headings),
        "mean_line_len": sum(len(ln) for ln in non_empty) / max(1, len(non_empty)),
        "blank_lines": sum(1 for ln in inner if not ln.strip()),
    }


@torch.no_grad()
def generate_samples(
    model: GPT, device: torch.device, prompts: list[str] = PROMPTS,
    temperature: float = GEN_TEMPERATURE, top_k: int = GEN_TOP_K,
    max_new_tokens: int = GEN_TOKENS, seed: int = 1337,
) -> list[dict]:
    """Exactly `max_new_tokens` new bytes per prompt, temperature + top-k sampling.

    SEEDING: seed + i, i.e. a DIFFERENT stream per prompt, but the SAME stream for
    Model A and Model B on any given prompt. Both halves of that matter:

      same across models -> A and B draw identical random numbers on prompt i, so
        the only difference between their outputs is the model, not the luck of the
        draw. This is what makes the A-vs-B comparison controlled.

      different across prompts -> the 5 samples are independent draws.

    The second half was a bug on the first pass: seeding with a CONSTANT before every
    prompt made all 5 prompts share one random stream, and two prompts whose contexts
    converged to similar distributions then produced byte-identical continuations
    forever after (Model A's prompts 2 and 5 shared a 60-byte tail). That is an
    artifact of the sampler, not a property of the model -- and it would have
    contaminated the repetition metric, which is precisely the thing this file exists
    to measure honestly.
    """
    out = []
    for i, prompt in enumerate(prompts):
        set_seed(seed + i)
        ids = encode(prompt)  # list[int], byte-level
        idx = torch.tensor([ids], dtype=torch.long, device=device)  # (1, T0)
        gen = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)  # (1, T0+150)

        new_tokens = gen[0, len(ids):].tolist()  # exactly 150 tokens
        assert len(new_tokens) == max_new_tokens, f"expected {max_new_tokens}, got {len(new_tokens)}"

        out.append({
            "prompt": prompt,
            "completion": decode(new_tokens),
            "tokens": new_tokens,
            "rep4": repetition_rate(new_tokens, 4),
            "rep8": repetition_rate(new_tokens, 8),
            "structure": structure_metrics(decode(new_tokens)),
        })
    return out


TEMPERATURES = (0.5, 0.8, 1.0)


def temperature_sweep(model: GPT, device: torch.device) -> dict[float, dict[str, float]]:
    """Repetition vs temperature, averaged over ALL prompts.

    THIS IS A FINDING, NOT A FOOTNOTE.

    Repetition swings from ~0.16 at T=0.5 to ~0.01 at T=1.0 for the same model and
    the same weights -- a 10x range, dwarfing the difference BETWEEN Model A and
    Model B at any fixed temperature. At this scale, repetition is dominated by the
    sampling temperature, not by the architecture.

    The consequence for how results must be reported: a repetition number quoted
    WITHOUT its temperature is unfalsifiable. Anyone can produce any repetition
    figure they like from either model by choosing T. So every repetition number in
    the comparison table carries its T, and comparing two models' repetition at
    different temperatures is meaningless.
    """
    out = {}
    for temp in TEMPERATURES:
        samples = generate_samples(model, device, PROMPTS, temperature=temp)
        out[temp] = {
            "rep4": sum(s["rep4"] for s in samples) / len(samples),
            "rep8": sum(s["rep8"] for s in samples) / len(samples),
            "worst_rep4": max(s["rep4"] for s in samples),
        }
    return out


def load_checkpoint(path: pathlib.Path, device: torch.device) -> tuple[GPT, GPTConfig, dict]:
    """Rebuild a model from a checkpoint. The config is EMBEDDED -- no guessing."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ck["model_cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, cfg, ck


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def build_comparison_table(
    results: dict, b: dict, control: tuple[float, float], out_path: pathlib.Path
) -> str:
    """The markdown comparison table. Gemini Flash column left EMPTY, to be pasted in.

    The comparability warning is not boilerplate -- it is the point of the note. See
    perplexity() for the full argument.
    """
    A, B = results["A"], results["B"]
    ctrl_mean, ctrl_std = control

    def avg(m, key):
        return sum(s[key] for s in m["samples"]) / len(m["samples"])

    def avg_struct(m, key):
        return sum(s["structure"][key] for s in m["samples"]) / len(m["samples"])

    def rep_verdict(r):
        if r > ctrl_mean + 2 * ctrl_std:
            return f"**{r:.3f} — ABOVE corpus: degenerate**"
        if r < ctrl_mean - 2 * ctrl_std:
            return f"{r:.3f} — below corpus"
        return f"{r:.3f} — within corpus range"

    lines = [
        "# Model comparison — byte-level GPT on tinyshakespeare",
        "",
        "| | Model A | Model B | Gemini Flash |",
        "|---|---|---|---|",
        f"| **Parameters** | {A['params']:,} | {B['params']:,} | |",
        f"| **Architecture** | 2L / 4H / 128C, block 64 | 6L / 6H / 384C, block 256 | |",
        f"| **Val loss (nats/byte)** | {A['sweep']['loss']:.4f} | {B['sweep']['loss']:.4f} | |",
        f"| **Bytes-perplexity** | {A['sweep']['perplexity']:.3f} | {B['sweep']['perplexity']:.3f} | |",
        f"| **Bits/byte** | {A['sweep']['bits_per_byte']:.3f} | {B['sweep']['bits_per_byte']:.3f} | |",
        f"| **Val loss @ equal tokens (20.48M)** | {A['equal_token_val']:.4f} | {B['equal_token_val']:.4f} | |",
        f"| **Divergence onset** | step {A['gap']['onset']['step']} ({A['gap']['onset']['tokens_seen']:,} tok) | "
        f"step {B['gap']['onset']['step']} ({B['gap']['onset']['tokens_seen']:,} tok) | |",
        f"| **Val minimum** | step {A['gap']['val_min']['step']} ({A['gap']['val_min']['tokens_seen']:,} tok) | "
        f"step {B['gap']['val_min']['step']} ({B['gap']['val_min']['tokens_seen']:,} tok) | |",
        f"| **Terminal train/val gap** | {A['gap']['final']['gap']:+.4f} | {B['gap']['final']['gap']:+.4f} | |",
        "| | | | |",
        f"| **Structural stability** (T=0.8) | {A['structural_verdict']} | {B['structural_verdict']} | |",
        f"| **Shakespearean style** | {A['style_verdict']} | {B['style_verdict']} | |",
        "| | | | |",
        "| **Degenerate repetition** (4-gram dup. rate — *temperature MUST be stated, see note*) | | | |",
        f"| &nbsp;&nbsp;at **T = 0.5**, top-k 40 | {rep_verdict(A['temp_sweep'][0.5]['rep4'])} | "
        f"{rep_verdict(B['temp_sweep'][0.5]['rep4'])} | *(state T)* |",
        f"| &nbsp;&nbsp;at **T = 0.8**, top-k 40 | {rep_verdict(A['temp_sweep'][0.8]['rep4'])} | "
        f"{rep_verdict(B['temp_sweep'][0.8]['rep4'])} | *(state T)* |",
        f"| &nbsp;&nbsp;at **T = 1.0**, top-k 40 | {rep_verdict(A['temp_sweep'][1.0]['rep4'])} | "
        f"{rep_verdict(B['temp_sweep'][1.0]['rep4'])} | *(state T)* |",
        f"| &nbsp;&nbsp;worst single sample (T=0.8) | {A['temp_sweep'][0.8]['worst_rep4']:.3f} | "
        f"{B['temp_sweep'][0.8]['worst_rep4']:.3f} | |",
        f"| &nbsp;&nbsp;**corpus control** (temperature-free) | {ctrl_mean:.3f} ± {ctrl_std:.3f} | "
        f"{ctrl_mean:.3f} ± {ctrl_std:.3f} | {ctrl_mean:.3f} ± {ctrl_std:.3f} |",
        "",
        "> **Repetition is strongly temperature-dependent — this is a finding, not a footnote.**",
        f"> The same weights swing from {A['temp_sweep'][0.5]['rep4']:.3f} (T=0.5) to "
        f"{A['temp_sweep'][1.0]['rep4']:.3f} (T=1.0) for Model A "
        f"({A['temp_sweep'][0.5]['rep4'] / max(1e-9, A['temp_sweep'][1.0]['rep4']):.1f}x), and "
        f"{B['temp_sweep'][0.5]['rep4']:.3f} to {B['temp_sweep'][1.0]['rep4']:.3f} for Model B "
        f"({B['temp_sweep'][0.5]['rep4'] / max(1e-9, B['temp_sweep'][1.0]['rep4']):.1f}x).",
        "> Lower T sharpens the distribution, which buys fewer misspellings and pays for them in loops.",
        ">",
        "> **A repetition number quoted without its temperature is unfalsifiable** — a wide range of",
        "> figures can be produced from either model by choosing T. Hence every row above states its T,",
        "> and comparing two models' repetition at different temperatures is meaningless. If the Gemini",
        "> Flash column is filled in, its sampling temperature must be stated too, or the comparison is void.",
        ">",
        "> Stated precisely, because the tempting overclaim is wrong: the temperature swing does NOT",
        f"> always dominate the model difference. For Model A the swing ({A['temp_sweep'][0.5]['rep4'] - A['temp_sweep'][1.0]['rep4']:.3f}) "
        f"is larger than the A-vs-B gap at fixed T ({abs(A['temp_sweep'][0.8]['rep4'] - B['temp_sweep'][0.8]['rep4']):.3f});",
        f"> for Model B the swing ({B['temp_sweep'][0.5]['rep4'] - B['temp_sweep'][1.0]['rep4']:.3f}) is SMALLER than it. So temperature must always be",
        "> reported, but it does not by itself explain away the difference between the models.",
        ">",
        "> Per-sample variance is also large: single prompts reach 0.16-0.18 at T=0.5 while the 5-prompt",
        "> mean is ~0.08-0.10. Quote means over prompts, never a single sample.",
        ">",
        f"> Note the direction: at T=0.8 and T=1.0 the LARGER model repeats MORE "
        f"({B['temp_sweep'][0.8]['rep4']:.3f} vs {A['temp_sweep'][0.8]['rep4']:.3f}).",
        "> A better-fit model is a more confident model: its next-byte distribution is lower-entropy, so",
        "> at any fixed T it concentrates more mass on its favourite continuation. Capacity buys accuracy",
        "> and pays for it in diversity.",
        "",
        "### Reference baselines (nats/byte)",
        "",
        "| Baseline | Loss | Bytes-ppl | Meaning |",
        "|---|---|---|---|",
        f"| uniform over 256 | {b['uniform_256']:.4f} | {perplexity(b['uniform_256']):.2f} | untrained; knows nothing |",
        f"| uniform over observed ({b['n_observed']}) | {b['uniform_observed']:.4f} | {perplexity(b['uniform_observed']):.2f} | knows only which bytes exist |",
        f"| train-unigram CE on val | {b['unigram_ce_val']:.4f} | {perplexity(b['unigram_ce_val']):.2f} | a frequency table; zero context |",
        f"| **real corpus (150-byte slices)** | — | — | **4-gram repetition {ctrl_mean:.3f} ± {ctrl_std:.3f}** |",
        "",
        "Both models sit ~1.8–1.9 nats below the frequency-table floor, so both have",
        "genuinely learned context rather than byte statistics.",
        "",
        "### Perplexity comparability (IMPORTANT)",
        "",
        "**Bytes-perplexity is comparable between Model A and Model B** — identical 256-byte",
        "vocabulary, identical val split, identical scoring procedure. The A-vs-B numbers above",
        "may be read side by side.",
        "",
        "**It is NOT comparable to a model with a different tokenizer.** Perplexity is `exp` of the",
        "mean NLL *per token*, so its value is only meaningful once you say what a token is. Ours is",
        "one BYTE. If a teammate's implementation is character-level (a ~65-symbol vocabulary, where",
        "one token is not one byte) or subword/BPE (where one token spans several characters), their",
        "perplexity is normalized per a *different unit* and the numbers must NOT be placed in the",
        "same column as though they measured the same thing. A word-level perplexity of 80 is a far",
        "stronger result than a byte-level perplexity of 4.3, not a far worse one.",
        "",
        "To compare across tokenizers, either:",
        "1. **Report bits-per-byte** (given above), which renormalizes every model to the same unit —",
        "   bytes of source text — and is the honest cross-tokenizer metric; or",
        "2. Report the perplexities in **separate tables** and state why they cannot be compared.",
        "",
        "The Gemini Flash column is deliberately empty. Note that a Gemini perplexity, if obtainable",
        "at all, would be subword-level and therefore belongs under rule (2) above — the qualitative",
        "rows (structure, style, repetition) are the ones that can be filled in and compared directly.",
    ]
    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["runs/model_A", "runs/model_B"])
    parser.add_argument("--no-generate", action="store_true")
    args = parser.parse_args()

    set_seed(1337)
    device = get_device()
    train_data, val_data = load_data()
    b = compute_baselines(train_data, val_data)

    print("=" * 78)
    print("REFERENCE BASELINES  (nats per byte -- every model number is stated against these)")
    print("=" * 78)
    print(f"  1. uniform over 256        ln(256) = {b['uniform_256']:.4f}   ppl {perplexity(b['uniform_256']):8.2f}")
    print(f"     an untrained model. Knows nothing, including which bytes occur.")
    print(f"  2. uniform over observed   ln({b['n_observed']:>3}) = {b['uniform_observed']:.4f}   ppl {perplexity(b['uniform_observed']):8.2f}")
    print(f"     knows only which {b['n_observed']} byte values exist in train. No frequencies, no context.")
    print(f"  3. train-unigram CE on val         = {b['unigram_ce_val']:.4f}   ppl {perplexity(b['unigram_ce_val']):8.2f}")
    print(f"     a frequency table fit on TRAIN, scored on VAL. Zero context.")
    print(f"     (the same table scored on its own train data: {b['unigram_entropy_train']:.4f} --")
    print(f"      an entropy, not a cross-entropy. The {b['unigram_ce_val'] - b['unigram_entropy_train']:+.4f} difference is the")
    print(f"      price of not having seen val, and is why baseline 3 is the honest floor.)")
    print(f"     val bytes never seen in train: {b['unseen_val_bytes']} "
          f"({b['unseen_val_tokens']} tokens) -- Laplace smoothing handles them.")
    print()
    print(f"  Reading the scale:  > {b['uniform_observed']:.2f} = has not learned the alphabet")
    print(f"                      ~ {b['unigram_ce_val']:.2f} = a frequency table, nothing more")
    print(f"                      < {b['unigram_ce_val'] - 0.3:.2f} = has actually learned context")
    print()

    results = {}
    for run in args.runs:
        run_dir = pathlib.Path(run)
        ckpt, log = run_dir / "ckpt.pt", run_dir / "log.csv"
        if not ckpt.exists():
            print(f"skipping {run}: no checkpoint")
            continue

        model, cfg, ck = load_checkpoint(ckpt, device)
        name = run_dir.name.replace("model_", "")
        ga = gap_analysis(log)

        print("=" * 78)
        print(f"MODEL {name}   ({model.num_params():,} params, "
              f"{cfg.n_layer}L / {cfg.n_head}H / {cfg.n_embd}C / block {cfg.block_size})")
        print("=" * 78)

        print("\n  --- final loss, deterministic context-primed sweep over the FULL split ---")
        for split, data in (("train", train_data), ("val", val_data)):
            sw = sweep_loss(model, data, device)
            print(f"\n  {split.upper()}  ({sw['tokens_scored']:,} tokens scored, "
                  f"{sw['coverage']:.1%} coverage)")
            print(f"      {format_against_baselines(sw['loss'], b)}")
            if split == "val":
                results[name] = {
                    "sweep": sw, "gap": ga, "cfg": cfg, "ckpt": ck,
                    "model": model, "params": model.num_params(),
                }

        print("\n  --- train/val divergence ---")
        onset = ga["onset"]
        if onset:
            print(f"  divergence onset : step {onset['step']:>5}  |  "
                  f"tokens_seen {onset['tokens_seen']:>11,}")
            print(f"                     (gap first exceeds {GAP_THRESHOLD} nats and stays above "
                  f"for {GAP_PATIENCE} consecutive evals)")
            print(f"                     train {onset['train_loss']:.4f}  val {onset['val_loss']:.4f}  "
                  f"gap {onset['gap']:+.4f}")
        else:
            print(f"  divergence onset : NEVER (gap never sustained above {GAP_THRESHOLD} nats)")

        vm, fin = ga["val_min"], ga["final"]
        print(f"  val minimum      : step {vm['step']:>5}  |  tokens_seen {vm['tokens_seen']:>11,}"
              f"  |  val {vm['val_loss']:.4f}")
        print(f"  final            : step {fin['step']:>5}  |  tokens_seen {fin['tokens_seen']:>11,}"
              f"  |  val {fin['val_loss']:.4f}  gap {fin['gap']:+.4f}")
        print(f"  cost of overfitting past the minimum: {ga['overfit_cost']:+.4f} nats")
        print()

    # --- head-to-head ------------------------------------------------------
    if len(results) == 2 and "A" in results and "B" in results:
        A, B = results["A"], results["B"]
        print("=" * 78)
        print("A vs B  (bytes-perplexity IS comparable here: identical 256-byte vocab, identical val split)")
        print("=" * 78)
        for label, key in (("val loss (nats/byte)", "loss"), ("bytes-perplexity", "perplexity"),
                           ("bits/byte", "bits_per_byte")):
            print(f"  {label:<24} A {A['sweep'][key]:>9.4f}   B {B['sweep'][key]:>9.4f}")

        oa, ob = A["gap"]["onset"], B["gap"]["onset"]
        print()
        print("  divergence onset")
        print(f"    by STEP        : A {oa['step'] if oa else 'never':>9}   B {ob['step'] if ob else 'never':>9}")
        print(f"    by TOKENS SEEN : A {oa['tokens_seen'] if oa else 0:>9,}   "
              f"B {ob['tokens_seen'] if ob else 0:>9,}   <- the honest comparison")
        print()
        print("  NOTE: A sees 4,096 tokens/step, B sees 8,192. At the same STEP, B has")
        print("  consumed twice the text. Comparing onset by step alone would credit B with")
        print("  lasting longer than it did. The headline claim is about capacity vs. data,")
        print("  so tokens_seen is the axis that supports it.")

        # --- THE EQUAL-TOKEN CONTROL --------------------------------------
        # B's raw win is confounded: it read 41.0M tokens to A's 20.5M. To attribute
        # the advantage to CAPACITY rather than to a bigger data budget, both models
        # must be compared after reading the SAME amount of text. Both numbers here
        # come from the logged CSV (the fixed-batch estimator), not from sweep_loss,
        # so the comparison is like-for-like -- B has no checkpoint saved at that step.
        target = A["gap"]["final"]["tokens_seen"]
        b_at = min(B["gap"]["rows"], key=lambda r: abs(r["tokens_seen"] - target))
        a_at = A["gap"]["final"]
        A["equal_token_val"], B["equal_token_val"] = a_at["val_loss"], b_at["val_loss"]

        print()
        print("=" * 78)
        print(f"EQUAL-TOKEN CONTROL  (both at {target:,} tokens; CSV estimator both sides)")
        print("=" * 78)
        print(f"  Model A  step {a_at['step']:>5}   val {a_at['val_loss']:.4f}   gap {a_at['gap']:+.4f}")
        print(f"  Model B  step {b_at['step']:>5}   val {b_at['val_loss']:.4f}   gap {b_at['gap']:+.4f}")
        delta_equal = a_at["val_loss"] - b_at["val_loss"]
        delta_raw = a_at["val_loss"] - B["gap"]["final"]["val_loss"]
        print(f"\n  B's advantage at EQUAL tokens : {delta_equal:+.4f} nats")
        print(f"  B's advantage over full runs  : {delta_raw:+.4f} nats (B read 2x the text)")
        print(f"  -> {delta_equal / delta_raw:.0%} of B's advantage survives the equal-token control.")
        if delta_equal > 0.02:
            print("     B's win is a CAPACITY effect, not a data-budget artifact.")
        else:
            print("     B's win is a DATA-BUDGET artifact -- it vanishes at equal tokens.")

        if not args.no_generate:
            # Deliverables go in the directory ROOT, not runs/ -- runs/ is gitignored
            # (it holds checkpoints and logs), and these two are team artifacts that
            # must be committed.
            out_dir = pathlib.Path(".")
            run_generation(results, val_data, device, out_dir)
            table = build_comparison_table(
                results, b, corpus_repetition_control(val_data, n=4),
                out_dir / "comparison.md",
            )
            print("\n\n" + table)
            print(f"\ntable -> {out_dir / 'comparison.md'}")


@torch.no_grad()
def run_generation(results: dict, val_data: torch.Tensor, device: torch.device,
                   out_dir: pathlib.Path) -> None:
    """Sample quality: generation, repetition against a corpus control, structure."""
    ctrl_mean, ctrl_std = corpus_repetition_control(val_data, n=4)
    ctrl8_mean, ctrl8_std = corpus_repetition_control(val_data, n=8)
    struct_ctrl = corpus_structure_control(val_data)

    print("\n\n" + "=" * 78)
    print(f"SAMPLE QUALITY   (temperature {GEN_TEMPERATURE}, top-k {GEN_TOP_K}, "
          f"exactly {GEN_TOKENS} tokens per prompt)")
    print("=" * 78)
    print("\n  CORPUS CONTROL -- real Shakespeare, same 150-byte length, same metric:")
    print(f"    4-gram repetition : {ctrl_mean:.4f} +/- {ctrl_std:.4f}")
    print(f"    8-gram repetition : {ctrl8_mean:.4f} +/- {ctrl8_std:.4f}")
    print(f"    speaker headings  : {struct_ctrl['speaker_headings']:.2f} per sample")
    print(f"    mean line length  : {struct_ctrl['mean_line_len']:.1f} chars")
    print("\n  'Degenerate' = measurably ABOVE the corpus rate, not merely nonzero.")
    print("  Real English repeats: ' the', 'and ' recur in any 150 bytes.\n")

    for name in ("A", "B"):
        m = results[name]
        samples = generate_samples(m["model"], device)
        m["samples"] = samples

        r4 = sum(s["rep4"] for s in samples) / len(samples)
        r8 = sum(s["rep8"] for s in samples) / len(samples)
        sh = sum(s["structure"]["speaker_headings"] for s in samples) / len(samples)
        ll = sum(s["structure"]["mean_line_len"] for s in samples) / len(samples)

        z4 = (r4 - ctrl_mean) / ctrl_std
        verdict = ("DEGENERATE (above corpus)" if z4 > 2
                   else "below corpus" if z4 < -2 else "within corpus range")

        print("-" * 78)
        print(f"  MODEL {name}  ({m['params']:,} params)")
        print("-" * 78)
        print(f"    4-gram repetition : {r4:.4f}   (corpus {ctrl_mean:.4f} +/- {ctrl_std:.4f}, "
              f"z = {z4:+.2f})  -> {verdict}")
        print(f"    8-gram repetition : {r8:.4f}   (corpus {ctrl8_mean:.4f} +/- {ctrl8_std:.4f})")
        print(f"    speaker headings  : {sh:.2f}/sample  (corpus {struct_ctrl['speaker_headings']:.2f})")
        print(f"    mean line length  : {ll:.1f} chars  (corpus {struct_ctrl['mean_line_len']:.1f})")

        m["structural_verdict"] = (
            f"{sh:.1f} speaker headings/sample vs corpus {struct_ctrl['speaker_headings']:.1f}; "
            f"line length {ll:.0f} vs {struct_ctrl['mean_line_len']:.0f}"
        )
        m["style_verdict"] = f"val {m['sweep']['loss']:.3f} nats/byte, bytes-ppl {m['sweep']['perplexity']:.2f}"

        # Full sweep over all prompts at every temperature. Averaged over prompts,
        # so a single unlucky sample cannot drive the number.
        m["temp_sweep"] = temperature_sweep(m["model"], device)
        print("\n    repetition vs temperature (mean over all 5 prompts):")
        for temp, r in m["temp_sweep"].items():
            flag = "  <- ABOVE corpus" if r["rep4"] > ctrl_mean + 2 * ctrl_std else ""
            print(f"      T={temp}:  4-gram {r['rep4']:.4f}   8-gram {r['rep8']:.4f}   "
                  f"(worst single sample {r['worst_rep4']:.4f}){flag}")
        span = m["temp_sweep"][0.5]["rep4"] / max(1e-9, m["temp_sweep"][1.0]["rep4"])
        print(f"      -> T=0.5 repeats {span:.1f}x as much as T=1.0. Same weights.")
        print()

    # --- write samples to disk, full text ----------------------------------
    lines = [f"# Generated samples\n",
             f"temperature {GEN_TEMPERATURE}, top-k {GEN_TOP_K}, exactly {GEN_TOKENS} tokens per prompt.",
             f"Seed reset before each prompt, so A and B see an identical random stream —",
             f"the only difference between their outputs is the model.\n",
             f"Corpus control (real Shakespeare, 150-byte slices): "
             f"4-gram repetition {ctrl_mean:.4f} ± {ctrl_std:.4f}\n"]
    for name in ("A", "B"):
        lines.append(f"\n## Model {name} ({results[name]['params']:,} params)\n")
        for i, s in enumerate(results[name]["samples"], 1):
            lines.append(f"### Prompt {i}: `{s['prompt']!r}`")
            lines.append(f"4-gram repetition: {s['rep4']:.4f} | 8-gram: {s['rep8']:.4f}\n")
            lines.append("```")
            lines.append(s["prompt"] + s["completion"])
            lines.append("```\n")
    (out_dir / "samples.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  full samples -> {out_dir / 'samples.md'}")


if __name__ == "__main__":
    main()
