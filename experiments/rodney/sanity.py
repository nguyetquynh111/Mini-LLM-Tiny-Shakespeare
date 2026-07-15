"""Correctness gate. Nothing long runs until this passes.

Four things, printed:

  1. Initial loss on REAL batches, both models, against ln(256) = 5.545.
     If init is wrong, or the logits reshape is wrong, or the targets are
     misaligned, this number is wrong -- and it is the cheapest place to catch it.
  2. Overfit a single batch to ~0 with dropout=0.0.
     Proves the model has enough capacity and that gradients actually flow to
     every parameter. A model that cannot memorize 4,096 tokens has a broken
     backward pass somewhere.
  3. MEASURED peak VRAM for Model B (torch.cuda.max_memory_allocated()).
  4. MEASURED wall-clock per 100 steps, both models, extrapolated to the full run.

    python sanity.py
"""

import math
import time

import torch

from model import CONFIGS, GPT, GPTConfig
from train import (
    DEFAULTS,
    TrainConfig,
    configure_optimizer,
    get_batch,
    get_device,
    load_data,
    set_seed,
)

LN_256 = math.log(256)  # 5.5452
# Tight, because it can be: lm_head uses a width-aware 1/sqrt(C) init (see model.py),
# so the initial loss is ln(256) regardless of n_embd. With a fixed-std head this
# would have to be ~0.15 to let Model B through, and a real 0.12-nat bug would slip
# past unnoticed.
INIT_TOL = 0.01  # nats
OVERFIT_TOL = 0.10  # nats
OVERFIT_STEPS = 500


def check_init_loss(train_data: torch.Tensor, device: torch.device, n_batches: int = 20) -> bool:
    """Untrained loss on real data must be ~ln(256).

    Why ln(256) and not ln(65): at init the model has learned NOTHING, including
    the fact that 191 byte values never occur. Its logits are ~0, so the softmax is
    uniform over all 256 classes and -log(1/256) = ln(256) = 5.545 per token. The
    fact that the corpus only uses 65 bytes is something the model must LEARN; it
    is not baked into the init.

    Deviations tell you exactly what is broken:
      much HIGHER  -> init std too large: logits are big and confidently wrong.
      much LOWER   -> information is leaking. Either the causal mask is not
                      applied (the model can read the answer) or the targets are
                      misaligned in a way that makes the task trivial.
    """
    print("=" * 72)
    print("[1/4] initial loss on real batches   (expect ln(256) = 5.5452)")
    print("=" * 72)

    ok = True
    for name, cfg in CONFIGS.items():
        set_seed(1337)
        model = GPT(cfg).to(device)
        model.eval()  # dropout OFF -- we want the clean init value

        losses = []
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = get_batch(train_data, cfg.block_size, 32, device, generator=gen)
                _, loss = model(x, y)
                losses.append(loss.item())

        mean = sum(losses) / len(losses)
        delta = abs(mean - LN_256)
        passed = delta < INIT_TOL
        ok &= passed
        print(
            f"  Model {name}: {mean:.4f}   (ln(256) = {LN_256:.4f}, "
            f"delta {delta:+.4f})   {'PASS' if passed else 'FAIL'}"
        )
    print()
    return ok


def check_overfit(train_data: torch.Tensor, device: torch.device) -> bool:
    """Drive a single fixed batch's loss toward 0 with dropout disabled.

    dropout=0.0 is essential and is not a convenience: dropout puts a FLOOR on how
    low the loss can go (you cannot perfectly memorize a batch while being randomly
    lobotomized on every forward pass), so a passing model would look like a failing
    one. This is a capacity-and-gradient-flow test, not a regularization test.
    """
    print("=" * 72)
    print(f"[2/4] overfit ONE batch, dropout=0.0, {OVERFIT_STEPS} steps   (expect -> ~0)")
    print("=" * 72)

    ok = True
    for name, base in CONFIGS.items():
        set_seed(1337)
        cfg = GPTConfig(**{**base.__dict__, "dropout": 0.0})
        model = GPT(cfg).to(device)
        model.train()

        bs = DEFAULTS[name]["batch_size"]
        gen = torch.Generator().manual_seed(0)
        x, y = get_batch(train_data, cfg.block_size, bs, device, generator=gen)  # ONE batch, reused

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
        first = None
        trace = []
        for step in range(OVERFIT_STEPS + 1):
            _, loss = model(x, y)
            if step == 0:
                first = loss.item()
            if step % 100 == 0:
                trace.append((step, loss.item()))
            if step == OVERFIT_STEPS:
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        final = trace[-1][1]
        passed = final < OVERFIT_TOL
        ok &= passed
        curve = "  ".join(f"{s}:{l:.3f}" for s, l in trace)
        print(f"  Model {name} ({bs} x {cfg.block_size} = {bs * cfg.block_size:,} tokens memorized)")
        print(f"    {curve}")
        print(f"    {first:.4f} -> {final:.4f}   {'PASS' if passed else 'FAIL'}")
    print()
    return ok


def measure_vram_and_speed(train_data: torch.Tensor, device: torch.device) -> None:
    """Measured peak VRAM and step time. No estimates."""
    print("=" * 72)
    print("[3/4] peak VRAM, measured   [4/4] wall-clock, measured")
    print("=" * 72)

    if device.type == "cuda":
        total = torch.cuda.mem_get_info()[1] / 1024**3
        free = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"  GPU: {torch.cuda.get_device_name(0)}   {total:.2f} GB total, "
              f"{free:.2f} GB free ({total - free:.2f} GB used by other apps)\n")

    results = {}
    for name, cfg in CONFIGS.items():
        d = DEFAULTS[name]
        tc = TrainConfig(model=name, steps=d["steps"], batch_size=d["batch_size"],
                         learning_rate=d["learning_rate"])
        set_seed(1337)
        torch.set_float32_matmul_precision("high")

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model = GPT(cfg).to(device)
        opt = configure_optimizer(model, tc)
        model.train()

        def one_step() -> None:
            x, y = get_batch(train_data, cfg.block_size, tc.batch_size, device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()

        for _ in range(10):  # warmup: allocator + autotune, excluded from timing
            one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(100):
            one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_100 = time.time() - t0

        # Eval cost, measured separately: it is NOT free. Every eval_interval we do
        # eval_iters forward passes on EACH of train and val.
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(100):
                x, y = get_batch(train_data, cfg.block_size, tc.batch_size, device)
                model(x, y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        eval_100 = time.time() - t0

        peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        reserved = torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0

        # Extrapolate the full run.
        n_evals = tc.steps // tc.eval_interval + 1
        eval_fwds = n_evals * tc.eval_iters * 2  # 2 splits
        train_s = train_100 / 100 * tc.steps
        eval_s = eval_100 / 100 * eval_fwds
        results[name] = dict(
            train_100=train_100, eval_100=eval_100, peak=peak, reserved=reserved,
            train_s=train_s, eval_s=eval_s, total_s=train_s + eval_s,
            tokens=tc.tokens_per_step * tc.steps, tps=tc.tokens_per_step,
            steps=tc.steps, eval_fwds=eval_fwds,
        )

        print(f"  Model {name}  (batch {tc.batch_size} x block {cfg.block_size}, "
              f"{model.num_params():,} params)")
        print(f"    peak VRAM allocated : {peak:6.3f} GB")
        print(f"    peak VRAM reserved  : {reserved:6.3f} GB   <- what nvidia-smi shows")
        print(f"    train: {train_100:6.2f} s / 100 steps  ({train_100 * 10:.1f} ms/step)")
        print(f"    eval : {eval_100:6.2f} s / 100 forward passes")
        print()

    print("  " + "-" * 68)
    print("  FULL RUN EXTRAPOLATION")
    print("  " + "-" * 68)
    for name, r in results.items():
        print(f"  Model {name}: {r['steps']:,} steps x {r['tps']:,} tok/step "
              f"= {r['tokens']:,} tokens")
        print(f"    training  : {r['train_s'] / 60:6.1f} min")
        print(f"    evaluation: {r['eval_s'] / 60:6.1f} min  "
              f"({r['eval_fwds']:,} forward passes at eval_iters=200)")
        print(f"    TOTAL     : {r['total_s'] / 60:6.1f} min")
    grand = sum(r["total_s"] for r in results.values()) / 60
    print(f"\n  Both models: {grand:.1f} min total")

    if device.type == "cuda":
        headroom = torch.cuda.mem_get_info()[1] / 1024**3 - results["B"]["reserved"]
        print(f"\n  Model B VRAM headroom: {headroom:.2f} GB spare of "
              f"{torch.cuda.mem_get_info()[1] / 1024**3:.2f} GB")


def main() -> int:
    device = get_device()
    print(f"\ndevice: {device}")
    if device.type == "cuda":
        print(f"torch.cuda.is_available() -> True")
        print(f"gpu: {torch.cuda.get_device_name(0)}\n")
    else:
        print("WARNING: not on CUDA. This will be slow.\n")

    train_data, _ = load_data()

    init_ok = check_init_loss(train_data, device)
    overfit_ok = check_overfit(train_data, device)
    measure_vram_and_speed(train_data, device)

    print("\n" + "=" * 72)
    if init_ok and overfit_ok:
        print("GATE PASSED. Both correctness checks green; safe to train.")
        return 0
    print("GATE FAILED. Do NOT train.")
    print("  init loss wrong    -> check weight init std, or the (B*T, V) logits reshape")
    print("  overfit failed     -> check target alignment (y = x shifted by 1), or")
    print("                        that gradients reach every parameter")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
