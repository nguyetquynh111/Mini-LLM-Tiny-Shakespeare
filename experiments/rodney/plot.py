"""loss_curves.png -- both models, train and val, against BOTH axes.

Two panels, and the second one is not decoration:

    left  : loss vs STEP
    right : loss vs TOKENS SEEN

Model A consumes 64 x 64 = 4,096 tokens per step; Model B consumes 32 x 256 =
8,192. So at any given step, B has read TWICE the text A has. A loss-vs-step plot
therefore compares two models on different diets, and silently flatters whichever
one has the bigger batch-tokens product. Plotting against tokens_seen puts both
models on the same x-axis -- amount of text actually read -- which is the axis the
capacity-vs-data argument needs.

If the two panels disagree about which model is ahead, that IS a finding, and the
script says so explicitly rather than leaving it to the reader to notice.
"""

import argparse
import csv
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import GAP_THRESHOLD, compute_baselines, gap_analysis
from train import load_data

COLORS = {"A": "#2563eb", "B": "#dc2626"}


def read_log(path: pathlib.Path) -> dict[str, list]:
    cols: dict[str, list] = {k: [] for k in
                             ["step", "tokens_seen", "train_loss", "val_loss", "lr", "wall_time"]}
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(r[k]))
    return cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/model_A", "runs/model_B"])
    ap.add_argument("--out", default="loss_curves.png")
    args = ap.parse_args()

    train_data, val_data = load_data()
    b = compute_baselines(train_data, val_data)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    logs, gaps = {}, {}
    for run in args.runs:
        d = pathlib.Path(run)
        if not (d / "log.csv").exists():
            continue
        name = d.name.replace("model_", "")
        logs[name] = read_log(d / "log.csv")
        gaps[name] = gap_analysis(d / "log.csv")

    for ax, xkey, xlabel in (
        (axes[0], "step", "step"),
        (axes[1], "tokens_seen", "tokens seen (bytes of text read)"),
    ):
        # Reference baselines as horizontal rules. They give the curves meaning:
        # without them a reader cannot tell whether 2.9 nats is good or terrible.
        for label, val, style in (
            (f"uniform-256 = {b['uniform_256']:.2f}", b["uniform_256"], ":"),
            (f"uniform-observed = {b['uniform_observed']:.2f}", b["uniform_observed"], "-."),
            (f"unigram CE (val) = {b['unigram_ce_val']:.2f}", b["unigram_ce_val"], "--"),
        ):
            ax.axhline(val, color="#9ca3af", linestyle=style, linewidth=1, zorder=1)
            ax.text(0.99, val, f" {label}", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=7.5, color="#6b7280")

        for name, log in logs.items():
            c = COLORS[name]
            x = log[xkey]
            # train dashed, val solid: same colour per model, so the GAP between the
            # two lines of one colour is the thing the eye is drawn to.
            ax.plot(x, log["train_loss"], color=c, linestyle="--", linewidth=1.4,
                    label=f"Model {name} train", zorder=3)
            ax.plot(x, log["val_loss"], color=c, linestyle="-", linewidth=1.9,
                    label=f"Model {name} val", zorder=3)

            ga = gaps[name]
            if ga["onset"]:
                ox = ga["onset"][xkey]
                ax.axvline(ox, color=c, alpha=0.28, linewidth=1, zorder=2)
                ax.plot([ox], [ga["onset"]["val_loss"]], marker="v", color=c, ms=7, zorder=4)
            vx = ga["val_min"][xkey]
            ax.plot([vx], [ga["val_min"]["val_loss"]], marker="o", color=c, ms=6,
                    mfc="white", mew=1.6, zorder=4)

        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25, zorder=0)
        ax.set_ylim(1.0, 5.8)

    axes[0].set_ylabel("loss (nats / byte)")
    axes[0].set_title("loss vs step\n(different batch sizes -- NOT a like-for-like comparison)",
                      fontsize=10)
    axes[1].set_title("loss vs tokens seen\n(same x-axis: text actually read -- the honest comparison)",
                      fontsize=10)
    axes[0].legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    fig.suptitle(
        "Byte-level GPT on tinyshakespeare  --  dashed = train, solid = val,  "
        "v = divergence onset,  o = val minimum",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    # --- do the two panels disagree? ---------------------------------------
    if len(logs) == 2:
        print("\nDo the two axes tell the same story?")
        # Compare the models at the largest COMMON token count, and at the largest
        # common step. If the ordering flips between the two, that is the finding.
        common_tok = min(max(l["tokens_seen"]) for l in logs.values())
        common_step = min(max(l["step"]) for l in logs.values())

        def val_at(log, key, target):
            best = min(range(len(log[key])), key=lambda i: abs(log[key][i] - target))
            return log["val_loss"][best]

        a_s, b_s = val_at(logs["A"], "step", common_step), val_at(logs["B"], "step", common_step)
        a_t, b_t = val_at(logs["A"], "tokens_seen", common_tok), val_at(logs["B"], "tokens_seen", common_tok)
        print(f"  at step {common_step:,.0f}        : A val {a_s:.4f}  B val {b_s:.4f}  "
              f"-> {'B' if b_s < a_s else 'A'} ahead")
        print(f"  at {common_tok:,.0f} tokens : A val {a_t:.4f}  B val {b_t:.4f}  "
              f"-> {'B' if b_t < a_t else 'A'} ahead")
        if (b_s < a_s) != (b_t < a_t):
            print("  ** THE TWO AXES DISAGREE. The step-based plot and the token-based plot")
            print("     name different winners. Report the token-based one: it controls for")
            print("     the fact that B reads twice as much text per step.")
        else:
            print("  the two axes agree on the winner (the token axis still matters for")
            print("  the ONSET comparison, where B's 2x tokens/step shifts the picture).")


if __name__ == "__main__":
    main()
