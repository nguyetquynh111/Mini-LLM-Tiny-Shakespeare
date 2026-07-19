![Training and validation convergence](outputs/evaluation/plots/loss_convergence.png)

# Mini-LLM Tiny Shakespeare

A reproducible byte-level GPT-style Transformer implemented directly in PyTorch. The project trains and compares a small baseline (Model A) with a scaled model (Model B) on the complete Tiny Shakespeare corpus.

## Results

Final checkpoints are evaluated deterministically over every next-byte target in the held-out validation split.

| Model | Architecture | Training tokens | Validation loss | Perplexity | Bits/byte |
|---|---|---:|---:|---:|---:|
| A | 2 layers, 4 heads, 128 embedding, context 64 | 10.24M | 1.7431 | 5.7149 | 2.5147 |
| B | 4 layers, 4 heads, 256 embedding, context 128 | 20.48M | 1.5059 | 4.5082 | 2.1726 |

The larger model also wins at the largest exact shared training budget. At 10.24M tokens, the saved training logs report validation losses of 1.7750 for Model A and 1.6459 for Model B. These equal-token values are logged batch estimates; the table above contains the separate full-validation results.

![Equal-token comparison](outputs/evaluation/plots/equal_token_comparison.png)

## Setup

```bash
conda create -n tiny_llm python=3.10
conda activate tiny_llm
python -m pip install -r requirements.txt
python -m pytest
```

The dataset is already included at `data/tiny_shakespeare.txt`. If it is removed, the data loader downloads the same public corpus automatically.

## Training

```bash
python -m mini_llm.train --config model_a --grad-clip 1.0
python -m mini_llm.train --config model_b --grad-clip 1.0
```

Resume without changing the existing CLI:

```bash
python -m mini_llm.train --config model_a \
  --resume-from outputs/checkpoints/model_a.pt --grad-clip 1.0
```

Training uses a fixed 90/10 byte-level split, vocabulary size 256, seed 1337, AdamW, cross-entropy loss, and explicit vectorized causal self-attention. Checkpoints embed model, optimizer, tokenizer, configuration, and loss-history metadata.

## Evaluation and plots

```bash
python evaluation/evaluate.py
python evaluation/training_analysis.py
python evaluation/plot_losses.py
```

`evaluate.py` performs a deterministic context-primed sweep and scores all 111,539 validation targets exactly once. It reports loss, perplexity, bits-per-byte, coverage, and three context-free baselines. `training_analysis.py` preserves the original fixed-step comparison and adds the equal-token comparison from saved logs. No intermediate checkpoint is claimed where none exists.

Artifacts:

- `outputs/evaluation/metrics.csv` — deterministic full-validation metrics and baselines.
- `outputs/evaluation/convergence_stats.csv` — best point, generalization gap, and sustained divergence.
- `outputs/evaluation/training_comparisons.csv` — fixed-step and equal-token log comparisons.
- `outputs/evaluation/plots/` — publication-quality plots.

## Generation comparison

```bash
python evaluation/generate_samples.py --max-new-tokens 150 --seed 1337
python evaluation/analyze_generations.py
```

Each local sample contains exactly 150 new byte tokens. Model A and Model B use paired per-prompt seeds, and every token ID and continuation is saved in JSONL. The analysis reuses the existing Gemini artifacts and makes no API calls. Because Gemini's tokenizer is unavailable and incompatible with byte-level perplexity, repetition and structure are compared over equal 150-byte text prefixes instead.

See `outputs/evaluation/qualitative_comparison.md`, `outputs/evaluation/generation_metrics.csv`, and `outputs/evaluation/generation_summary.csv` for the systematic comparison and failure analysis.

The optional existing DeepInfra command remains available for explicitly authorized regeneration:

```bash
DEEPINFRA_API_KEY=your_key python evaluation/generate_gemini_deepinfra.py
```

Record the exact provider model and evaluation date if those external artifacts are regenerated.

## Repository layout

```text
mini_llm/                  model, data, training, and generation code
evaluation/                evaluation scripts and fixed prompt inputs
outputs/checkpoints/       final and best saved checkpoints
outputs/logs/              training and validation loss histories
outputs/evaluation/        saved metrics, comparisons, generations, and plots
tests/                     automated correctness and compatibility tests
```
