# Mini-LLM: Tiny Shakespeare

![Model A and Model B training and validation loss convergence](evaluation/plots/loss_convergence.png)

A compact, reproducible byte-level GPT implementation in PyTorch. This repository trains and compares two causal Transformer language models on Tiny Shakespeare, saves complete checkpoints and loss histories, generates fixed-seed samples, and evaluates both models over the full validation split.

## Results

| Model | Layers | Heads | Embedding | Context | Training tokens | Validation loss | Perplexity | Bits/byte |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Model A — Baseline** | 2 | 4 | 128 | 64 | 10.24M | 1.7431 | 5.7149 | 2.5147 |
| **Model B — Scaled** | 4 | 4 | 256 | 128 | 20.48M | 1.5059 | 4.5082 | 2.1726 |

The reported checkpoint metrics score every next-byte target in the held-out validation split exactly once. At the largest shared budget of 10.24M processed tokens, the saved training logs report validation losses of 1.7750 for Model A and 1.6459 for Model B.

## Repository layout

```text
mini_llm/
├── configs.py              # Model A and Model B presets
├── data.py                 # Byte tokenizer and dataset loading
├── model.py                # GPT architecture
├── train.py                # Training and checkpoint loop
├── generate.py             # Single-prompt generation CLI
├── utils.py                # Paths, seeding, and checkpoint helpers
├── checkpoints/            # Final and best Model A/Model B checkpoints
└── logs/                   # Reproducible training-loss CSV files
evaluation/
├── prompts.txt             # Fixed evaluation prompts
├── evaluate.py             # Deterministic full-validation metrics
├── generate_samples.py     # Paired local-model generation
├── analyze_generations.py  # Generation-quality metrics
├── training_analysis.py    # Convergence and budget analysis
├── plot_losses.py          # Publication-quality plots
├── generations/            # Saved text and JSONL outputs
├── results/                # Evaluation and analysis CSV files
└── plots/                  # Loss and comparison figures
data/tiny_shakespeare.txt   # Included training corpus
tests/                      # Automated unit and integration tests
requirements.txt            # Pinned Python dependencies
```

## Install

Run all commands from the repository root. Python 3.10 is recommended and is the reference environment for the pinned dependencies.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. PyTorch automatically uses CUDA or Apple MPS when available; pass `--device cpu` to force CPU execution.

The corpus is already included at `data/tiny_shakespeare.txt`. If it is removed, the loader downloads the same public Tiny Shakespeare file automatically.

## Train and generate: Model A

Train Model A from its fixed preset and seed:

```bash
python -m mini_llm.train \
  --config model_a \
  --grad-clip 1.0 \
  --seed 1337
```

This writes `mini_llm/checkpoints/model_a.pt`, `mini_llm/checkpoints/model_a_best.pt`, and `mini_llm/logs/model_a_loss.csv`. Generate the documented 150-token sample from the final checkpoint:

```bash
python -m mini_llm.generate \
  --checkpoint mini_llm/checkpoints/model_a.pt \
  --prompt "To be, or not to" \
  --max-new-tokens 150 \
  --temperature 1.0 \
  --seed 1337 \
  --output evaluation/generations/model_a_generation.json
```

## Train and generate: Model B

Train Model B from its fixed preset and seed:

```bash
python -m mini_llm.train \
  --config model_b \
  --grad-clip 1.0 \
  --seed 1337
```

This writes `mini_llm/checkpoints/model_b.pt`, `mini_llm/checkpoints/model_b_best.pt`, and `mini_llm/logs/model_b_loss.csv`. Generate the matching sample from Model B:

```bash
python -m mini_llm.generate \
  --checkpoint mini_llm/checkpoints/model_b.pt \
  --prompt "To be, or not to" \
  --max-new-tokens 150 \
  --temperature 1.0 \
  --seed 1337 \
  --output evaluation/generations/model_b_generation.json
```

The trained checkpoints are committed, so both generation commands work immediately after installation. Full training defaults to 5,000 optimizer steps per model and can take substantial time on CPU. For a fast pipeline check that preserves the committed artifacts, use a temporary destination:

```bash
python -m mini_llm.train \
  --config model_a \
  --max-iters 2 \
  --eval-interval 1 \
  --eval-iters 1 \
  --checkpoint-dir /tmp/mini_llm_smoke/checkpoints \
  --log-dir /tmp/mini_llm_smoke/logs
```

To resume training, provide the matching checkpoint and a larger total step target:

```bash
python -m mini_llm.train \
  --config model_a \
  --resume-from mini_llm/checkpoints/model_a.pt \
  --max-iters 5500 \
  --grad-clip 1.0
```

## Reproduce evaluation and plots

After both checkpoints exist, reproduce the local generations, deterministic metrics, convergence tables, generation analysis, and figures with:

```bash
python -m evaluation.generate_samples --max-new-tokens 150 --seed 1337
python -m evaluation.evaluate --seed 1337
python -m evaluation.training_analysis
python -m evaluation.analyze_generations
python -m evaluation.plot_losses
```

`evaluation.generate_samples` uses the same prompt-specific seeds for Model A and Model B, producing paired JSONL and readable text files under `evaluation/generations/`. `evaluation.evaluate` scores the entire validation split and writes `evaluation/results/metrics.csv`. Re-running `evaluation.plot_losses` recreates the figure displayed at the top of this README from the CSV loss logs.

The optional Gemini comparison requires a DeepInfra key. Copy `.env.example` to `.env`, set `DEEPINFRA_API_KEY`, and run:

```bash
python -m evaluation.generate_gemini_deepinfra
python -m evaluation.analyze_generations
```

Gemini uses provider tokens rather than this project’s byte tokens, so its saved generations are treated as a qualitative reference and normalized by UTF-8 byte length only for text-shape metrics.

## Saved artifacts

| Artifact | Location |
| --- | --- |
| Model A and B loss histories | [`mini_llm/logs/`](mini_llm/logs/) |
| Final and best checkpoints | [`mini_llm/checkpoints/`](mini_llm/checkpoints/) |
| Deterministic validation metrics | [`evaluation/results/metrics.csv`](evaluation/results/metrics.csv) |
| Convergence statistics | [`evaluation/results/convergence_stats.csv`](evaluation/results/convergence_stats.csv) |
| Equal-budget comparison | [`evaluation/results/training_comparisons.csv`](evaluation/results/training_comparisons.csv) |
| Generation metrics and summary | [`evaluation/results/`](evaluation/results/) |
| Local and Gemini generations | [`evaluation/generations/`](evaluation/generations/) |
| Convergence and equal-token plots | [`evaluation/plots/`](evaluation/plots/) |

## Reproducibility details

- Fixed 90/10 train-validation split with no shuffled preprocessing.
- Fixed 256-value UTF-8 byte vocabulary.
- Fixed default seed `1337` for Python, NumPy, PyTorch, and CUDA.
- AdamW optimization, cross-entropy loss, causal self-attention, and optional gradient clipping.
- Checkpoints include the model configuration, optimizer state, tokenizer metadata, training step, losses, and loss history.
- Exact direct dependency versions are pinned in `requirements.txt`.
