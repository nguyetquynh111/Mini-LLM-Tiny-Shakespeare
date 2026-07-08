![Loss convergence plot](outputs/evaluation/plots/loss_convergence.png)

# Mini-LLM Tiny Shakespeare

A small byte-level GPT-style Transformer trained on Tiny Shakespeare.

Code lives in the `mini_llm/` Python package. Generated checkpoints, logs, metrics, plots, and sample text live in `outputs/`.

## Quick Start

```bash
conda create -n tiny_llm python=3.10
conda activate tiny_llm
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest 
```

## Main Commands

```bash
# Train
python -m mini_llm.train --config model_a --grad-clip 1.0
python -m mini_llm.train --config model_b --grad-clip 1.0

# Resume training
python -m mini_llm.train --config model_a --resume-from outputs/checkpoints/model_a.pt --grad-clip 1.0

# Evaluate trained checkpoints
python evaluation/evaluate.py
python evaluation/plot_losses.py

# Generate local samples
python evaluation/generate_samples.py --max-new-tokens 150

# Generate from one checkpoint
python -m mini_llm.generate --checkpoint outputs/checkpoints/model_a.pt --prompt "To be, or not to " --max-new-tokens 150
```

The dataset is downloaded automatically on first use and saved as `data/tiny_shakespeare.txt`.
Training writes a final checkpoint such as `model_a.pt` and a best-validation checkpoint such as `model_a_best.pt`.

## DeepInfra / Gemini

Generate Gemini Flash 3.5 comparison samples through DeepInfra:

```bash
DEEPINFRA_API_KEY=your_key python evaluation/generate_gemini_deepinfra.py
```

Or use `.env`:

```text
DEEPINFRA_API_KEY=your_key
DEEPINFRA_MODEL=google/gemini-3.5-flash
```

## Outputs

```text
outputs/checkpoints/model_a.pt
outputs/checkpoints/model_a_best.pt
outputs/checkpoints/model_b.pt
outputs/checkpoints/model_b_best.pt
outputs/logs/model_a_loss.csv
outputs/logs/model_b_loss.csv
evaluation/metrics.csv
outputs/evaluation/generations/model_a.txt
outputs/evaluation/generations/model_a.jsonl
outputs/evaluation/generations/model_b.txt
outputs/evaluation/generations/model_b.jsonl
outputs/evaluation/generations/gemini_flash.jsonl
outputs/evaluation/plots/loss_convergence.png
```

## Model Presets

`model_a`: smaller baseline, 2 layers, 128 embeddings, block size 64.

`model_b`: larger model, 4 layers, 256 embeddings, block size 128.

Both local models use byte-level tokens with `vocab_size=256`.