# Mini-LLM Tiny Shakespeare

![Training and validation convergence](outputs/evaluation/plots/loss_convergence.png)

A compact byte-level GPT-style Transformer implemented directly in PyTorch. The project trains and compares two model configurations on the complete Tiny Shakespeare corpus to examine how increased model capacity and context length affect performance.

## Team collaboration

Because the project is compact and its components are tightly coupled, the team primarily worked through in-person collaborative coding and pair-programming sessions. Major design decisions, implementation, and debugging were completed together. After each session, individual members reviewed and validated the components assigned to them in the accompanying report.

## Model configurations

| Model                  | Layers | Attention heads | Embedding size | Context length |
| ---------------------- | -----: | --------------: | -------------: | -------------: |
| **Model A — Baseline** |      2 |               4 |            128 |             64 |
| **Model B — Scaled**   |      4 |               4 |            256 |            128 |

Both models use byte-level tokenization with a fixed vocabulary of 256 possible byte values.

## Results

Final checkpoints are evaluated deterministically over all next-byte targets in the held-out validation split.

| Model       | Training tokens | Validation loss | Perplexity | Bits/byte |
| ----------- | --------------: | --------------: | ---------: | --------: |
| **Model A** |          10.24M |          1.7431 |     5.7149 |    2.5147 |
| **Model B** |          20.48M |          1.5059 |     4.5082 |    2.1726 |

Model B achieves lower validation loss, perplexity, and bits per byte than Model A.

### Equal-token comparison

Because Model B processes more tokens per training step due to its longer context window, the project also compares both models at the largest exact shared training budget of **10.24 million tokens**.

| Model       | Shared training budget | Validation loss |
| ----------- | ---------------------: | --------------: |
| **Model A** |          10.24M tokens |          1.7750 |
| **Model B** |          10.24M tokens |          1.6459 |

Model B therefore performs better even when both models are compared at the same number of processed training tokens.

These equal-token values come from the saved training logs. The primary results table reports deterministic full-validation evaluation of the final checkpoints.

![Equal-token comparison](outputs/evaluation/plots/equal_token_comparison.png)

## Setup

Create and activate the environment:

```bash
conda create -n tiny_llm python=3.10
conda activate tiny_llm
```

Install the required packages and run the automated tests:

```bash
python -m pip install -r requirements.txt
python -m pytest
```

The Tiny Shakespeare dataset is included at:

```text
data/tiny_shakespeare.txt
```

If the file is missing, the data loader automatically downloads the same public corpus.

## Training

Train both configurations:

```bash
python -m mini_llm.train --config model_a --grad-clip 1.0
python -m mini_llm.train --config model_b --grad-clip 1.0
```

Resume Model A from a saved checkpoint:

```bash
python -m mini_llm.train --config model_a \
  --resume-from outputs/checkpoints/model_a.pt \
  --grad-clip 1.0
```

Replace `model_a` with `model_b` and provide the corresponding checkpoint path to resume Model B.

Training uses:

* a fixed 90/10 byte-level train-validation split;
* vocabulary size 256;
* random seed 1337;
* AdamW optimization;
* cross-entropy loss;
* causal multi-head self-attention;
* gradient clipping.

Training histories and checkpoints are saved under:

```text
outputs/logs/
outputs/checkpoints/
```

## Evaluation

Generate the Gemini reference samples first:

```bash
python evaluation/generate_gemini_deepinfra.py
```

This command requires `DEEPINFRA_API_KEY` in the environment or a local `.env` file.

Then run deterministic evaluation, local generation, and saved-generation analysis:

```bash
python evaluation/evaluate.py
python evaluation/training_analysis.py
python evaluation/plot_losses.py
python evaluation/generate_samples.py \
  --max-new-tokens 150 \
  --seed 1337
python evaluation/analyze_generations.py
```

### Methodology

* Model A and Model B final checkpoints are compared after the same number of training steps.
* Model B has a larger context length, embedding dimension, depth, and parameter count than Model A.
* Local models generate exactly 150 new byte tokens for each fixed prompt.
* Gemini requests 150 provider completion tokens and records provider-reported token usage, finish reason, provider, provider model, UTC generation time, prompt, and returned text when the API returns them.

### Artifacts

| Artifact | Path |
| --- | --- |
| Deterministic metrics | [outputs/evaluation/metrics.csv](outputs/evaluation/metrics.csv) |
| Convergence statistics | [outputs/evaluation/convergence_stats.csv](outputs/evaluation/convergence_stats.csv) |
| Training comparisons | [outputs/evaluation/training_comparisons.csv](outputs/evaluation/training_comparisons.csv) |
| Generation metrics | [outputs/evaluation/generation_metrics.csv](outputs/evaluation/generation_metrics.csv) |
| Generation summary | [outputs/evaluation/generation_summary.csv](outputs/evaluation/generation_summary.csv) |
| Model A generations | [outputs/evaluation/generations/model_a.jsonl](outputs/evaluation/generations/model_a.jsonl) |
| Model B generations | [outputs/evaluation/generations/model_b.jsonl](outputs/evaluation/generations/model_b.jsonl) |
| Gemini Flash generations | [outputs/evaluation/generations/gemini_flash.jsonl](outputs/evaluation/generations/gemini_flash.jsonl) |
| Equal-token plot | [outputs/evaluation/plots/equal_token_comparison.png](outputs/evaluation/plots/equal_token_comparison.png) |

## Reproducibility

The project uses fixed model configurations, a fixed dataset split, and seed `1337`. Saved checkpoints, training histories, evaluation tables, generated samples, and plots are included so that the reported results can be inspected without retraining the models.
