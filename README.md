![Loss convergence plot](evaluation/loss_convergence.png)

# Mini-LLM Tiny Shakespeare

This repository implements a small end-to-end byte-level Transformer language model trained on Tiny Shakespeare. It includes a fixed tokenizer, data batching, GPT-style causal self-attention, training, checkpointing, generation, plotting, evaluation, and report artifacts.

## Repository Structure

```text
.
|-- configs.py
|-- data.py
|-- model.py
|-- train.py
|-- generate.py
|-- requirements.txt
|-- .env.example
|-- checkpoints/
|   |-- model_a.pt
|   `-- model_b.pt
|-- logs/
|   |-- model_a_loss.csv
|   `-- model_b_loss.csv
|-- evaluation/
|   |-- plot_losses.py
|   |-- evaluate.py
|   |-- generate_samples.py
|   |-- generate_gemini_deepinfra.py
|   |-- loss_convergence.png
|   |-- metrics.csv
|   |-- prompts.txt
|   |-- generations_model_a.txt
|   |-- generations_model_b.txt
|   |-- generations_gemini.txt
|   `-- comparison_table.md
`-- report.md
```

## Installation

Create and activate an environment named `tiny_llm`, then install the required packages:

```bash
python -m venv tiny_llm
source tiny_llm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you prefer Conda, use:

```bash
conda create -n tiny_llm python=3.10
conda activate tiny_llm
python -m pip install -r requirements.txt
```

Create a `.env` file for DeepInfra API access:

```bash
cp .env.example .env
```

Then edit `.env` and set `DEEPINFRA_API_KEY`.

The dataset loader downloads Tiny Shakespeare automatically on first use and saves it as `tiny_shakespeare.txt`.

## Byte-Level Tokenizer

The tokenizer is intentionally fixed and simple. Text is encoded with `text.encode("utf-8")`, and each byte becomes one integer token from `0` to `255`. Decoding uses `bytes(tokens).decode("utf-8", errors="replace")`. This makes the vocabulary size exactly `256`, avoids unknown tokens, and lets the model learn directly over byte sequences.

## Model A vs Model B

Model A is the smaller baseline: `block_size=64`, `batch_size=32`, `n_embd=128`, `n_head=4`, `n_layer=2`, and `dropout=0.2`.

Model B is larger: `block_size=128`, `batch_size=32`, `n_embd=256`, `n_head=4`, `n_layer=4`, and `dropout=0.2`. It has more capacity and a longer context window, so it should usually reach lower validation loss and produce more stable text than Model A.

Both models use `vocab_size=256` and `learning_rate=3e-4`.

## Training

Train the two local models from the repository root:

```bash
python train.py --config model_a
python train.py --config model_b
```

For a quick smoke test:

```bash
python train.py --config model_a --max_iters 2 --eval_interval 1 --eval_iters 1
python train.py --config model_b --max_iters 2 --eval_interval 1 --eval_iters 1
```

Training writes CSV logs to `logs/` and checkpoints to `checkpoints/`.

## Generation

Generate exactly 150 new tokens from a saved checkpoint:

```bash
python generate.py --checkpoint checkpoints/model_a.pt --prompt "To be, or not to " --max_new_tokens 150
python generate.py --checkpoint checkpoints/model_b.pt --prompt "To be, or not to " --max_new_tokens 150
```

Generate all evaluation prompt samples:

```bash
python evaluation/generate_samples.py
```

Generate Gemini Flash samples through DeepInfra and update the qualitative comparison table:

```bash
source tiny_llm/bin/activate
python evaluation/generate_gemini_deepinfra.py
```

## Evaluation

Create the loss convergence plot:

```bash
python evaluation/plot_losses.py
```

Compute validation cross-entropy and perplexity:

```bash
python evaluation/evaluate.py
```

## Expected Output Files

The completed workflow should produce:

```text
logs/model_a_loss.csv
logs/model_b_loss.csv
checkpoints/model_a.pt
checkpoints/model_b.pt
evaluation/loss_convergence.png
evaluation/metrics.csv
evaluation/generations_model_a.txt
evaluation/generations_model_b.txt
evaluation/generations_gemini.txt
evaluation/comparison_table.md
report.md
```
