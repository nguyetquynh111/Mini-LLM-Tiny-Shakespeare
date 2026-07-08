# Architecture Mapping

The implementation is organized as a small GPT-style pipeline:

```text
tiny_shakespeare.txt
  -> mini_llm/data.py byte tokenizer and train/val batches
  -> mini_llm/model.py causal Transformer
  -> mini_llm/train.py optimization, logs, checkpoints
  -> evaluation/evaluate.py validation loss and perplexity
  -> evaluation/generate_samples.py prompt continuations
  -> evaluation/plot_losses.py loss convergence plot
```

Artifacts are written under `outputs/`: checkpoints in `outputs/checkpoints/`, loss logs in `outputs/logs/`, and evaluation artifacts in `outputs/evaluation/`.
