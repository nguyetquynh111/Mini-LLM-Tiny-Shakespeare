# Tiny Shakespeare Mini-LLM Report Draft

## Introduction

This project builds a small end-to-end language model trained on the Tiny Shakespeare corpus. The goal is not to compete with industrial systems, but to make the core parts of a modern autoregressive language model visible and runnable: tokenization, data batching, a causal Transformer, loss tracking, checkpointing, sampling, evaluation, and comparison. The finished repository provides two local model sizes, training logs, checkpoints, plots, validation metrics, generation samples, and a qualitative comparison against Gemini Flash.

## microGPT Blueprint Mapping Summary

The implementation follows the microGPT-style blueprint of a compact GPT pipeline. The dataset is converted into integer tokens, batches are sampled as fixed-length input and target sequences, a Transformer predicts the next token at every position, and cross-entropy measures how well the predicted distribution matches the true next byte. During generation, the model repeatedly crops the context to the configured block size, predicts the next-token distribution, samples with softmax and multinomial sampling, appends the token, and continues until the requested number of new tokens is produced.

The mapping is direct: `mini_llm/data.py` owns the byte tokenizer and train-validation split, `mini_llm/model.py` owns the GPT-style neural network, `mini_llm/train.py` owns optimization and checkpoint saving, `mini_llm/generate.py` owns single-prompt sampling, and `evaluation/` owns plotting, metrics, prompt batches, and qualitative comparison artifacts.

## Data Escalation From Simple Word-List Data to Tiny Shakespeare

Early language-model experiments often use simple word lists or tiny toy corpora because they make data flow easy to inspect. This project escalates that idea to Tiny Shakespeare, which is still small enough for a student-scale experiment but rich enough to expose real modeling problems. The corpus contains character names, punctuation, line breaks, archaic phrasing, dialogue structure, and long-range stylistic patterns. This makes it a better test of whether the model is learning more than isolated word frequencies.

Tiny Shakespeare also creates useful failure modes. Small models may learn local spelling and punctuation while failing at coherent scenes or speaker continuity. Those errors are expected and educational because they show the limits of model size, context length, and training budget.

## Byte-Level Tokenizer Explanation

The tokenizer is fixed at the byte level. Encoding uses `text.encode("utf-8")`, which converts text into raw bytes. Each byte is then treated as an integer token from `0` through `255`, so the vocabulary size is exactly `256`. Decoding reverses the process with `bytes(tokens).decode("utf-8", errors="replace")`.

This design has several advantages for a small project. It avoids unknown tokens, does not require training a separate tokenizer, and can represent any UTF-8 text. The tradeoff is that byte-level sequences are longer than word-level or subword-level sequences. The model must learn spelling, word boundaries, punctuation, and style from smaller units, which makes the task harder but keeps the pipeline transparent.

## PyTorch Transformer Architecture

The model is a GPT-style causal Transformer written with PyTorch modules. It includes token embeddings, positional embeddings, masked multi-head self-attention, a feedforward MLP, LayerNorm, residual connections, a final LayerNorm, and a language modeling head. The attention mask prevents each position from attending to future tokens, preserving the autoregressive next-token prediction setup.

The forward pass accepts `idx` with shape `(B, T)` and optional `targets` with the same shape. It returns logits with shape `(B, T, 256)`. When targets are present, the model flattens the batch and time dimensions and computes cross-entropy over the 256-byte vocabulary.

## Model A vs Model B Hyperparameter Comparison

Model A is the compact baseline. It uses a block size of `64`, embedding size of `128`, `4` attention heads, `2` Transformer layers, dropout of `0.2`, batch size of `32`, and learning rate of `3e-4`. This model is faster to train and easier to use for smoke tests, but it has limited capacity and a shorter context window.

Model B is the larger local model. It uses a block size of `128`, embedding size of `256`, `4` attention heads, `4` Transformer layers, dropout of `0.2`, batch size of `32`, and learning rate of `3e-4`. The longer context and deeper network should improve validation loss and make generations more stable, although the model is still small compared with production LLMs.

## Training and Validation Loss Discussion

Training loss measures how well the model predicts next bytes on sampled training batches. Validation loss measures the same objective on held-out data from the final 10 percent of the corpus. The loss convergence plot in `outputs/evaluation/plots/loss_convergence.png` should show whether each model continues improving, plateaus, or begins to overfit.

In a typical result, Model B is expected to reach a lower validation loss than Model A because it has more parameters and a longer context length. If training loss falls while validation loss stops improving, that suggests overfitting or diminishing returns from the current training budget. If both losses remain high, the model may need more training steps, adjusted learning rate, or more capacity.

## Cross-Entropy and Perplexity Comparison

The evaluation script computes final validation cross-entropy and perplexity for each checkpoint. Cross-entropy is the direct loss optimized during training. Perplexity is computed as `exp(loss)` and can be interpreted as the effective number of choices the model is uncertain among at each prediction step.

Lower values are better for both metrics. Model B should generally have lower cross-entropy and perplexity than Model A, although the exact result depends on training duration, random seed, hardware, and whether either model overfits. The metrics are saved in `outputs/evaluation/metrics.csv`; checkpoints are saved in `outputs/checkpoints/model_a.pt` and `outputs/checkpoints/model_b.pt`; training logs are saved in `outputs/logs/model_a_loss.csv` and `outputs/logs/model_b_loss.csv`.

## Qualitative Generation Comparison Against Gemini Flash

The local models generate continuations from the prompts in `evaluation/prompts.txt` and save them to `outputs/evaluation/generations/model_a.txt` and `outputs/evaluation/generations/model_b.txt`. The Gemini Flash comparison output is kept at `outputs/evaluation/generations/gemini_flash.txt`. Because the local models are byte-level mini-models, their outputs are expected to be imperfect. Model A may produce short readable fragments mixed with broken words, unstable punctuation, and repetition. Model B should usually produce more convincing local texture and better line structure, but it can still lose coherence or repeat patterns.

Gemini Flash is expected to be much stronger. It can intentionally follow a Shakespearean style, maintain readable syntax, and preserve a coherent idea across the whole continuation. The comparison is therefore not meant to be a fair competition. It is meant to show the gap between a small educational Transformer and an industrial-scale LLM trained on far more data with far more compute.

## Scale Narrative Comparing This Mini-LLM to Industrial LLMs

This mini-LLM contains the same conceptual ingredients as larger autoregressive language models: tokenization, embeddings, masked attention, residual Transformer blocks, cross-entropy training, checkpoints, and sampling. The difference is scale. Industrial LLMs use much larger datasets, many more parameters, longer context windows, sophisticated tokenizer training, distributed optimization, careful data filtering, alignment stages, and extensive evaluation.

The project is valuable because it compresses the central idea into a form that can be read and modified directly. The local models reveal why scale matters: small networks can learn texture and local patterns, but they struggle with deep coherence, factual control, and robust instruction following. Larger systems improve these qualities through model capacity, data diversity, training duration, and post-training methods.

## Team Member Role and Contribution Statement

Member 1: Architecture mapping and documentation.

Member 2: Data pipeline and tokenizer.

Member 3: Transformer implementation.

Member 4: Training pipeline.

Member 5: Evaluation and comparison.

Member 6: Experiments, visualization, and reproducibility.

Together, these roles cover the full project lifecycle: understanding the blueprint, building the model backbone, training local checkpoints, measuring performance, comparing qualitative outputs, and preparing the final documentation.
