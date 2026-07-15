# Model comparison — byte-level GPT on tinyshakespeare

| | Model A | Model B | Gemini Flash |
|---|---|---|---|
| **Parameters** | 470,784 | 10,942,720 | |
| **Architecture** | 2L / 4H / 128C, block 64 | 6L / 6H / 384C, block 256 | |
| **Val loss (nats/byte)** | 1.5617 | 1.4564 | |
| **Bytes-perplexity** | 4.767 | 4.291 | |
| **Bits/byte** | 2.253 | 2.101 | |
| **Val loss @ equal tokens (20.48M)** | 1.6064 | 1.5020 | |
| **Divergence onset** | step 500 (2,048,000 tok) | step 500 (4,096,000 tok) | |
| **Val minimum** | step 4900 (20,070,400 tok) | step 4000 (32,768,000 tok) | |
| **Terminal train/val gap** | +0.2111 | +0.5177 | |
| | | | |
| **Structural stability** (T=0.8) | 1.2 speaker headings/sample vs corpus 1.3; line length 34 vs 31 | 1.6 speaker headings/sample vs corpus 1.3; line length 30 vs 31 | |
| **Shakespearean style** | val 1.562 nats/byte, bytes-ppl 4.77 | val 1.456 nats/byte, bytes-ppl 4.29 | |
| | | | |
| **Degenerate repetition** (4-gram dup. rate — *temperature MUST be stated, see note*) | | | |
| &nbsp;&nbsp;at **T = 0.5**, top-k 40 | 0.098 — within corpus range | 0.076 — within corpus range | *(state T)* |
| &nbsp;&nbsp;at **T = 0.8**, top-k 40 | 0.049 — within corpus range | 0.086 — within corpus range | *(state T)* |
| &nbsp;&nbsp;at **T = 1.0**, top-k 40 | 0.024 — within corpus range | 0.053 — within corpus range | *(state T)* |
| &nbsp;&nbsp;worst single sample (T=0.8) | 0.102 | 0.177 | |
| &nbsp;&nbsp;**corpus control** (temperature-free) | 0.061 ± 0.054 | 0.061 ± 0.054 | 0.061 ± 0.054 |

> **Repetition is strongly temperature-dependent — this is a finding, not a footnote.**
> The same weights swing from 0.098 (T=0.5) to 0.024 (T=1.0) for Model A (4.0x), and 0.076 to 0.053 for Model B (1.4x).
> Lower T sharpens the distribution, which buys fewer misspellings and pays for them in loops.
>
> **A repetition number quoted without its temperature is unfalsifiable** — a wide range of
> figures can be produced from either model by choosing T. Hence every row above states its T,
> and comparing two models' repetition at different temperatures is meaningless. If the Gemini
> Flash column is filled in, its sampling temperature must be stated too, or the comparison is void.
>
> Stated precisely, because the tempting overclaim is wrong: the temperature swing does NOT
> always dominate the model difference. For Model A the swing (0.073) is larger than the A-vs-B gap at fixed T (0.037);
> for Model B the swing (0.023) is SMALLER than it. So temperature must always be
> reported, but it does not by itself explain away the difference between the models.
>
> Per-sample variance is also large: single prompts reach 0.16-0.18 at T=0.5 while the 5-prompt
> mean is ~0.08-0.10. Quote means over prompts, never a single sample.
>
> Note the direction: at T=0.8 and T=1.0 the LARGER model repeats MORE (0.086 vs 0.049).
> A better-fit model is a more confident model: its next-byte distribution is lower-entropy, so
> at any fixed T it concentrates more mass on its favourite continuation. Capacity buys accuracy
> and pays for it in diversity.

### Reference baselines (nats/byte)

| Baseline | Loss | Bytes-ppl | Meaning |
|---|---|---|---|
| uniform over 256 | 5.5452 | 256.00 | untrained; knows nothing |
| uniform over observed (65) | 4.1744 | 65.00 | knows only which bytes exist |
| train-unigram CE on val | 3.3475 | 28.43 | a frequency table; zero context |
| **real corpus (150-byte slices)** | — | — | **4-gram repetition 0.061 ± 0.054** |

Both models sit ~1.8–1.9 nats below the frequency-table floor, so both have
genuinely learned context rather than byte statistics.

### Perplexity comparability (IMPORTANT)

**Bytes-perplexity is comparable between Model A and Model B** — identical 256-byte
vocabulary, identical val split, identical scoring procedure. The A-vs-B numbers above
may be read side by side.

**It is NOT comparable to a model with a different tokenizer.** Perplexity is `exp` of the
mean NLL *per token*, so its value is only meaningful once you say what a token is. Ours is
one BYTE. If a teammate's implementation is character-level (a ~65-symbol vocabulary, where
one token is not one byte) or subword/BPE (where one token spans several characters), their
perplexity is normalized per a *different unit* and the numbers must NOT be placed in the
same column as though they measured the same thing. A word-level perplexity of 80 is a far
stronger result than a byte-level perplexity of 4.3, not a far worse one.

To compare across tokenizers, either:
1. **Report bits-per-byte** (given above), which renormalizes every model to the same unit —
   bytes of source text — and is the honest cross-tokenizer metric; or
2. Report the perplexities in **separate tables** and state why they cannot be compared.

The Gemini Flash column is deliberately empty. Note that a Gemini perplexity, if obtainable
at all, would be subword-level and therefore belongs under rule (2) above — the qualitative
rows (structure, style, repetition) are the ones that can be filled in and compared directly.