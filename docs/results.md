# Full results

All numbers: LongBench, 8 English tasks, official prompts and scorer, full test split of every task, greedy decoding
(see [`spectra/eval/longbench.py`](../spectra/eval/longbench.py)). SPECTRA uses the calibration recipe
`wikitext+code` (8 WikiText-2 + 8 code windows of 2,048 tokens). KV compression counts everything a method stores
(scales, zero points, exact windows) and is averaged by bytes over the evaluated samples
([`spectra/accounting.py`](../spectra/accounting.py)). Rows are sorted by compression. SPECTRA rows give the public
preset name and the internal config ([presets.md](presets.md)); on Qwen2.5-7B the ratio names do not apply, so only
the config is given.

## Llama-3.1-8B-Instruct

| Method | KV compression | Qasper | QMSum | MultiNews | TREC | TriviaQA | SAMSum | LCC | RepoBench-P | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| Dense (fp16) | 1.00× | 46.57 | 25.19 | 27.03 | 72.50 | 91.65 | 43.47 | 63.38 | 56.55 | 53.29 |
| **SPECTRA** `3.6x` (`s35`) | 3.56× | 46.14 | 25.57 | 26.90 | 72.50 | 92.17 | 43.65 | 63.42 | 55.08 | 53.18 |
| PolarQuant r4t4v4 | 3.66× | 46.01 | 25.56 | 26.69 | 72.50 | 91.86 | 42.88 | 65.38 | 58.44 | 53.66 |
| KIVI K4V4 | 3.76× | 46.11 | 25.14 | 27.11 | 72.50 | 92.42 | 44.10 | 65.05 | 58.91 | 53.92 |
| OTT K4V4 | 3.76× | 46.47 | 25.30 | 26.66 | 72.50 | 91.73 | 43.73 | 64.96 | 59.28 | 53.83 |
| **SPECTRA** `4x` (`t40b-B`) | 3.95× | 47.46 | 25.41 | 26.27 | 72.00 | 90.81 | 43.16 | 63.65 | 54.79 | 52.94 |
| TurboQuant K4V4 | 4.00× | 45.87 | 25.21 | 26.79 | 73.00 | 91.06 | 43.68 | 62.44 | 54.65 | 52.84 |
| PolarQuant r3t3v4 | 4.13× | 45.60 | 25.38 | 26.60 | 72.50 | 92.16 | 42.44 | 65.36 | 57.97 | 53.50 |
| **SPECTRA** `4.6x` (`s45`) | 4.57× | 44.48 | 25.04 | 27.02 | 71.50 | 90.90 | 43.43 | 63.14 | 55.47 | 52.62 |
| PolarQuant r2t2v4 | 4.74× | 42.52 | 24.37 | 25.46 | 66.00 | 91.50 | 40.91 | 61.13 | 53.18 | 50.63 |
| TurboQuant K4V2 | 5.33× | 43.94 | 25.14 | 26.29 | 71.50 | 91.23 | 42.87 | 62.57 | 57.01 | 52.57 |
| RotateKV 3-bit | 5.33× | 45.67 | 24.69 | 26.85 | 73.50 | 90.34 | 42.86 | 61.42 | 53.71 | 52.38 |
| **SPECTRA** `5.5x` (`L48-B`) | 5.51× | 43.40 | 25.32 | 26.61 | 72.00 | 89.66 | 43.67 | 60.57 | 52.66 | 51.74 |
| **SPECTRA** `5.8x` (`L55-B`) | 5.75× | 43.62 | 24.83 | 26.61 | 70.00 | 90.93 | 44.12 | 60.78 | 54.68 | 51.95 |
| **SPECTRA** `6x` (`L65-B`) | 6.02× | 43.85 | 25.07 | 26.26 | 71.50 | 91.15 | 44.15 | 62.07 | 56.70 | 52.59 |
| **SPECTRA** `6.7x` (`L70-B`) | 6.74× | 44.20 | 25.51 | 25.83 | 70.50 | 90.14 | 44.11 | 61.83 | 60.95 | 52.88 |
| PolarQuant r2t2v2 | 6.74× | 41.66 | 24.03 | 23.54 | 59.00 | 92.25 | 38.88 | 58.23 | 50.46 | 48.51 |
| OTT K2V2 | 7.11× | 44.98 | 24.81 | 26.61 | 72.50 | 91.85 | 43.95 | 64.25 | 57.44 | 53.30 |
| KIVI K2V2 | 7.11× | 42.60 | 24.73 | 26.45 | 71.50 | 91.14 | 43.42 | 61.75 | 55.80 | 52.17 |
| **SPECTRA** `7.7x` (`t80e-B`) | 7.65× | 45.34 | 25.12 | 26.17 | 71.50 | 91.78 | 43.40 | 61.77 | 61.26 | 53.29 |
| RotateKV 2-bit | 8.00× | 37.92 | 23.41 | 26.38 | 61.00 | 87.64 | 39.71 | 53.47 | 46.50 | 47.00 |
| TurboQuant K2V2 | 8.00× | 40.50 | 23.17 | 24.63 | 62.00 | 86.96 | 42.82 | 48.80 | 42.57 | 46.43 |
| **SPECTRA** `8.8x` (`s95-B`) | 8.82× | 45.84 | 25.24 | 25.69 | 73.00 | 89.77 | 43.64 | 62.77 | 57.95 | 52.99 |
| **SPECTRA** `11x` (`s12-B`) | 11.09× | 47.00 | 23.75 | 25.78 | 69.50 | 88.60 | 44.23 | 63.88 | 56.81 | 52.44 |

## Mistral-7B-Instruct-v0.3

| Method | KV compression | Qasper | QMSum | MultiNews | TREC | TriviaQA | SAMSum | LCC | RepoBench-P | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| Dense (fp16) | 1.00× | 38.15 | 25.57 | 26.43 | 76.00 | 88.59 | 47.46 | 59.46 | 60.56 | 52.78 |
| **SPECTRA** `3.6x` (`s35`) | 3.56× | 38.47 | 25.74 | 26.46 | 76.00 | 88.02 | 47.46 | 59.09 | 61.24 | 52.81 |
| OTT K4V4 | 3.76× | 37.95 | 25.80 | 26.50 | 76.00 | 88.89 | 47.08 | 60.40 | 61.30 | 52.99 |
| KIVI K4V4 | 3.76× | 36.88 | 25.42 | 26.58 | 76.00 | 88.89 | 47.09 | 60.53 | 60.99 | 52.80 |
| **SPECTRA** `4x` (`t40b-B`) | 3.97× | 38.59 | 25.70 | 26.54 | 76.00 | 88.87 | 47.57 | 59.32 | 60.94 | 52.94 |
| RotateKV 4-bit | 4.00× | 39.62 | 24.91 | 26.48 | 77.50 | 88.31 | 46.94 | 59.88 | 61.49 | 53.14 |
| TurboQuant K4V4 | 4.00× | 37.77 | 25.59 | 25.73 | 76.00 | 88.23 | 46.25 | 58.69 | 61.29 | 52.44 |
| **SPECTRA** `4.6x` (`s45`) | 4.57× | 37.38 | 25.65 | 26.75 | 75.50 | 89.05 | 47.05 | 60.37 | 61.84 | 52.95 |
| RotateKV 3-bit | 5.33× | 34.46 | 25.75 | 26.63 | 76.00 | 87.90 | 45.93 | 59.98 | 61.96 | 52.33 |
| TurboQuant K4V2 | 5.33× | 35.64 | 24.59 | 26.14 | 76.50 | 89.53 | 46.36 | 56.38 | 59.61 | 51.84 |
| **SPECTRA** `5.5x` (`L48-B`) | 5.55× | 38.40 | 25.46 | 26.24 | 75.50 | 87.42 | 46.04 | 57.00 | 59.35 | 51.93 |
| **SPECTRA** `5.8x` (`L55-B`) | 5.80× | 38.54 | 25.42 | 26.21 | 76.50 | 88.29 | 46.01 | 57.26 | 58.16 | 52.05 |
| **SPECTRA** `6x` (`L65-B`) | 6.07× | 38.09 | 24.60 | 26.15 | 77.00 | 87.80 | 45.88 | 55.90 | 58.65 | 51.76 |
| **SPECTRA** `6.7x` (`L70-B`) | 6.81× | 36.94 | 25.33 | 26.36 | 75.50 | 89.23 | 47.01 | 58.27 | 59.78 | 52.30 |
| OTT K2V2 | 7.11× | 36.55 | 24.66 | 26.36 | 76.00 | 89.47 | 47.03 | 60.25 | 59.79 | 52.51 |
| KIVI K2V2 | 7.11× | 36.94 | 24.43 | 25.86 | 76.00 | 89.27 | 46.83 | 58.75 | 58.96 | 52.13 |
| **SPECTRA** `7.7x` (`t80e-B`) | 7.73× | 36.59 | 25.63 | 26.56 | 76.00 | 89.47 | 47.07 | 57.37 | 57.87 | 52.07 |
| RotateKV 2-bit | 8.00× | 31.06 | 24.24 | 26.04 | 72.50 | 85.35 | 42.75 | 55.79 | 54.37 | 49.01 |
| TurboQuant K2V2 | 8.00× | 34.83 | 23.12 | 25.06 | 71.50 | 85.96 | 44.64 | 50.24 | 51.13 | 48.31 |
| **SPECTRA** `8.8x` (`s95-B`) | 8.93× | 38.67 | 25.72 | 26.61 | 75.00 | 89.10 | 46.80 | 58.86 | 57.78 | 52.32 |
| **SPECTRA** `11x` (`s12-B`) | 11.27× | 36.75 | 25.85 | 26.22 | 71.50 | 87.55 | 45.60 | 56.31 | 56.26 | 50.76 |

## Qwen2.5-7B-Instruct

| Method | KV compression | Qasper | QMSum | MultiNews | TREC | TriviaQA | SAMSum | LCC | RepoBench-P | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| Dense (fp16) | 1.00× | 43.52 | 23.61 | 23.98 | 71.50 | 89.64 | 45.98 | 60.77 | 66.82 | 53.23 |
| **SPECTRA** `s35-B` | 3.48× | 43.26 | 23.81 | 24.11 | 71.50 | 88.85 | 45.52 | 59.35 | 66.90 | 52.91 |
| PolarQuant r4t4v4 | 3.66× | 43.94 | 24.42 | 25.52 | 72.00 | 89.46 | 45.76 | 63.68 | 66.41 | 53.90 |
| TurboQuant K4V4 | 4.00× | 42.65 | 23.53 | 23.89 | 72.00 | 88.38 | 45.72 | 61.34 | 66.33 | 52.98 |
| PolarQuant r3t3v4 | 4.13× | 44.13 | 24.02 | 25.54 | 72.00 | 90.10 | 46.49 | 64.07 | 64.75 | 53.89 |
| **SPECTRA** `L55-B` | 4.36× | 45.15 | 23.55 | 24.30 | 71.50 | 88.93 | 46.65 | 62.59 | 65.63 | 53.54 |
| **SPECTRA** `L48-B` | 4.36× | 45.15 | 23.55 | 24.30 | 71.50 | 88.93 | 46.65 | 62.59 | 65.63 | 53.54 |
| **SPECTRA** `s45-B` | 4.43× | 44.66 | 23.36 | 23.61 | 71.50 | 88.81 | 44.91 | 59.92 | 64.16 | 52.62 |
| PolarQuant r2t2v4 | 4.74× | 38.51 | 22.70 | 24.87 | 69.00 | 89.54 | 44.25 | 60.61 | 57.45 | 50.87 |
| **SPECTRA** `t80e-B` | 4.91× | 42.32 | 23.38 | 24.01 | 68.00 | 89.49 | 45.14 | 60.69 | 60.78 | 51.73 |
| **SPECTRA** `s65` | 4.92× | 42.96 | 23.40 | 23.05 | 70.00 | 88.83 | 46.04 | 59.76 | 64.14 | 52.27 |
| TurboQuant K4V2 | 5.33× | 42.56 | 23.27 | 23.53 | 72.00 | 88.29 | 45.30 | 55.14 | 59.88 | 51.25 |
| **SPECTRA** `s12` | 5.64× | 39.11 | 23.66 | 24.09 | 71.50 | 86.77 | 46.26 | 60.47 | 54.90 | 50.84 |
| PolarQuant r2t2v2 | 6.74× | 37.63 | 23.09 | 25.41 | 67.50 | 88.08 | 44.44 | 58.11 | 56.74 | 50.12 |
| **SPECTRA** `w544-B` | 7.65× | 31.01 | 23.42 | 25.52 | 70.00 | 89.69 | 45.17 | 53.04 | 51.87 | 48.72 |
| **SPECTRA** `w512h-B` | 8.00× | 31.05 | 23.04 | 25.58 | 70.50 | 90.11 | 43.49 | 53.40 | 51.49 | 48.58 |
| **SPECTRA** `w512-B` | 8.00× | 32.53 | 23.16 | 25.14 | 72.50 | 89.48 | 44.04 | 50.15 | 50.74 | 48.47 |
| TurboQuant K2V2 | 8.00× | 36.18 | 21.85 | 22.25 | 66.50 | 81.39 | 42.92 | 55.44 | 50.93 | 47.18 |
| **SPECTRA** `w448-B` | 8.82× | 34.18 | 21.85 | 25.24 | 70.00 | 88.20 | 43.29 | 52.13 | 46.50 | 47.67 |

`L48-B` and `L55-B` give the same model on Qwen2.5-7B: both latent widths are capped at its KV width.

## Needle in a haystack

6 contexts (1K, 2K, 4K, 8K, 16K, 32K) × 7 depths (0–100%), one prompt per cell, greedy, 24 new tokens (32 with 4
needles). A cell counts when the planted code appears in the output (4 needles: the fraction of the 4 codes found).
SPECTRA ratios are those of the LongBench tables above; see [`spectra/eval/niah.py`](../spectra/eval/niah.py).

| Model | Method | single needle (%) | 4 needles (%) |
|---|---|---|---|
| Llama-3.1-8B | Dense (fp16) | 100.0 | 85.71 |
| Llama-3.1-8B | **SPECTRA** `3.6x` (`s35`), 3.56× | 100.0 | 84.52 |
| Llama-3.1-8B | **SPECTRA** `6x` (`L65-B`), 6.02× | 100.0 | 85.12 |
| Llama-3.1-8B | **SPECTRA** `8.8x` (`s95-B`), 8.82× | 100.0 | 94.64 |
| Llama-3.1-8B | **SPECTRA** `11x` (`s12-B`), 11.09× | 100.0 | 100.0 |
| Llama-3.1-8B | PolarQuant r2t2v4, 4.7× | 50.0 | 42.86 |
| Llama-3.1-8B | KIVI K2V2, 7.1× | 100.0 | 93.45 |
| Llama-3.1-8B | OTT K2V2, 7.1× | 100.0 | 88.1 |
| Llama-3.1-8B | RotateKV 2-bit, 8.0× | 97.62 | 81.55 |
| Llama-3.1-8B | TurboQuant K2V2, 8.0× | 88.1 | 63.1 |
| Mistral-7B | Dense (fp16) | 100.0 | 66.67 |
| Mistral-7B | **SPECTRA** `3.6x` (`s35`), 3.56× | 100.0 | 66.67 |
| Mistral-7B | **SPECTRA** `6x` (`L65-B`), 6.07× | 100.0 | 65.48 |
| Mistral-7B | **SPECTRA** `8.8x` (`s95-B`), 8.93× | 100.0 | 67.86 |
| Mistral-7B | **SPECTRA** `11x` (`s12-B`), 11.27× | 100.0 | 70.83 |
| Mistral-7B | KIVI K2V2, 7.1× | 92.86 | 52.38 |
| Mistral-7B | OTT K2V2, 7.1× | 95.24 | 56.55 |
| Mistral-7B | RotateKV 2-bit, 8.0× | 100.0 | 56.55 |
| Mistral-7B | TurboQuant K2V2, 8.0× | 66.67 | 47.02 |
| Qwen2.5-7B | Dense (fp16) | 100.0 | 76.19 |
| Qwen2.5-7B | **SPECTRA** `s35-B`, 3.48× | 100.0 | 75.6 |
| Qwen2.5-7B | **SPECTRA** `s45-B`, 4.43× | 100.0 | 75.6 |
| Qwen2.5-7B | **SPECTRA** `s65`, 4.92× | 100.0 | 78.57 |
| Qwen2.5-7B | **SPECTRA** `s12`, 5.64× | 100.0 | 76.19 |
| Qwen2.5-7B | PolarQuant r2t2v4, 4.7× | 95.24 | 74.4 |
| Qwen2.5-7B | TurboQuant K2V2, 8.0× | 100.0 | 66.67 |

## Baselines

Each baseline runs from its released code under the same LongBench and NIAH protocol, with these settings and fixes.
The adapters are not part of this repo.

| Method | Settings |
|---|---|
| KIVI | keys per channel, values per token, group 128, fp16 residual of 128 tokens; ratio counts scales and zeros |
| OTT | KIVI-style layout with outlier-token tracking (sink 5 / 32), Triton kernels |
| PolarQuant | keys in polar form (rbits / tbits, group 128) with the upstream mixed K+V attention, values KIVI-style per token |
| RotateKV | uniform bits after rotation (model-specific key); ratio counts scales and zeros |
| TurboQuant | a public TurboQuantMSE port, K / V bits and an fp16 residual of 128 tokens; on Qwen it quantizes bias-free keys and re-adds the k_proj bias at read |

KIVI, OTT and RotateKV have no Qwen2.5 path in their released code, so Qwen2.5-7B has PolarQuant and TurboQuant only.
PolarQuant has no Mistral path.

## Notes

- **Calibration recipe.** arXiv v1 calibrated on WikiText-2 only (`calib="wikitext"`). Adding code windows changes
  most presets by less than a point either way; on Llama-3.1-8B `s12-B` it adds 4.5 points (47.90 → 52.44), mostly
  on LCC and RepoBench-P.
- **Exact window in batched runs.** Generation is batched with left padding. The 4 exact "sink" positions of a `-B`
  config are then the first 4 positions of the padded batch, so for shorter prompts in a batch they fall on padding.
  All numbers above were produced this way, and the harness keeps it so they reproduce. The ratio always counts the
  68 exact tokens per sample.
