# Presets

A preset is one of the paper's operating points. Its public name is the KV compression it reaches on
Llama-3.1-8B-Instruct (LongBench, exact window counted). Every preset also has an internal config name, which the
paper and [results.md](results.md) use. `spectra.compress` accepts both.

```python
spectra.compress(model, tok, preset="8.8x")    # same as preset="s95-B"
```

## Public presets

| Preset | Config | Kind | Latent width B | Bits | Exact window | Llama-3.1-8B | Mistral-7B |
|---|---|---|---|---|---|---|---|
| `3.6x` | `s35` | water-filling | 2048 | 4 | no | 3.56× · 53.18 | 3.56× · 52.81 |
| `4x` | `t40b-B` | water-filling | 1792 | 4 | yes | 3.95× · 52.94 | 3.97× · 52.94 |
| `4.6x` | `s45` | water-filling | 2048 | 3 | no | 4.57× · 52.62 | 4.57× · 52.95 |
| `5.5x` | `L48-B` | low rank | 2048 | 3 | yes | 5.51× · 51.74 | 5.55× · 51.93 |
| `5.8x` | `L55-B` | low rank | 1792 | 3 | yes | 5.75× · 51.95 | 5.80× · 52.05 |
| `6x` | `L65-B` | low rank | 1536 | 3 | yes | 6.02× · 52.59 | 6.07× · 51.76 |
| `6.7x` | `L70-B` | low rank | 1408 | 3 | yes | 6.74× · 52.88 | 6.81× · 52.30 |
| `7.7x` | `t80e-B` | low rank | 1216 | 3 | yes | 7.65× · 53.29 | 7.73× · 52.07 |
| `8.8x` | `s95-B` | low rank | 1024 | 3 | yes | 8.82× · 52.99 | 8.93× · 52.32 |
| `11x` | `s12-B` | low rank + rotation | 768 | 3 | yes | 11.09× · 52.44 | 11.27× · 50.76 |

Cells: KV compression · LongBench-8 average. Dense: 53.29 (Llama), 52.78 (Mistral).

## Qwen2.5-7B

Qwen2.5-7B has 4 KV heads, so its fp16 KV cache is half as wide (1024 values per token per layer against 2048), and
the latent width of a preset is capped at that width. The same preset therefore compresses much less, and the ratio
names do not apply. These configs were run on Qwen2.5-7B (dense: 53.23):

| Config | Kind | Latent width B | Bits | Exact window | Qwen2.5-7B |
|---|---|---|---|---|---|
| `s35-B` | water-filling | 2048 | 4 | yes | 3.48× · 52.91 |
| `L48-B`, `L55-B` | low rank | 2048, 1792 (both capped) | 3 | yes | 4.36× · 53.54 |
| `s45-B` | water-filling | 2048 | 3 | yes | 4.43× · 52.62 |
| `t80e-B` | low rank | 1216 | 3 | yes | 4.91× · 51.73 |
| `s65` | water-filling | 1536 | 3 | no | 4.92× · 52.27 |
| `s12` | low rank + rotation | 768 | 3 | no | 5.64× · 50.84 |
| `w544-B` | low rank | 544 | 3 | yes | 7.65× · 48.72 |
| `w512-B` | low rank | 512 | 3 | yes | 8.00× · 48.47 |
| `w512h-B` | low rank + rotation | 512 | 3 | yes | 8.00× · 48.58 |
| `w448-B` | low rank | 448 | 3 | yes | 8.82× · 47.67 |

The `w` configs fix a narrow latent to reach the 8× tier on narrow-KV models. On an 8-KV-head model they compress
far more and have not been evaluated there.

## What the fields mean

All values are in [`spectra/api.py`](../spectra/api.py) (`PRESETS`, `NAMED`).

- **Latent width B**: the number of latent channels per token per layer, summed over K and V. Each side is capped at
  the model's KV width (1024 on Llama-3.1-8B and Mistral-7B).
- **Water-filling** (`mode="wf"`): K and V get half of B each. The latent is split into groups of 64 or 128 channels
  in energy order, and each group gets its bits by reverse water-filling around the average `bits`; low-energy groups
  get 0 bits. The latent is mean-centred before quantization.
- **Low rank** (`mode="lowrank"`): K gets a third of B and V two thirds. All kept channels are quantized at `bits`,
  except the 32 highest-energy channels of each side, which are kept at 8 bits. `+ rotation` applies a fixed random
  orthogonal rotation to the other channels before quantization (folded into the factors; `hadamard=True` in the
  config).
- **Quantization**: per token, per group of 64 or 128 channels, asymmetric min-max, with an fp16 scale and zero
  point per group. The compression ratio counts these.
- **Exact window** (configs ending in `-B`): the first 4 and the last 64 prompt tokens of K and V stay in fp16.
  Generated tokens are always compressed. The ratio counts the 68 exact tokens per sequence, so it is lower on short
  contexts ([`spectra/accounting.py`](../spectra/accounting.py)).
- **W_O correction**: after the factorization, `o_proj` gets a rank-4 closed-form correction fitted on the
  calibration windows.

## Calibration

`calib="wikitext+code"` (default, used for every number in this repo): 16 windows of 2,048 tokens, 8 from the
WikiText-2 train split and 8 from codeparrot-clean-valid. `calib="wikitext"`: 16 WikiText-2 windows, the recipe of
arXiv v1. `calib_text=...` uses your own text. The fit is closed form; nothing is trained.

## Custom points and older names

- `spectra.compress(model, tok, B=1280, bits=3)` builds a custom point (water-filling for `bits >= 4`, low rank
  below), and `window=True` adds the exact window.
- `preset="vllm-7x"`: a uniform 4-bit latent of width half the KV width, about 7× on Llama-3.1-8B. This is
  the format the packed INT4 caches (`spectra.hf_serve`, `spectra.vllm`, `modeling_spectra.py`) store.
- The v0.1 names `3x`, `5x`, `9x` and `12x` still work. They map to `s35`, `s45`, `s95` and `s12` without the exact
  window, so `9x` and `12x` give 9.48× and 12.19× latent ratios. In v0.1, `6x` was `s65`; in v0.2 `6x` is `L65-B`.
