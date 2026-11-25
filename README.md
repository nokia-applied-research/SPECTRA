<div align="center">

# SPECTRA

**Cache the latent, not the keys and values.**

Training-free KV-cache compression by spectral transform coding, for 🤗 Transformers.

[![arXiv](https://img.shields.io/badge/arXiv-2608.07915-b31b1b.svg)](https://arxiv.org/abs/2608.07915)
[![Transformers](https://img.shields.io/badge/transformers-%E2%89%A54.57-ffcc4d.svg)](#quick-start)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-3776ab.svg)](#installation)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Paper](https://arxiv.org/abs/2608.07915) · [Quick start](#quick-start) · [Results](#results) · [Presets](#presets) · [Reproduce](#reproduce) · [TODO](#todo) · [Citation](#citation)

</div>

---

Every layer of an LLM caches a key and a value vector for every token, and this KV cache often limits how long a
context or how many requests a GPU can hold. Most compressors keep the keys and values and store them with fewer bits.
Below about 2 bits per value, these scalar and rotation quantizers break down. **SPECTRA** changes what is cached. It
maps each layer's keys and values into a latent whose channels are decorrelated and ordered by energy, and then gives
the bits to the channels that carry the most energy. Low-energy channels get zero bits, so the method moves from
quantization to low rank smoothly. This is transform coding, as in JPEG, applied to the KV cache.

The latent comes from an SVD of `k_proj` and `v_proj`, weighted by the activations the model actually produces. The
transform folds into the projections, `o_proj` is corrected in closed form, and nothing is trained.

<p align="center">
  <img src="assets/overview.png" width="92%" alt="SPECTRA: a G-weighted SVD gives an energy-ordered latent, water-filling allocates bits per channel group, and only the quantized latent is cached">
</p>
<p align="center"><sub>(a) The latent comes from an SVD of the projection weights, whitened by the activation
second moment G. (b) Reverse water-filling gives each channel group its bits; groups under the water level get zero
bits. (c) At decode only the quantized latent is stored; keys and values are rebuilt from it.</sub></p>

## News

- **2026-11**: v0.2. The paper's operating points as named presets, with its calibration recipe (WikiText-2 + code)
  and exact-window option. LongBench now runs the official prompts and scorer on the full test split. Compression
  ratios count the exact window. The in-house baseline re-implementations are removed.
- **2026-08**: v0.1, open-source release.

## What is in this repo

| | What it stores | Use it for |
|---|---|---|
| [`spectra.compress`](#quick-start) (🤗 Transformers) | keys and values rebuilt from the quantized latent, in the stock fp16 cache | accuracy: the paper's numbers |
| [Packed INT4 cache](#memory-packed-int4-cache) (🤗 Transformers, vLLM 0.7) | the INT4 latent itself | memory: about 7× more tokens per GPU |

SPECTRA reduces memory, not decode time. Every path that stores the latent rebuilds the keys and values from it at
each decoding step, which is slower than dense attention.

## Installation

```bash
git clone https://github.com/nokia-applied-research/SPECTRA.git
cd SPECTRA
pip install -e .
pip install -e '.[eval]'          # LongBench metrics (rouge, fuzzywuzzy, jieba)
```

You need Python ≥ 3.10, PyTorch ≥ 2.5, transformers ≥ 4.57 and `datasets` < 4 (version 4 dropped the script loader
that LongBench needs). Tested on H100 with torch 2.11, transformers 5.11 and datasets 3.6
([`environment.yml`](environment.yml)).

---

## Quick start

```python
import spectra

model, tok = spectra.compress("NousResearch/Meta-Llama-3.1-8B-Instruct", preset="s95-B")

messages = [{"role": "user", "content": "Summarize the following report: ..."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                 return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=256)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))

spectra.save(model, "llama31_s95B")        # small delta: factors and the corrected o_proj
model, tok = spectra.load("llama31_s95B")  # fetches the base model by id and applies the delta
```

`compress` reads 16 calibration windows of 2,048 tokens (8 from WikiText-2, 8 from Python code), fits every layer in
closed form, and replaces `k_proj` and `v_proj` in place. It prints the ratio of the latent and, for `-B` presets, the
ratio with the exact window counted at 8K and 32K tokens.

The same from the command line:

```bash
python -m spectra.api compress --model NousResearch/Meta-Llama-3.1-8B-Instruct --preset s12-B --out llama31_s12B
python -m spectra.api generate --path llama31_s12B --prompt "The KV cache is"
```

**What `-B` means.** A preset with the `-B` suffix keeps the first 4 and the last 64 prompt tokens of K and V exact
(fp16) and compresses everything else. Generated tokens are always compressed. The 68 exact tokens are counted in
every ratio we report ([`spectra/accounting.py`](spectra/accounting.py)).

---

## Results

LongBench, 8 English tasks (Qasper, QMSum, MultiNews, TREC, TriviaQA, SAMSum, LCC, RepoBench-P): official prompts
and scorer, full test split of every task, greedy decoding. KV compression counts everything a method stores (scales,
zero points, exact windows) and is averaged over the evaluated samples by bytes.

**Llama-3.1-8B-Instruct**, LongBench-8 average:

| Method | KV compression | LongBench-8 |
|---|---|---|
| Dense (fp16) | 1× | 53.29 |
| KIVI K4V4 | 3.76× | 53.92 |
| PolarQuant r4t4v4 | 3.66× | 53.66 |
| SPECTRA s35 | 3.56× | 53.18 |
| OTT K2V2 | 7.11× | 53.30 |
| KIVI K2V2 | 7.11× | 52.17 |
| SPECTRA t80e-B | 7.65× | 53.29 |
| RotateKV 2-bit | 8.00× | 47.00 |
| TurboQuant K2V2 | 8.00× | 46.43 |
| **SPECTRA s95-B** | **8.82×** | **52.99** |
| **SPECTRA s12-B** | **11.09×** | **52.44** |

**Three models**, the strongest baseline at 7× or more against SPECTRA:

| Model | Dense | Best baseline at ≥ 7× | SPECTRA at ~8–9× | SPECTRA, highest ratio |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 53.29 | OTT K2V2, 7.11×: 53.30 | s95-B, 8.82×: 52.99 | s12-B, 11.09×: 52.44 |
| Mistral-7B-Instruct-v0.3 | 52.78 | OTT K2V2, 7.11×: 52.51 | s95-B, 8.93×: 52.32 | s12-B, 11.27×: 50.76 |
| Qwen2.5-7B-Instruct | 53.23 | TurboQuant K2V2, 8.00×: 47.18 | w512h-B, 8.00×: 48.58 | w448-B, 8.82×: 47.67 |

- **Up to about 7×**, SPECTRA is on par with the strongest quantizers (KIVI, OTT, PolarQuant), not ahead of them.
- **Past 8×**, SPECTRA is the only method with usable points. No baseline reaches beyond 8×, and at 8× the
  2-bit quantizers lose 6–7 points on Llama.
- **Qwen2.5-7B** has 4 KV heads, so its cache is already half as wide and harder to compress. The `w*` presets
  narrow the latent to reach the 8× tier; there SPECTRA is about 4.7 points below dense.

**Needle in a haystack**, 6 contexts (1K–32K) × 7 depths, % of cells retrieved (single needle / 4 needles):

| | Llama-3.1-8B | Mistral-7B | Qwen2.5-7B |
|---|---|---|---|
| Dense (fp16) | 100 / 85.7 | 100 / 66.7 | 100 / 76.2 |
| SPECTRA, highest-ratio preset (s12-B; Qwen: s12) | **100 / 100** | **100 / 70.8** | **100 / 76.2** |
| KIVI K2V2 | 100 / 93.5 | 92.9 / 52.4 | — |
| TurboQuant K2V2 | 88.1 / 63.1 | 66.7 / 47.0 | 100 / 66.7 |

Every SPECTRA preset we tested retrieves the single needle in every cell, on all three models.

The baseline rows come from each method's released code (TurboQuant: a public port), run under the same protocol
with small compatibility fixes. They are not part of this repo. KIVI, OTT and RotateKV have no Qwen2.5 path in their
released code. The paper has the full tables, including the per-task scores.

---

## Presets

LongBench-8 average and KV compression of every preset in the paper. The same preset compresses differently on
different models, because the latent width is capped by the model's KV width and the window costs more on short
contexts.

| Preset | Kind | Llama-3.1-8B | Mistral-7B | Qwen2.5-7B |
|---|---|---|---|---|
| dense | | 1× · 53.29 | 1× · 52.78 | 1× · 53.23 |
| `s35` | water-filling, 4 bits | 3.56× · 53.18 | 3.56× · 52.81 | |
| `s35-B` | water-filling, 4 bits | | | 3.48× · 52.91 |
| `t40b-B` | water-filling, 4 bits | 3.95× · 52.94 | 3.97× · 52.94 | |
| `s45` | water-filling, 3 bits | 4.57× · 52.62 | 4.57× · 52.95 | |
| `s45-B` | water-filling, 3 bits | | | 4.43× · 52.62 |
| `s65` | water-filling, 3 bits | | | 4.92× · 52.27 |
| `L48-B` | low rank, 3 bits | 5.51× · 51.74 | 5.55× · 51.93 | 4.36× · 53.54 |
| `L55-B` | low rank, 3 bits | 5.75× · 51.95 | 5.80× · 52.05 | 4.36× · 53.54 |
| `L65-B` | low rank, 3 bits | 6.02× · 52.59 | 6.07× · 51.76 | |
| `L70-B` | low rank, 3 bits | 6.74× · 52.88 | 6.81× · 52.30 | |
| `t80e-B` | low rank, 3 bits | 7.65× · 53.29 | 7.73× · 52.07 | 4.91× · 51.73 |
| `s95-B` | low rank, 3 bits | 8.82× · 52.99 | 8.93× · 52.32 | |
| `s12-B` | low rank, 3 bits, Hadamard | 11.09× · 52.44 | 11.27× · 50.76 | |
| `s12` | low rank, 3 bits, Hadamard | | | 5.64× · 50.84 |
| `w544-B` | low rank, 3 bits | | | 7.65× · 48.72 |
| `w512-B` | low rank, 3 bits | | | 8.00× · 48.47 |
| `w512h-B` | low rank, 3 bits, Hadamard | | | 8.00× · 48.58 |
| `w448-B` | low rank, 3 bits | | | 8.82× · 47.67 |

Empty cells were not run. Water-filling presets allocate bits per channel group by reverse water-filling; low-rank
presets keep the top channels, the first 32 of them at 8 bits. The configuration of each preset is in
[`spectra/api.py`](spectra/api.py), and `B=` / `bits=` give a custom point. The v0.1 names `3x`, `5x`, `6x`, `9x` and
`12x` still work and map to `s35`, `s45`, `s65`, `s95` and `s12`.

All presets use the calibration recipe `wikitext+code`. The recipe of arXiv v1 (WikiText-2 only) is
`calib="wikitext"`. On most presets the two are within a point; on Llama `s12-B` the code windows add 4.5 points,
mostly on the two code tasks.

---

## Reproduce

```bash
M=NousResearch/Meta-Llama-3.1-8B-Instruct

# LongBench-8, full test split (about 1 h per run on one GPU)
python scripts/run_longbench.py --model $M --out runs/l31_dense.json                    # 53.29
python scripts/run_longbench.py --model $M --preset s12-B --out runs/l31_s12B.json      # 52.44, 11.09x

# needle in a haystack, single needle and 4 needles
python scripts/run_niah.py --model $M --preset s12-B --out runs/l31_s12B_niah.json              # 100
python scripts/run_niah.py --model $M --preset s12-B --needles 4 --out runs/l31_s12B_niah4.json # 100
```

The LongBench output records the score and the number of samples per task, the context length of every sample, and
the compression over those samples. Generation is batched (left-padded, up to 16 prompts), as in the paper's runs.

WikiText-2 perplexity and zero-shot accuracy (`lm-eval`) for custom points:

```bash
python -m spectra.method --model $M --configs 1024:3 --heavy-k 32 --kv-ratio 0.33 --group 128 --out runs/ppl.json
```

---

## Memory: packed INT4 cache

`spectra.compress` measures accuracy; the stock cache still holds fp16 keys and values. To store the latent itself,
use the symmetric INT4 preset `vllm-7x` (about 7× on Llama-3.1-8B) with one of these paths:

```python
import spectra, spectra.vllm

m, tok = spectra.compress("NousResearch/Meta-Llama-3.1-8B-Instruct", preset="vllm-7x")
spectra.save(m, "llama31_vllm7x")

# vLLM 0.7.x (+ transformers 4.48.x): the INT4 latent lives in vLLM's paged KV cache
llm = spectra.vllm.LLM("NousResearch/Meta-Llama-3.1-8B-Instruct", "llama31_vllm7x",
                       gpu_memory_utilization=0.85, max_model_len=32768)
```

`python demo/vllm_demo.py build ...` then `... serve ...` prints the KV capacity against fp16. In 🤗 Transformers,
`spectra.hf_serve.enable(model, spectra.hf_serve.SpectraCache())` stores the packed latent in place of the stock cache,
and [`modeling_spectra.py`](modeling_spectra.py) is a self-contained Llama with the same cache
(`trust_remote_code`).

### Limits (v0.2)

- Decode is slower than dense on both packed paths: keys and values are rebuilt from the latent at every step, in
  PyTorch.
- The packed paths support the symmetric INT4 preset (`vllm-7x`) on Llama-family models. The water-filling and
  mixed-precision presets of the table are evaluated through `spectra.compress` only.
- vLLM 0.7.x only, with `enforce_eager=True`.
- With left-padded batches, the 4 exact "sink" positions of a `-B` preset are the first 4 positions of the padded
  batch, so for shorter prompts in a batch they fall on padding. The paper's batched runs behave the same way.

---

## Tests

```bash
python tests/test_cpu.py     # accounting, quantizers, the exact window, presets (CPU, no download)
bash demo/demo.sh            # compress + generate + WikiText-2 perplexity on one GPU
```

## TODO

- ✅ SPECTRA in 🤗 Transformers: the paper's presets, calibration recipe and exact window
- ✅ LongBench-8 (official prompts and scorer, full test split) and needle-in-a-haystack protocols of the paper
- ✅ Packed INT4 latent cache in 🤗 Transformers and vLLM 0.7
- ⬜ A decode path faster than dense: attention on the latent without rebuilding the keys and values
- ⬜ Packed storage for the water-filling and mixed-precision presets
- ⬜ Current vLLM releases
- ⬜ PyPI package

## Citation

```bibtex
@misc{zhang2026spectrapushingkvcache,
      title={SPECTRA: Pushing the KV Cache Beyond the 2-Bit Cliff via Spectral Transform Coding}, 
      author={Jiamu Zhang and Liang Wu and Kelly Wan and Hanjie Chen and Liangjie Hong},
      year={2026},
      eprint={2608.07915},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2608.07915}, 
}
```

## License

SPECTRA is released under the [MIT](LICENSE) license. `spectra/eval/metrics.py` and `spectra/eval/config/` are
copied from [THUDM/LongBench](https://github.com/THUDM/LongBench) (MIT).

## Acknowledgements

SPECTRA builds on [🤗 Transformers](https://github.com/huggingface/transformers) and
[vLLM](https://github.com/vllm-project/vllm). Evaluation uses [LongBench](https://github.com/THUDM/LongBench);
calibration uses [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) and
[codeparrot-clean-valid](https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid).
