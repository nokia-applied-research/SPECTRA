<div align="center">

# SPECTRA

**Cache the latent, not the keys and values.**

KV-cache compression by spectral transform coding, for 🤗 Transformers and vLLM.

[![arXiv](https://img.shields.io/badge/arXiv-2608.07915-b31b1b.svg)](https://arxiv.org/abs/2608.07915)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20models-SPECTRA--Flash-ffcc4d.svg)](#released-model)
[![vLLM](https://img.shields.io/badge/vLLM-0.29-30a2ff.svg)](#serving-with-vllm)
[![Transformers](https://img.shields.io/badge/transformers-%E2%89%A54.57-ffcc4d.svg)](#quick-start)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.10-3776ab.svg)](#installation)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Paper](https://arxiv.org/abs/2608.07915) · [SPECTRA](#spectra-1) · [SPECTRA-Flash](#spectra-flash) · [Results](#results) · [Docs](docs/) · [Citation](#citation)

</div>

---

Every layer of an LLM caches a key and a value vector for every token, and this KV cache often limits how long a
context or how many requests a GPU can hold. Most compressors keep the keys and values and store them with fewer
bits, and they break down below about 2 bits per value. **SPECTRA** changes what is cached. It maps each layer's keys
and values into a latent whose channels are ordered by energy, and gives the bits to the channels that carry the most
energy. Low-energy channels get zero bits, so the method moves from quantization to low rank smoothly. This is
transform coding, as in JPEG, applied to the KV cache.

<p align="center">
  <img src="assets/overview.png" width="92%" alt="SPECTRA: a G-weighted SVD gives an energy-ordered latent, water-filling allocates bits per channel group, and only the quantized latent is cached">
</p>
<p align="center"><sub>(a) The latent comes from an SVD of the key and value projections, weighted by the
activations. (b) Reverse water-filling gives each channel group its bits; groups under the water level get zero bits.
(c) At decode only the quantized latent is stored, and keys and values are rebuilt from it.</sub></p>

## News

- **2026-11**: v0.2. **SPECTRA-Flash**: a 7× smaller KV cache and faster decode than dense, on FlashMLA and vLLM 0.29,
  with converted params for Llama-3.1-8B-Instruct. SPECTRA gets the paper's operating points as presets, and LongBench
  runs the official prompts and scorer on the full test split.
- **2026-08**: v0.1, open-source release.

## What is in this repo

| | KV cache | Decode speed vs dense | Setup |
|---|---|---|---|
| [**SPECTRA**](#spectra-1) | up to 11× smaller | slower: keys and values are rebuilt at every step | none: fitted in closed form in about 2 min, no training |
| [**SPECTRA-Flash**](#spectra-flash) | 7× smaller | 1.0× at batch 1, up to 4.3× throughput | download the converted params (1.5 GB) |

SPECTRA is the training-free, closed-form codec; its speed gain needs a new decode kernel and serving support.
SPECTRA-Flash starts from the same fit, adds a light tuning step (2.6 GPU hours), and stores the latent in the layout
that existing kernels read, so it is fast today.

## Installation

```bash
git clone https://github.com/nokia-applied-research/SPECTRA.git
cd SPECTRA
pip install -e .
pip install -e '.[eval]'          # LongBench metrics
pip install -e '.[flash]'         # SPECTRA-Flash: vLLM 0.29 (its FlashMLA and kernels); Hopper GPU
```

Python ≥ 3.10, PyTorch ≥ 2.5, transformers ≥ 4.57. SPECTRA was tested on H100 with torch 2.11 and transformers 5.11
([`environment.yml`](environment.yml)); SPECTRA-Flash with vLLM 0.29 (torch 2.13, transformers 5.20) on H100 NVL.

---

## SPECTRA

### Quick start

```python
import spectra

model, tok = spectra.compress("NousResearch/Meta-Llama-3.1-8B-Instruct", preset="8.8x")

messages = [{"role": "user", "content": "Summarize the following report: ..."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                 return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=256)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))

spectra.save(model, "llama31_8.8x")        # the factors and the corrected o_proj (1.3 GB for this model)
model, tok = spectra.load("llama31_8.8x")  # fetches the base model by id and applies them
```

`compress` reads 16 calibration windows of 2,048 tokens (WikiText-2 and Python code) and fits every layer in closed
form. It simulates the compressed cache, so it measures accuracy. Each preset is named by the KV compression it
reaches on Llama-3.1-8B:

| Preset | KV compression | LongBench-8 |
|---|---|---|
| dense | 1× | 53.29 |
| `"3.6x"` | ~3.6× | 53.18 |
| `"4.6x"` | ~4.6× | 52.62 |
| `"6x"` | ~6× | 52.59 |
| `"7.7x"` | ~7.7× | 53.29 |
| `"8.8x"` | ~8.8× | 52.99 |
| `"11x"` | ~11× | 52.44 |

Mistral-7B compresses about the same. Qwen2.5-7B has a narrower KV cache and compresses less with the same preset. All
presets, what they set, and the Qwen2.5 settings are in [docs/presets.md](docs/presets.md).

### Packed INT4 cache

To store the latent itself, use the preset `vllm-7x` (about 7× on Llama-3.1-8B) with `spectra.hf_serve` (🤗
Transformers), [`modeling_spectra.py`](modeling_spectra.py) (a self-contained Llama), or `spectra.vllm` (vLLM 0.7.x):

```python
import spectra, spectra.vllm

m, tok = spectra.compress("NousResearch/Meta-Llama-3.1-8B-Instruct", preset="vllm-7x")
spectra.save(m, "llama31_vllm7x")
llm = spectra.vllm.LLM("NousResearch/Meta-Llama-3.1-8B-Instruct", "llama31_vllm7x",
                       gpu_memory_utilization=0.85, max_model_len=32768)
```

These paths rebuild the keys and values in PyTorch at every step, so they save memory but decode slower than dense.

---

## SPECTRA-Flash

SPECTRA-Flash is the form of SPECTRA built for decode speed. Each layer keeps one record per token in the layout that
FlashMLA reads natively (512 latent + 64 RoPE dims, fp8). The record maps start from SPECTRA's closed-form fit and are
refined by block-wise distillation; the base model is frozen. At decode, attention reads the fp8 record through
FlashMLA plus a small exact window (the first 4 and the latest 123 tokens), and merges the two in one softmax. Nothing
is reconstructed.

- **7× smaller KV cache.** At 32K tokens, one token takes 18.5 KiB instead of 128 KiB.
- **Faster than dense at decode.** On one H100, a decode step is as fast as dense at batch 1 and 2.36× faster at
  batch 16 (32K context). The freed memory holds 4× the batch, so the best throughput is 4.3× dense (4.5× at 128K).
- **99.3% of dense accuracy** on LongBench-8 ([Results](#results)).
- **The base model is not changed.** The release is a small delta: only the record maps (1.5 GB). The base weights
  come from Meta.

### Released model

| Model | Conversion | LongBench-8 (% of dense) |
|---|---|---|
| [Llama-3.1-8B-Instruct-SPECTRA-Flash](https://huggingface.co/morriszjm/Llama-3.1-8B-Instruct-SPECTRA-Flash) | closed-form init + block-wise distillation, 2.6 H100 GPU hours | 99.3 |

The base model is [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (accept
its license on the Hub first). Built with Llama.

### HuggingFace Transformers

```python
import torch, spectra.flash
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda()

spectra.flash.enable(model, "morriszjm/Llama-3.1-8B-Instruct-SPECTRA-Flash")

messages = [{"role": "user", "content": "Summarize the following report: ..."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                 return_tensors="pt", return_dict=True).to("cuda")
out = model.generate(**inputs, max_new_tokens=256)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))

spectra.flash.disable(model)   # back to the stock attention
```

### Serving with vLLM

The plugin is installed with the package. Select SPECTRA-Flash per model with `--hf-overrides`:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 32768 \
    --no-enable-prefix-caching --no-enable-chunked-prefill --max-num-batched-tokens 32768 \
    --hf-overrides '{"architectures": ["SpectraFlashLlamaForCausalLM"],
                     "spectra_flash": "morriszjm/Llama-3.1-8B-Instruct-SPECTRA-Flash"}'
```

Then query it with any OpenAI client. The default vLLM compilation settings work (torch.compile, CUDA graphs).

**Use exact prefill.** With `--no-enable-chunked-prefill` and `--max-num-batched-tokens` at least the longest prompt,
each prompt is prefilled in one step with exact attention, which is the evaluated setting. With chunked prefill
(vLLM's default), a later chunk sees the earlier chunks only through the compressed record. It runs, but it is an
approximation and was not evaluated.

### Reproduce

```bash
P=morriszjm/Llama-3.1-8B-Instruct-SPECTRA-Flash
M=meta-llama/Llama-3.1-8B-Instruct

# LongBench-8, full test split, through the SPECTRA-Flash engine and its dense reference (batch 1)
python scripts/run_longbench.py --model $M --flash $P --out runs/flash.json           # 53.45
python scripts/run_longbench.py --model $M --flash-dense --out runs/flash_dense.json  # 53.80

# decode speed (a 32K needle check runs first), and vLLM serving at one memory budget
spectra-flash bench decode --params $P
spectra-flash bench serve  --params $P

# convert the base model yourself (51 min on 4 x H100 NVL)
spectra-flash convert --model $M --out spectra-flash-llama31-8b --gpus 0,1,2,3
```

### Limits (v0.2)

- Llama-3.1-8B-Instruct; bfloat16; one GPU (vLLM tensor parallel size 1); Hopper GPUs.
- vLLM: no prefix caching, no speculative decoding, no KV connectors.
- 🤗 Transformers: batches must be unpadded (batch size 1 or equal-length prompts).

---

## Results

LongBench, 8 English tasks, official prompts and scorer, full test split of every task. KV compression counts
everything a method stores, including scales and the tokens kept exact.

**Llama-3.1-8B-Instruct**, LongBench-8 average (🤗 Transformers, fp16):

| Method | KV compression | LongBench-8 |
|---|---|---|
| Dense (fp16) | 1× | 53.29 |
| KIVI | ~3.8× | 53.92 |
| **SPECTRA** | ~3.6× | 53.18 |
| OTT | ~7.1× | 53.30 |
| KIVI | ~7.1× | 52.17 |
| **SPECTRA** | ~7.7× | 53.29 |
| RotateKV | ~8× | 47.00 |
| TurboQuant | ~8× | 46.43 |
| **SPECTRA** | **~8.8×** | **52.99** |
| **SPECTRA** | **~11×** | **52.44** |

**Three models**, the strongest baseline at 7× or more against SPECTRA:

| Model | Dense | Best baseline at ≥ 7× | SPECTRA at ~8–9× | SPECTRA, highest ratio |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | 53.29 | 53.30 (OTT, ~7.1×) | 52.99 (~8.8×) | 52.44 (~11×) |
| Mistral-7B-Instruct-v0.3 | 52.78 | 52.51 (OTT, ~7.1×) | 52.32 (~8.9×) | 50.76 (~11×) |
| Qwen2.5-7B-Instruct | 53.23 | 47.18 (TurboQuant, ~8×) | 48.58 (~8×) | 47.67 (~8.8×) |

- **Up to about 7×**, SPECTRA is on par with the strongest quantizers, not ahead of them.
- **Past 8×**, only SPECTRA stays usable. No baseline goes beyond 8×, and at 8× the baselines lose 6–7 points on
  Llama.
- **Qwen2.5-7B** has 4 KV heads, so its cache is already narrow. SPECTRA reaches 8× there at about 4.7 points below
  dense.

**SPECTRA-Flash**, Llama-3.1-8B-Instruct, LongBench-8 as % of the dense model on the same backend:

| Method | KV compression | % of dense |
|---|---|---|
| **SPECTRA-Flash** | ~6.4× | **99.3** |
| SPECTRA | ~6.7× | 99.2 |
| SPECTRA | ~7.7× | 100.0 |
| SPECTRA | ~11× | 98.4 |

SPECTRA-Flash runs in its decode engine (bf16, dense 53.80); SPECTRA in 🤗 Transformers (fp16, dense 53.29). Its 6.4×
is lower than 7× because LongBench prompts are short (median 5.7K tokens), so the 127 exact tokens weigh more.

**Speed**, Llama-3.1-8B-Instruct at 32K context, one H100 NVL, relative to dense:

| | KV cache per token | Decode step, batch 1 | Max decode throughput | vLLM serving throughput |
|---|---|---|---|---|
| Dense | 128 KiB | 1.00× | 1.00× | 1.00× |
| SPECTRA (~11×) | 10.7 KiB | — | 0.04× | — |
| **SPECTRA-Flash** | **18.5 KiB** | **1.00×** | **4.3×** | **1.21–2.26×** |

Max decode throughput: each method at the largest batch its cache fits. SPECTRA: its rebuild decode path in 🤗
Transformers at the memory of the dense cache. SPECTRA-Flash: its decode engine with CUDA graphs, dense with
FlashAttention-3. vLLM serving: all requests submitted at once at one memory budget, 32K prompts + 256 output tokens
(1.21×) and 8K prompts + 2K output tokens (2.26×).

**Needle in a haystack**, contexts 1K–32K × 7 depths, % of cells retrieved (one needle / four needles):

| | Llama-3.1-8B | Mistral-7B | Qwen2.5-7B |
|---|---|---|---|
| Dense (fp16) | 100 / 85.7 | 100 / 66.7 | 100 / 76.2 |
| **SPECTRA**, highest ratio tested | **100 / 100** | **100 / 70.8** | **100 / 76.2** |
| KIVI, ~7.1× | 100 / 93.5 | 92.9 / 52.4 | — |
| TurboQuant, ~8× | 88.1 / 63.1 | 66.7 / 47.0 | 100 / 66.7 |

SPECTRA here is ~11× on Llama and Mistral and ~5.6× on Qwen2.5. Every SPECTRA preset we tested finds the single
needle in every cell, on all three models. SPECTRA-Flash finds the 32K needle.

Baselines run from their released code under the same protocol; their adapters are not part of this repo. Per-task
scores, every preset, the baseline settings and all NIAH rows are in [docs/results.md](docs/results.md).

### Reproduce (SPECTRA)

```bash
M=NousResearch/Meta-Llama-3.1-8B-Instruct

# LongBench-8, full test split (about 1 h per run on one GPU)
python scripts/run_longbench.py --model $M --out runs/dense.json                # 53.29
python scripts/run_longbench.py --model $M --preset 11x --out runs/11x.json     # 52.44, 11.09x

# needle in a haystack, one needle and four needles
python scripts/run_niah.py --model $M --preset 11x --out runs/niah.json                 # 100
python scripts/run_niah.py --model $M --preset 11x --needles 4 --out runs/niah4.json    # 100
```

---

## Tests

```bash
# SPECTRA
python tests/test_cpu.py                    # accounting, quantizers, presets (CPU, no download)
python tests/check_roundtrip.py             # compress -> save -> load gives the same model
# SPECTRA-Flash
python tests/test_flash_kernels.py          # Triton kernels vs PyTorch
python tests/check_flash_engine.py --params morriszjm/Llama-3.1-8B-Instruct-SPECTRA-Flash   # needle, engine vs 🤗
python tests/check_flash_short.py --params morriszjm/Llama-3.1-8B-Instruct-SPECTRA-Flash    # short prompts vs dense
```

## TODO

- ✅ SPECTRA in 🤗 Transformers with the paper's presets; the paper's LongBench and NIAH protocols
- ✅ Packed INT4 latent cache in 🤗 Transformers and vLLM 0.7
- ✅ SPECTRA-Flash: conversion, 🤗 Transformers path, vLLM 0.29 plugin, converted Llama-3.1-8B-Instruct params
- ⬜ SPECTRA-Flash for more models (Mistral, Qwen2.5)
- ⬜ A SPECTRA decode kernel that attends on the latent without rebuilding the keys and values
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

SPECTRA is released under the [MIT](LICENSE) license. `spectra/eval/metrics.py` and `spectra/eval/config/` are copied
from [THUDM/LongBench](https://github.com/THUDM/LongBench) (MIT). The SPECTRA-Flash params are derived from Llama 3.1
and are distributed under the [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/).

## Acknowledgements

SPECTRA builds on [🤗 Transformers](https://github.com/huggingface/transformers),
[vLLM](https://github.com/vllm-project/vllm), [FlashMLA](https://github.com/deepseek-ai/FlashMLA) and
[Triton](https://github.com/triton-lang/triton). Evaluation uses [LongBench](https://github.com/THUDM/LongBench);
calibration uses [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) and
[codeparrot-clean-valid](https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid).
