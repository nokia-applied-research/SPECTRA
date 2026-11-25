#!/usr/bin/env bash
# One-click SPECTRA demo (single GPU). Shows: (1) compress + generate + near-lossless perplexity,
# (2) real memory / max-batch efficiency vs fp16.
#
#   bash demo/demo.sh                       # default model NousResearch/Meta-Llama-3.1-8B-Instruct, preset 8.8x
#   MODEL=Qwen/Qwen2.5-7B-Instruct PRESET=4.6x bash demo/demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}"
PRESET="${PRESET:-8.8x}"
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"   # silence long-context tokenizer warnings
export TOKENIZERS_PARALLELISM=false
# Tip: set HF_HOME=/a/big/disk/hf if your default cache disk is small (~15-30 GB per model).

echo "=============================================================="
echo " SPECTRA demo | model=$MODEL | preset=$PRESET | GPU=$CUDA_VISIBLE_DEVICES"
echo "=============================================================="

echo; echo "### [1/2] Compress + generate + perplexity ###"
"$PYTHON" - "$MODEL" "$PRESET" <<'PY'
import sys, torch, spectra
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from spectra.core import make_windows, perplexity
model_id, preset = sys.argv[1], sys.argv[2]
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

# WikiText-2 test PPL, 40 windows of 2048 tokens.
tok = AutoTokenizer.from_pretrained(model_id)
te = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
eval_w = make_windows(tok, "\n\n".join(t for t in te["text"] if t.strip()), 2048, 40)

base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(dev).eval()
p_base = perplexity(base, eval_w, dev); print(f"[base ] WikiText-2 PPL = {p_base:.3f}")
del base; torch.cuda.empty_cache()

model, tok = spectra.compress(model_id, preset=preset, device=dev)
p_spec = perplexity(model, eval_w, dev)
print(f"[spectra {preset}] WikiText-2 PPL = {p_spec:.3f}  (delta {p_spec-p_base:+.3f})")
prompt = "In one sentence, what is a KV cache?"
out = model.generate(**tok(prompt, return_tensors="pt").to(dev), max_new_tokens=48)
print("[generate]", tok.decode(out[0], skip_special_tokens=True))
PY

echo; echo "### [2/2] Efficiency: peak GPU memory & max batch vs fp16 ###"
"$PYTHON" scripts/tools/bench_memory.py --model "$MODEL" --mode peakmem --contexts 4096,16384,32768
echo
echo "(optional) real OOM-searched max batch @32k:"
echo "  python scripts/tools/bench_memory.py --model $MODEL --mode maxbatch --e2-ctx 32768"

echo; echo "Done. See README.md to reproduce the paper's LongBench and NIAH numbers."
