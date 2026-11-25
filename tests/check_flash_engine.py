"""SPECTRA-Flash decode engine checks (Hopper GPU, vLLM 0.29 installed for the kernels):
  1. needle gate: 32K passkey through prefill + graph-replayed decode, dense and flash8;
  2. greedy parity: engine flash8 vs the 🤗 Transformers fast path on one LongBench prompt (same params);
  3. a small decode-step benchmark (32K, H100 NVL: dense B1 7.24 / B16 24.89 ms, flash8 B1 7.24 / B16 10.56 ms).
usage: python tests/check_flash_engine.py --params <params dir or HF Hub repo> [--model meta-llama/Llama-3.1-8B-Instruct]
"""
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
ap.add_argument("--params", required=True)
ap.add_argument("--task", default="qasper", help="LongBench task of the parity prompt")
ap.add_argument("--prompt", type=int, default=0, help="index of the parity prompt in the task")
ap.add_argument("--tg", type=int, default=64)
ap.add_argument("--B", default="1,16")
a = ap.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer

import spectra.flash as BFl
from spectra.flash import engine as EN

dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(a.model)
from spectra.eval.longbench import encode, load_task
ids_p = encode(tok, a.task, load_task(a.task)[a.prompt], 31500).long().to(dev)
with torch.inference_mode():
    M = EN.Model(a.model, a.params, dev)
    print(f"[engine] loaded: sink {M.sink} recent {M.recent}, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    ok = [EN.needle_check(M, tok, v) for v in ("dense", "flash8")]
    E = EN.Engine(M, EN.Cache(M, "flash8", 1, ids_p.numel() + a.tg + 64))
    toks_e = E.generate(ids_p, a.tg)
    del E
    res = EN.bench(M, ["dense", "flash8"], [int(x) for x in a.B.split(",")], [32768], steps=50)
    del M
    torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
BFl.enable(model, a.params, backend="fast")
with torch.inference_mode():
    toks_h = model.generate(ids_p[None], max_new_tokens=a.tg, do_sample=False)[0, ids_p.numel():].tolist()
m = sum(x == y for x, y in zip(toks_e, toks_h))
fd = next((i for i, (x, y) in enumerate(zip(toks_e, toks_h)) if x != y), None)
print(f"[engine] greedy parity engine flash8 vs HF fast: {m}/{a.tg}, first divergence {fd}", flush=True)
print(f"[engine] needle dense {ok[0]} flash8 {ok[1]}", flush=True)
print("ENGINE_CHECK_DONE", flush=True)
