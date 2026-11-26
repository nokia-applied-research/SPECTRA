"""`spectra-flash bench vllm-decode`: steady-state decode throughput in vLLM at a fixed batch and context, dense vs
SPECTRA-Flash.

For each batch B: B random-token prompts of --ctx tokens run twice (same prompts, no prefix caching) with greedy
outputs of L1 = B + 64 and L2 = L1 + --steps tokens (EOS ignored). Both runs share the same prefill phase (one prompt
per step) and the same finishing tail, so t(L2) - t(L1) is --steps decode steps with all B requests running:
    ms/step = (t2 - t1) / steps,   decode tok/s = B * steps / (t2 - t1).
Each variant runs in its own vLLM engine (a subprocess, one GPU) with the same settings as `bench serve`. A batch
larger than the engine's KV cache holds would queue; it is dropped from the table.
  spectra-flash bench vllm-decode --model Qwen/Qwen3.5-9B --params <dir|repo> --ctx 32000 --batches 1,48 \\
      --flash-batches 1,48,128,256 --gpus 0,1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from .bench_serve import CONCURRENCY, child_env, engine_kwargs

RESULT = "BENCH_DECODE_RESULT "


def child(a) -> int:
    import random

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    Bs = [int(x) for x in a.batches.split(",")]
    maxlen = a.ctx + max(Bs) + 64 + a.steps + 16
    llm = LLM(**engine_kwargs(a.model, a.variant, a.params, maxlen, a.gpu_mem, max(Bs)))
    rng = random.Random(a.seed)
    llm.generate([TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(2048)])],
                 SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False)                 # warm-up
    for B in Bs:
        prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(a.ctx)]) for _ in range(B)]
        L1 = B + 64
        t = []
        for L in (L1, L1 + a.steps):
            t0 = time.perf_counter()
            outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=L, ignore_eos=True), use_tqdm=False)
            t.append(time.perf_counter() - t0)
            assert all(len(o.outputs[0].token_ids) == L for o in outs)
        dt = t[1] - t[0]
        print(RESULT + json.dumps(dict(variant=a.variant, ctx=a.ctx, B=B, ms_step=1000 * dt / a.steps,
                                       tok_s=B * a.steps / dt, t1=t[0], t2=t[1])), flush=True)
    return 0


def run_variant(a, variant: str, batches: str, gpu: str, log_path: str) -> list[dict]:
    cmd = [sys.executable, "-m", "spectra.flash.bench_vllm_decode", "--child", "--variant", variant, "--model", a.model,
           "--params", a.params, "--ctx", str(a.ctx), "--batches", batches, "--steps", str(a.steps),
           "--gpu-mem", str(a.gpu_mem), "--seed", str(a.seed)]
    with open(log_path, "w") as f:
        p = subprocess.run(cmd, env=child_env(gpu), stdout=f, stderr=subprocess.STDOUT)
    text = open(log_path).read()
    rows = [json.loads(l[len(RESULT):]) for l in text.splitlines() if l.startswith(RESULT)]
    if p.returncode != 0 or not rows:
        raise RuntimeError(f"{variant} failed (exit {p.returncode}); log: {log_path}\n" + "\n".join(text.splitlines()[-20:]))
    m = CONCURRENCY.search(text)
    conc = float(m.group(3)) if m else float("inf")
    for r in rows:
        r["max_concurrency"] = conc
    return [r for r in rows if r["B"] <= conc]


def table(rows: list[dict], a, gpu_name: str) -> str:
    out = [f"vLLM 0.29 on {gpu_name}, gpu_memory_utilization {a.gpu_mem}, {a.model}: steady-state decode at {a.ctx} "
           f"context tokens, {a.steps} steps differenced.\n",
           "| variant | max concurrent requests | batch | ms / step | decode tok/s |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["variant"] != "dense", r["B"])):
        out.append(f"| {r['variant']} | {r['max_concurrency']:.1f} | {r['B']} | {r['ms_step']:.2f} | {r['tok_s']:.0f} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spectra-flash bench vllm-decode", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--params", default="", help="SPECTRA-Flash params dir or HF Hub repo id")
    ap.add_argument("--variants", default="dense,flash8")
    ap.add_argument("--ctx", type=int, default=32000)
    ap.add_argument("--batches", default="1,48", help="batch sizes (dense, and Flash unless --flash-batches)")
    ap.add_argument("--flash-batches", default="", help="batch sizes for Flash (its cache holds larger batches)")
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--gpus", default="0", help="one GPU per variant (comma-separated; variants run in parallel)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="write rows + table as JSON")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--variant", default="", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        return child(a)
    variants = a.variants.split(",")
    if "flash8" in variants and not a.params:
        ap.error("flash8 needs --params")
    gpus = (a.gpus.split(",") * len(variants))[:len(variants)]
    logdir = os.path.dirname(os.path.abspath(a.out)) if a.out else "."
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(variants)) as ex:
        futs = [ex.submit(run_variant, a, v, (a.flash_batches or a.batches) if v == "flash8" else a.batches, g,
                          os.path.join(logdir, f"bench_vllm_decode_{v}_{a.ctx}.log")) for v, g in zip(variants, gpus)]
        rows = [r for f in futs for r in f.result()]
    import torch
    md = table(rows, a, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU")
    print(md)
    if a.out:
        json.dump(dict(args=vars(a), rows=rows, table=md), open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
