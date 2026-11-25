"""`spectra-flash bench serve`: vLLM serving throughput, dense vs SPECTRA-Flash, at one GPU memory budget.

Each variant runs in its own vLLM engine (a subprocess, one GPU) with the same --gpu-memory-utilization, so both get
the same KV-cache memory; Flash fits more requests into it. All requests (random token prompts of --input-len, greedy
--output-len tokens, EOS ignored) are submitted at once and vLLM's scheduler batches them as memory allows: the
numbers include scheduling, prefill and decode. Reported: KV capacity (tokens, and requests of this length), wall time,
requests/s and output / total tokens/s.
Both variants: bfloat16, no prefix caching, the whole prompt in one prefill step (Flash v0.2 needs it; dense uses the
same scheduler settings).
  spectra-flash bench serve --params <dir|repo> [--input-len 32768 --output-len 256 --num-prompts 256 --gpus 0,1]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

RESULT = "BENCH_SERVE_RESULT "


def child(a) -> int:
    import random

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    maxlen = a.input_len + a.output_len + 16
    kw = dict(model=a.model, dtype="bfloat16", enable_prefix_caching=False, enable_chunked_prefill=False,
              max_model_len=maxlen, max_num_batched_tokens=maxlen, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=a.max_num_seqs, seed=0)
    if a.variant == "flash8":
        kw["hf_overrides"] = {"architectures": ["SpectraFlashLlamaForCausalLM"], "spectra_flash": a.params}
    llm = LLM(**kw)
    rng = random.Random(a.seed)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(a.input_len)])
               for _ in range(a.num_prompts)]
    llm.generate(prompts[:2], SamplingParams(temperature=0.0, max_tokens=8), use_tqdm=False)        # warm-up
    sp = SamplingParams(temperature=0.0, max_tokens=a.output_len, ignore_eos=True)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    out_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(RESULT + json.dumps(dict(variant=a.variant, wall_s=dt, req_s=len(outs) / dt, out_tok_s=out_tok / dt,
                                   total_tok_s=(len(outs) * a.input_len + out_tok) / dt, requests=len(outs),
                                   output_tokens=out_tok)), flush=True)
    return 0


def run_variant(a, variant: str, gpu: str, log_path: str) -> dict:
    cmd = [sys.executable, "-m", "spectra.flash.bench_serve", "--child", "--variant", variant, "--model", a.model,
           "--params", a.params, "--input-len", str(a.input_len), "--output-len", str(a.output_len),
           "--num-prompts", str(a.num_prompts), "--gpu-mem", str(a.gpu_mem), "--max-num-seqs", str(a.max_num_seqs),
           "--seed", str(a.seed)]
    with open(log_path, "w") as f:
        p = subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, stdout=f, stderr=subprocess.STDOUT)
    text = open(log_path).read()
    res = next((json.loads(l[len(RESULT):]) for l in text.splitlines() if l.startswith(RESULT)), None)
    if p.returncode != 0 or res is None:
        raise RuntimeError(f"{variant} failed (exit {p.returncode}); log: {log_path}\n" + "\n".join(text.splitlines()[-20:]))
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x",
                  text)
    if m:
        res.update(kv_tokens=int(m.group(1).replace(",", "")), max_concurrency=float(m.group(3)))
    return res


def table(rows: list[dict], a, gpu_name: str) -> str:
    out = [f"vLLM 0.29 on {gpu_name}, gpu_memory_utilization {a.gpu_mem}: {a.num_prompts} requests of {a.input_len} "
           f"random prompt tokens + {a.output_len} greedy tokens, all submitted at once.\n",
           "| variant | KV cache (tokens) | max concurrent requests | wall (s) | requests/s | output tok/s | total tok/s "
           "| vs dense |", "|---|---|---|---|---|---|---|---|"]
    base = next((r for r in rows if r["variant"] == "dense"), None)
    for r in rows:
        rel = f"{r['out_tok_s'] / base['out_tok_s']:.2f}x" if base else "-"
        out.append(f"| {r['variant']} | {r.get('kv_tokens', 0):,} | {r.get('max_concurrency', 0):.1f} | "
                   f"{r['wall_s']:.1f} | {r['req_s']:.2f} | {r['out_tok_s']:.0f} | {r['total_tok_s']:.0f} | {rel} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spectra-flash bench serve", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--params", default="", help="SPECTRA-Flash params dir or HF Hub repo id")
    ap.add_argument("--variants", default="dense,flash8")
    ap.add_argument("--input-len", type=int, default=32768)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--num-prompts", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--max-num-seqs", type=int, default=256)
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
    gpus = a.gpus.split(",")
    gpus = (gpus * len(variants))[:len(variants)]
    logdir = os.path.dirname(os.path.abspath(a.out)) if a.out else "."
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(variants)) as ex:
        futs = [ex.submit(run_variant, a, v, g, os.path.join(logdir, f"bench_serve_{v}_{a.input_len}_{a.output_len}.log"))
                for v, g in zip(variants, gpus)]
        rows = [f.result() for f in futs]
    import torch
    md = table(rows, a, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU")
    print(md)
    if a.out:
        json.dump(dict(args=vars(a), rows=rows, table=md), open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
