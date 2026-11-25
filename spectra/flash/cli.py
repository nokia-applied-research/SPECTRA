"""`spectra-flash` command line.

  spectra-flash convert ...   convert a base model (python -m spectra.flash.convert)
  spectra-flash bench decode  end-to-end decode speed: needle gate, then step time over context x batch x variant
  spectra-flash bench serve   vLLM serving throughput, dense vs Flash at one GPU memory budget
LongBench: python scripts/run_longbench.py --flash <params> (see README).
"""
from __future__ import annotations

import argparse
import json
import sys

VARIANTS = ["dense", "dense8", "flash16", "flash8"]


def table(rows, gpu: str) -> str:
    """Markdown: ms/step per variant, flash speedups at equal batch, best throughput within memory."""
    out = [f"GPU: {gpu}. Decode step time (ms): one CUDA graph per step (all layers, lm_head, argmax), median of the "
           "timed replays; '-' = the cache does not fit.\n"]
    for T in sorted({r["T"] for r in rows}):
        R = {(r["B"], r["variant"]): r for r in rows if r["T"] == T}
        V = [v for v in VARIANTS if any(vv == v for _, vv in R)]
        out += [f"### context {T // 1024}K\n", "| B | " + " | ".join(V) + " | flash8 vs dense | flash8 vs dense8 |",
                "|---|" + "---|" * (len(V) + 2)]
        for B in sorted({b for b, _ in R}):
            ms = {v: R[(B, v)].get("ms") for v in V if (B, v) in R}
            cell = lambda v: f"{ms[v]:.2f}" if ms.get(v) else "-"
            sp = lambda a, b: f"{ms[b] / ms[a]:.2f}x" if ms.get(a) and ms.get(b) else "-"
            out.append(f"| {B} | " + " | ".join(cell(v) for v in V) + f" | {sp('flash8', 'dense')} | {sp('flash8', 'dense8')} |")
        best = {}
        for v in V:
            ok = [r for (b, vv), r in R.items() if vv == v and r.get("ms")]
            if ok:
                best[v] = max(ok, key=lambda r: r["tok_s"])
        out += ["\nBest throughput within memory (the largest batch that fit):\n",
                "| variant | batch | ms/step | tok/s | cache GiB | vs dense |", "|---|---|---|---|---|---|"]
        for v, r in best.items():
            rel = f"{r['tok_s'] / best['dense']['tok_s']:.2f}x" if "dense" in best else "-"
            out.append(f"| {v} | {r['B']} | {r['ms']:.2f} | {r['tok_s']:.0f} | {r['cache_gib']:.1f} | {rel} |")
        out.append("")
    return "\n".join(out)


def bench_main(argv):
    ap = argparse.ArgumentParser(prog="spectra-flash bench decode", description="End-to-end decode speed (vLLM kernels, Hopper)")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--params", required=True, help="SPECTRA-Flash params dir or HF Hub repo id")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--B", default="1,4,8,16,32,64,128")
    ap.add_argument("--T", default="32768,131072")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--needle", type=int, default=32768, help="passkey length for the correctness gate (0 = skip)")
    ap.add_argument("--out", default="", help="write rows + table as JSON")
    a = ap.parse_args(argv)
    import torch
    from transformers import AutoTokenizer
    from . import engine as EN
    torch.backends.cuda.matmul.allow_tf32 = False
    variants = a.variants.split(",")
    with torch.inference_mode():
        M = EN.Model(a.model, a.params, "cuda")
        if a.needle:
            tok = AutoTokenizer.from_pretrained(a.model)
            failed = [v for v in variants if not EN.needle_check(M, tok, v, T=a.needle)]
            if failed:
                print(f"[bench] needle gate failed for {failed}: not timing a decode path that reads the wrong context")
                return 1
        rows = EN.bench(M, variants, [int(x) for x in a.B.split(",")], [int(x) for x in a.T.split(",")], a.steps)
    gpu = torch.cuda.get_device_name(0)
    md = table(rows, gpu)
    print(md)
    if a.out:
        json.dump(dict(args=vars(a), gpu=gpu, rows=rows, table=md), open(a.out, "w"), indent=1)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "bench":
        if rest[:1] == ["decode"]:
            return bench_main(rest[1:])
        if rest[:1] == ["serve"]:
            from .bench_serve import main as serve_main
            return serve_main(rest[1:])
        print("usage: spectra-flash bench decode|serve [--help]   (end-to-end decode step time | vLLM serving "
              "throughput at one memory budget)")
        return 2
    if cmd == "convert":
        from .convert import main as m
    else:
        print(f"unknown command {cmd!r}\n{__doc__}")
        return 2
    return m(rest) or 0


if __name__ == "__main__":
    sys.exit(main())
