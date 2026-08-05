"""E1/E2 (analytic) — KV-cache memory & memory-enabled max batch for SPECTRA.

KV-cache size is exact arithmetic (dtype x dims x context x batch + our scales), so the KV columns
here are ground truth, not an approximation. The only estimated quantity is the whole-GPU max batch
(activations / framework overhead / fragmentation are lumped into `--overhead-gb`); validate that
against a real OOM search on H100 with `bench_memory.py`.

Usage:
  python scripts/tools/memory_roofline.py --model qwen2.5-14b \
      --contexts 4096,8192,16384,32768,65536,131072 --gpu-mem 94 --e2-ctx 32768
"""
import argparse

FP16 = 2  # bytes

# name -> dict(L, kv_dim, weights_gb). kv_dim = kv_heads * head_dim (post-GQA).
MODELS = {
    "llama3.1-8b": dict(L=32, kv_dim=1024, weights_gb=16),
    "qwen2.5-14b": dict(L=48, kv_dim=1024, weights_gb=30),
    "llama2-13b":  dict(L=40, kv_dim=5120, weights_gb=26),   # MHA reference
}

# (label, compression x). Nominal recipe targets (measured total_x is within ~2% on 7-8B runs).
POINTS = [
    ("fp16 (base)",       1.00),
    ("KIVI b4",           3.20),
    ("SPECTRA-WF b4",     3.56),
    ("SPECTRA-WF b3",     4.57),
    ("KIVI b2",           5.33),
    ("SPECTRA-LR (~9x)",  9.14),
    ("SPECTRA-LR (~12x)", 12.0),
]

def kv_bytes_per_token_fp16(m):
    return 2 * m["kv_dim"] * FP16 * m["L"]     # K + V, all layers

def gib(nbytes):
    return nbytes / (1024 ** 3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-14b", choices=list(MODELS))
    ap.add_argument("--contexts", default="4096,8192,16384,32768,65536,131072")
    ap.add_argument("--gpu-mem", type=float, default=94.0, help="GB")
    ap.add_argument("--overhead-gb", type=float, default=3.0, help="activations/framework/frag headroom")
    ap.add_argument("--e2-ctx", type=int, default=32768)
    args = ap.parse_args()

    m = MODELS[args.model]
    ctxs = [int(x) for x in args.contexts.split(",")]
    fp16_bt = kv_bytes_per_token_fp16(m)

    print(f"# {args.model}: L={m['L']} kv_dim={m['kv_dim']} weights~{m['weights_gb']}GB | "
          f"fp16 KV = {fp16_bt/1024:.0f} KB/token | GPU {args.gpu_mem}GB\n")

    # ---- Table E1: KV GB/seq (batch=1) vs context ----
    print("## E1 — KV-cache memory (GB, batch=1)\n")
    print("| Method | Comp. | " + " | ".join(f"{c//1024}k" for c in ctxs) + " |")
    print("|" + "---|" * (len(ctxs) + 2))
    for name, x in POINTS:
        cells = " | ".join(f"{gib(fp16_bt / x * c):.2f}" for c in ctxs)
        print(f"| {name} | {x:.2f}x | {cells} |")

    # ---- Table E2: memory-enabled max batch @ e2-ctx ----
    budget = args.gpu_mem - m["weights_gb"] - args.overhead_gb
    print(f"\n## E2 — memory-enabled max batch @ {args.e2_ctx//1024}k "
          f"(budget = {args.gpu_mem} - {m['weights_gb']} - {args.overhead_gb} = {budget:.1f} GB)\n")
    print("| Method | Comp. | KV GB/seq | Max batch |")
    print("|---|---|---|---|")
    for name, x in POINTS:
        gb_seq = gib(fp16_bt / x * args.e2_ctx)
        mb = int(budget // gb_seq)
        print(f"| {name} | {x:.2f}x | {gb_seq:.2f} | {mb} |")

    print("\n_KV columns are exact; Max batch is analytic (overhead lumped) — validate with bench_memory.py._")

if __name__ == "__main__":
    main()
