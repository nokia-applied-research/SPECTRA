"""E1/E2 (measured, H100) — real peak GPU memory & OOM-searched max batch.

Two modes:
  peakmem : forward the real model at each context (batch=1), report measured peak GPU memory and
            the actual KV-cache bytes (validates the analytic E1 fp16 row).
  maxbatch: keep model weights resident, allocate KV-cache tensors for a (batch, ctx) request plus a
            one-step decode activation, and binary-search the largest batch that fits (validates E2).
            Compressed points are emulated by shrinking the allocated KV bytes by the compression ratio
            (KV storage is deterministic), so the same ceiling applies without a compressed serving loop.

Run on the target GPU, e.g.:
  python scripts/tools/bench_memory.py --model Qwen/Qwen2.5-14B-Instruct --mode peakmem \
      --contexts 4096,8192,16384,32768,65536
  python scripts/tools/bench_memory.py --model Qwen/Qwen2.5-14B-Instruct --mode maxbatch \
      --e2-ctx 32768 --compressions 1.0,3.2,3.56,4.57,5.33,9.14,12.0
"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoConfig

GiB = 1024 ** 3

def _kv_bytes(pkv):
    """Sum KV-cache tensor bytes across transformers Cache API variants (5.x DynamicCache, legacy)."""
    tensors = []
    try:
        if hasattr(pkv, "layers"):                       # transformers 5.x
            for lyr in pkv.layers:
                tensors += [getattr(lyr, "keys", None), getattr(lyr, "values", None)]
        elif hasattr(pkv, "key_cache"):                  # older DynamicCache
            tensors += list(pkv.key_cache) + list(pkv.value_cache)
        else:                                            # legacy tuple-of-tuples
            for layer in pkv:
                tensors += list(layer)
    except Exception:
        return 0
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)

def kv_dims(cfg):
    L = cfg.num_hidden_layers
    kvh = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    return L, kvh, hd

def peakmem(args):
    dev = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(dev).eval()
    cfg = model.config
    L, kvh, hd = kv_dims(cfg)
    torch.cuda.synchronize()
    w = torch.cuda.memory_allocated() / GiB
    print(f"# {args.model}: L={L} kv_heads={kvh} head_dim={hd} | weights={w:.2f} GB (measured)\n")
    print("| ctx | peak GPU (GB) | KV cache (GB, measured) | KV (GB, analytic) |")
    print("|---|---|---|---|")
    for ctx in [int(x) for x in args.contexts.split(",")]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(0, cfg.vocab_size, (1, ctx), device=dev)
        try:
            with torch.no_grad():
                out = model(ids, use_cache=True)
            peak = torch.cuda.max_memory_allocated() / GiB
            kv_bytes = _kv_bytes(out.past_key_values)
            analytic = 2 * kvh * hd * 2 * L * ctx / GiB
            kv_str = f"{kv_bytes/GiB:.2f}" if kv_bytes else "-"
            print(f"| {ctx//1024}k | {peak:.2f} | {kv_str} | {analytic:.2f} |")
            del out
        except torch.cuda.OutOfMemoryError:
            print(f"| {ctx//1024}k | OOM | - | - |")
        torch.cuda.empty_cache()

def _fits(model, cfg, L, kvh, hd, ctx, batch, dev):
    """Allocate KV for (batch, ctx) + a 1-token decode activation; return True if it fits."""
    torch.cuda.empty_cache()
    kv = []
    try:
        for _ in range(L):
            k = torch.empty(batch, kvh, ctx, hd, device=dev, dtype=torch.float16)
            v = torch.empty(batch, kvh, ctx, hd, device=dev, dtype=torch.float16)
            kv.append((k, v))
        ids = torch.randint(0, cfg.vocab_size, (batch, 1), device=dev)
        with torch.no_grad():
            model(ids, use_cache=False)      # one-step activation footprint
        torch.cuda.synchronize()
        ok = True
    except torch.cuda.OutOfMemoryError:
        ok = False
    del kv
    torch.cuda.empty_cache()
    return ok

def maxbatch(args):
    dev = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(dev).eval()
    cfg = model.config
    L, kvh, hd = kv_dims(cfg)
    ctx = args.e2_ctx
    print(f"# {args.model} max batch @ {ctx//1024}k (real OOM search)\n")
    print("| Comp. | KV GB/seq | Max batch (measured) |")
    print("|---|---|---|")
    for comp in [float(x) for x in args.compressions.split(",")]:
        eff_ctx = int(ctx / comp)                 # fewer KV bytes == shorter effective ctx
        gb_seq = 2 * kvh * hd * 2 * L * eff_ctx / GiB
        # exponential grow then binary search
        lo, hi = 1, 1
        while _fits(model, cfg, L, kvh, hd, eff_ctx, hi, dev):
            lo = hi; hi *= 2
            if hi > 4096: break
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if _fits(model, cfg, L, kvh, hd, eff_ctx, mid, dev): lo = mid
            else: hi = mid
        print(f"| {comp:.2f}x | {gb_seq:.2f} | {lo} |")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--mode", choices=["peakmem", "maxbatch"], default="peakmem")
    ap.add_argument("--contexts", default="4096,8192,16384,32768,65536")
    ap.add_argument("--e2-ctx", type=int, default=32768)
    ap.add_argument("--compressions", default="1.0,3.2,3.56,4.57,5.33,9.14,12.0")
    args = ap.parse_args()
    (peakmem if args.mode == "peakmem" else maxbatch)(args)

if __name__ == "__main__":
    main()
