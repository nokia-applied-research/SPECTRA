"""SPECTRA Mode-A decode-attention microbenchmark (PyTorch reference, standalone).

Measures, per decode step (single new query token attending to n cached tokens), one layer,
for Llama-3.1-8B GQA dims: (1) KV-cache VRAM, (2) decode-step latency, for fp16 vs SPECTRA
Mode-A (store quantized latent c_k/c_v of rank r; reconstruct K=c@W_UP, V=c@W_UP at read,
then standard GQA attention). SPECTRA's win is memory/bandwidth (smaller cache); the extra
reconstruction GEMM is the overhead — this bench finds where the net decode latency crosses.

Latent stored as int8 (proxy for <=8-bit packing; true mixed-bit bytes reported separately).
"""
import argparse, time, torch

MODELS = {                        # (NH attn heads, NKV kv heads, HD head_dim, HID hidden)
    "llama3.1-8b": dict(NH=32, NKV=8,  HD=128, HID=4096),
    "qwen2.5-14b": dict(NH=40, NKV=8,  HD=128, HID=5120),
}
NH, NKV, HD, HID = 32, 8, 128, 4096   # defaults (Llama-3.1-8B); overridden by --model in main()
KVDIM = NKV * HD


def sync(): torch.cuda.synchronize()


@torch.no_grad()
def bench_fp16(n, B, dev, iters=50):
    q = torch.randn(B, NH, 1, HD, device=dev, dtype=torch.float16)
    K = torch.randn(B, NKV, n, HD, device=dev, dtype=torch.float16)
    V = torch.randn(B, NKV, n, HD, device=dev, dtype=torch.float16)
    cache_bytes = (K.numel() + V.numel()) * 2
    def step():
        Kr = K.repeat_interleave(NH // NKV, dim=1); Vr = V.repeat_interleave(NH // NKV, dim=1)
        s = (q @ Kr.transpose(-1, -2)) / HD ** 0.5
        a = s.softmax(-1)
        return a @ Vr
    for _ in range(5): step()
    sync(); t = time.time()
    for _ in range(iters): step()
    sync(); return (time.time() - t) / iters * 1e3, cache_bytes


@torch.no_grad()
def bench_spectra_modeA(n, B, r, avg_bits, dev, iters=50):
    # cache: latent c_k, c_v of rank r (per token), stored int8 (proxy). reconstruct via W_UP.
    ck = torch.randint(-127, 127, (B, n, r), device=dev, dtype=torch.int8)
    cv = torch.randint(-127, 127, (B, n, r), device=dev, dtype=torch.int8)
    sk = torch.randn(B, n, 1, device=dev, dtype=torch.float16).abs() + 1e-2
    sv = torch.randn(B, n, 1, device=dev, dtype=torch.float16).abs() + 1e-2
    WUk = torch.randn(r, KVDIM, device=dev, dtype=torch.float16) / r ** 0.5
    WUv = torch.randn(r, KVDIM, device=dev, dtype=torch.float16) / r ** 0.5
    q = torch.randn(B, NH, 1, HD, device=dev, dtype=torch.float16)
    true_bytes = int(2 * B * n * r * avg_bits / 8)             # mixed-bit packed (reported)
    int8_bytes = (ck.numel() + cv.numel()) * 1 + (sk.numel() + sv.numel()) * 2
    def step():
        K = (ck.half() * sk) @ WUk                              # (B,n,KVDIM) reconstruct
        V = (cv.half() * sv) @ WUv
        K = K.view(B, n, NKV, HD).transpose(1, 2)               # (B,NKV,n,HD)
        V = V.view(B, n, NKV, HD).transpose(1, 2)
        Kr = K.repeat_interleave(NH // NKV, dim=1); Vr = V.repeat_interleave(NH // NKV, dim=1)
        s = (q @ Kr.transpose(-1, -2)) / HD ** 0.5
        return s.softmax(-1) @ Vr
    for _ in range(5): step()
    sync(); t = time.time()
    for _ in range(iters): step()
    sync(); return (time.time() - t) / iters * 1e3, true_bytes, int8_bytes


@torch.no_grad()
def bench_spectra_modeB(n, B, r, avg_bits, dev, iters=50):
    """Absorption / latent-space attention: read ONLY the latent (no K/V materialization).
    q' = q @ W_UPk^T (per kv-head); score = q'·c_k^T; a=softmax; agg=a·c_v; out=agg @ W_UPv."""
    ck = torch.randint(-127, 127, (B, n, r), device=dev, dtype=torch.int8)
    cv = torch.randint(-127, 127, (B, n, r), device=dev, dtype=torch.int8)
    sk = torch.randn(B, n, 1, device=dev, dtype=torch.float16).abs() + 1e-2
    sv = torch.randn(B, n, 1, device=dev, dtype=torch.float16).abs() + 1e-2
    WUk = (torch.randn(NKV, HD, r, device=dev, dtype=torch.float16) / r ** 0.5)   # per kv-head
    WUv = (torch.randn(NKV, r, HD, device=dev, dtype=torch.float16) / r ** 0.5)
    q = torch.randn(B, NKV, NH // NKV, HD, device=dev, dtype=torch.float16)         # grouped by kv-head
    true_bytes = int(2 * B * n * r * avg_bits / 8)
    grp = NH // NKV
    def step():
        Kf = (ck.half() * sk)                                   # (B,n,r) dequant
        Vf = (cv.half() * sv)
        # q' per kv-head: (B,NKV,grp,HD) @ (NKV,HD,r) -> (B,NKV,grp,r)
        qp = torch.einsum("bghd,gdr->bghr", q, WUk)
        s = torch.einsum("bghr,bnr->bghn", qp, Kf) / HD ** 0.5  # scores over n
        a = s.softmax(-1)
        agg = torch.einsum("bghn,bnr->bghr", a, Vf)             # (B,NKV,grp,r)
        out = torch.einsum("bghr,grd->bghd", agg, WUv)          # -> (B,NKV,grp,HD)
        return out
    for _ in range(5): step()
    sync(); t = time.time()
    for _ in range(iters): step()
    sync(); return (time.time() - t) / iters * 1e3, true_bytes


@torch.no_grad()
def bench_spectra_triton(n, B, r, avg_bits, dev, iters=50):
    """Full Mode-B with the Triton flash-decode latent kernel (q' and V up-proj outside)."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from spectra.triton_kernel import latent_decode
    grp = NH // NKV
    ck = torch.randint(-100, 100, (B, n, r), device=dev, dtype=torch.int8)
    cv = torch.randint(-100, 100, (B, n, r), device=dev, dtype=torch.int8)
    sk = torch.rand(B, n, device=dev, dtype=torch.float16) + 0.1
    sv = torch.rand(B, n, device=dev, dtype=torch.float16) + 0.1
    WUk = torch.randn(NKV, HD, r, device=dev, dtype=torch.float16) / r ** 0.5
    WUv = torch.randn(NKV, r, HD, device=dev, dtype=torch.float16) / r ** 0.5
    q = torch.randn(B, NKV, grp, HD, device=dev, dtype=torch.float16)
    def step():
        qp = torch.einsum("bghd,gdr->bghr", q, WUk)
        agg = latent_decode(qp, ck, sk, cv, sv, HD)
        return torch.einsum("bghr,grd->bghd", agg, WUv)
    for _ in range(5): step()
    sync(); t = time.time()
    for _ in range(iters): step()
    sync(); return (time.time() - t) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1-8b", choices=list(MODELS))
    ap.add_argument("--contexts", default="4096,16384,32768,65536")
    ap.add_argument("--batch", default="1,16")
    ap.add_argument("--r", type=int, default=512)        # low-rank latent (per K/V)
    ap.add_argument("--avg-bits", type=float, default=3.0)
    args = ap.parse_args()
    global NH, NKV, HD, HID, KVDIM
    d = MODELS[args.model]
    NH, NKV, HD, HID = d["NH"], d["NKV"], d["HD"], d["HID"]
    KVDIM = NKV * HD
    dev = "cuda:0"
    print(f"# SPECTRA decode microbench | model={args.model} r={args.r} avg_bits={args.avg_bits} | dims NH{NH} NKV{NKV} HD{HD} HID{HID}")
    print(f"| {'ctx':>6s} | {'B':>3s} | {'fp16 ms':>8s} | {'modeA ms':>8s} | {'A spd':>5s} | {'modeB ms':>8s} | {'B spd':>5s} | {'fp16 KV MB':>10s} | {'SPECTRA MB':>10s} | {'mem x':>5s} |")
    print("|" + "-"*8 + "|" + "-"*5 + "|" + ("-"*10 + "|")*2 + "-"*7 + "|" + "-"*10 + "|" + "-"*7 + "|" + ("-"*12 + "|")*2 + "-"*7 + "|")
    for n in [int(x) for x in args.contexts.split(",")]:
        for B in [int(x) for x in args.batch.split(",")]:
            try:
                t0, b0 = bench_fp16(n, B, dev)
                tA, tb, ib = bench_spectra_modeA(n, B, args.r, args.avg_bits, dev)
                tB, _ = bench_spectra_modeB(n, B, args.r, args.avg_bits, dev)
                try:
                    tT = bench_spectra_triton(n, B, args.r, args.avg_bits, dev)
                    tstr = f"{tT:8.3f} | {t0/tT:4.2f}x"
                except Exception as e:
                    tstr = f"{'err':>8s} | {str(e)[:6]}"
                print(f"| {n:6d} | {B:3d} | {t0:8.3f} | {tA:8.3f} {t0/tA:4.2f}x | {tB:8.3f} {t0/tB:4.2f}x | {tstr} | {b0/tb:4.1f}x mem |")
                torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"| {n:6d} | {B:3d} | OOM/err: {str(e)[:40]}")
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
