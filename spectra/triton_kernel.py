"""Triton flash-decode kernel for SPECTRA Mode-B (latent-space attention) — v2.

vs v1: (1) split-K over the token axis (parallel programs + combine pass) to saturate the GPU
at long context / small batch; (2) tensor-core `tl.dot` by processing all GRP query heads of a
kv-head together (GEMM, not GEMV); (3) in-kernel int8 dequant (no fp16 latent intermediate).

Latent c_k/c_v (int8, per-token scale sk/sv) are shared across the GRP heads of a kv-head (GQA),
read once. The per-kv-head query is pre-absorbed (q' = q @ W_UPk^T) and the V up-projection
(agg @ W_UPv) is applied outside. Targets H100 (Hopper); Triton lowers tl.dot to WGMMA.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _splitk(Qp, Ck, Sk, Cv, Sv, Mout, Lout, Aout,
            n, scale,
            s_qp_bh, s_ck_b, s_ck_n, s_sk_b,
            s_m_bh, s_m_s, s_a_bh, s_a_s,
            R: tl.constexpr, GRP: tl.constexpr, NKV: tl.constexpr,
            BLOCK_N: tl.constexpr, N_SPLIT: tl.constexpr):
    bh = tl.program_id(0)          # over B*NKV
    sp = tl.program_id(1)          # split index
    b = bh // NKV
    offs_r = tl.arange(0, R)
    offs_g = tl.arange(0, GRP)
    qp = tl.load(Qp + bh * s_qp_bh + offs_g[:, None] * R + offs_r[None, :]).to(tl.float16)  # (GRP,R)
    chunk = tl.cdiv(n, N_SPLIT)
    start = sp * chunk
    end = tl.minimum(start + chunk, n)
    ckb = Ck + b * s_ck_b; cvb = Cv + b * s_ck_b
    skb = Sk + b * s_sk_b; svb = Sv + b * s_sk_b
    m_i = tl.zeros([GRP], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([GRP], dtype=tl.float32)
    acc = tl.zeros([GRP, R], dtype=tl.float32)
    for s0 in range(start, end, BLOCK_N):
        offs_n = s0 + tl.arange(0, BLOCK_N)
        mask = offs_n < end
        ck = tl.load(ckb + offs_n[:, None] * s_ck_n + offs_r[None, :], mask=mask[:, None], other=0.0).to(tl.float16)
        sk = tl.load(skb + offs_n, mask=mask, other=0.0).to(tl.float16)
        kf = ck * sk[:, None]                                   # (BLOCK_N,R) fp16
        s = tl.dot(qp, tl.trans(kf)).to(tl.float32) * scale     # (GRP,BLOCK_N) tensor-core
        s = tl.where(mask[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        cv = tl.load(cvb + offs_n[:, None] * s_ck_n + offs_r[None, :], mask=mask[:, None], other=0.0).to(tl.float16)
        sv = tl.load(svb + offs_n, mask=mask, other=0.0).to(tl.float16)
        vf = cv * sv[:, None]
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), vf).to(tl.float32)   # (GRP,R)
        m_i = m_new
    tl.store(Mout + bh * s_m_bh + sp * s_m_s + offs_g, m_i)
    tl.store(Lout + bh * s_m_bh + sp * s_m_s + offs_g, l_i)
    tl.store(Aout + bh * s_a_bh + sp * s_a_s + offs_g[:, None] * R + offs_r[None, :], acc)


@triton.jit
def _combine(Mout, Lout, Aout, Out,
             s_m_bh, s_m_s, s_a_bh, s_a_s, s_o_bh,
             R: tl.constexpr, GRP: tl.constexpr, N_SPLIT: tl.constexpr):
    bh = tl.program_id(0)
    offs_g = tl.arange(0, GRP); offs_r = tl.arange(0, R)
    m = tl.zeros([GRP], dtype=tl.float32) - float("inf")
    for sp in range(N_SPLIT):
        m = tl.maximum(m, tl.load(Mout + bh * s_m_bh + sp * s_m_s + offs_g))
    l = tl.zeros([GRP], dtype=tl.float32)
    acc = tl.zeros([GRP, R], dtype=tl.float32)
    for sp in range(N_SPLIT):
        ms = tl.load(Mout + bh * s_m_bh + sp * s_m_s + offs_g)
        ls = tl.load(Lout + bh * s_m_bh + sp * s_m_s + offs_g)
        a = tl.load(Aout + bh * s_a_bh + sp * s_a_s + offs_g[:, None] * R + offs_r[None, :])
        w = tl.exp(ms - m)
        l += w * ls
        acc += w[:, None] * a
    out = acc / l[:, None]
    tl.store(Out + bh * s_o_bh + offs_g[:, None] * R + offs_r[None, :], out.to(tl.float16))


def latent_decode(qp, ck, sk, cv, sv, hd, n_split=8, block_n=64):
    """qp:(B,NKV,GRP,R) fp16; ck/cv:(B,n,R) int8; sk/sv:(B,n) fp16 -> agg (B,NKV,GRP,R)."""
    B, NKV, GRP, R = qp.shape
    n = ck.shape[1]; BH = B * NKV
    qpf = qp.reshape(BH, GRP, R).contiguous()
    dev = qp.device
    M = torch.empty(BH, n_split, GRP, device=dev, dtype=torch.float32)
    Lc = torch.empty(BH, n_split, GRP, device=dev, dtype=torch.float32)
    A = torch.empty(BH, n_split, GRP, R, device=dev, dtype=torch.float32)
    out = torch.empty(BH, GRP, R, device=dev, dtype=torch.float16)
    scale = 1.0 / (hd ** 0.5)
    _splitk[(BH, n_split)](
        qpf, ck, sk, cv, sv, M, Lc, A, n, scale,
        qpf.stride(0), ck.stride(0), ck.stride(1), sk.stride(0),
        M.stride(0), M.stride(1), A.stride(0), A.stride(1),
        R=R, GRP=GRP, NKV=NKV, BLOCK_N=block_n, N_SPLIT=n_split)
    _combine[(BH,)](M, Lc, A, out, M.stride(0), M.stride(1), A.stride(0), A.stride(1), out.stride(0),
                    R=R, GRP=GRP, N_SPLIT=n_split)
    return out.reshape(B, NKV, GRP, R)


@torch.no_grad()
def _reference(qp, ck, sk, cv, sv, hd):
    kf = ck.float() * sk[..., None]; vf = cv.float() * sv[..., None]
    s = torch.einsum("bghr,bnr->bghn", qp.float(), kf) / hd ** 0.5
    return torch.einsum("bghn,bnr->bghr", s.softmax(-1), vf)


if __name__ == "__main__":
    torch.manual_seed(0); dev = "cuda:0"
    B, NKV, GRP, R, n, HD = 2, 8, 4, 512, 1000, 128
    qp = torch.randn(B, NKV, GRP, R, device=dev, dtype=torch.float16)
    ck = torch.randint(-100, 100, (B, n, R), device=dev, dtype=torch.int8)
    cv = torch.randint(-100, 100, (B, n, R), device=dev, dtype=torch.int8)
    sk = torch.rand(B, n, device=dev, dtype=torch.float16) + 0.1
    sv = torch.rand(B, n, device=dev, dtype=torch.float16) + 0.1
    out = latent_decode(qp, ck, sk, cv, sv, HD)
    ref = _reference(qp, ck, sk, cv, sv, HD).half()
    print(f"v2 triton vs ref relerr = {(out-ref).norm()/ref.norm():.5f}  (should be < 1e-2)")
