"""Fused Triton kernels of the SPECTRA-Flash decode path.

  q_prep     : W_UK absorption (per kv group) + RoPE of the rope query + fp8 quantization with a static per-layer
               scale, written straight into the FlashMLA query layout (B, H, 576).
  tail_merge : exact attention over the window (sink + recent, looped in chunks of <= 128 keys) + W_UV up-projection of
               the FlashMLA output + LSE merge of the two partial softmaxes.
  rec_write  : record entry of the current token: c = e0 + e1 (the block's two layers' partial H W_d, produced by their
               qkv GEMMs), RoPE of the s part, fp8 quantization, store at the record slot.
Heads within a kv group (G = 4) are padded to 16 rows for tl.dot. Tests: tests/test_flash_kernels.py.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _q_prep_kernel(QKV, s_qkv, WUK, POS, THETA, SS, QS, OUT,
                   H: tl.constexpr, G: tl.constexpr, D: tl.constexpr, R: tl.constexpr, RR: tl.constexpr,
                   QR0: tl.constexpr, BR: tl.constexpr, F8: tl.constexpr):
    b = tl.program_id(0); g = tl.program_id(1); t = tl.program_id(2)
    i = tl.arange(0, 16)
    hm = i < G
    h = g * G + i
    row = OUT + (b * H + h) * (R + RR)
    if F8:
        inv = 1.0 / tl.load(QS)
    else:
        inv = 1.0
    if t < R // BR:
        d = tl.arange(0, D)
        q = tl.load(QKV + b * s_qkv + h[:, None] * D + d[None, :], mask=hm[:, None], other=0.0)
        j = t * BR + tl.arange(0, BR)
        w = tl.load(WUK + g * D * R + d[:, None] * R + j[None, :])
        val = tl.dot(q, w) * inv
        if F8:
            val = tl.minimum(tl.maximum(val, -448.0), 448.0)
        tl.store(row[:, None] + j[None, :], val.to(OUT.dtype.element_ty), mask=hm[:, None])
    else:
        p = tl.arange(0, RR // 2)
        base = QKV + b * s_qkv + QR0 + h[:, None] * RR + 2 * p[None, :]
        x0 = tl.load(base, mask=hm[:, None], other=0.0).to(tl.float32)
        x1 = tl.load(base + 1, mask=hm[:, None], other=0.0).to(tl.float32)
        ang = tl.load(POS + b).to(tl.float32) * tl.load(THETA + p)
        c = libdevice.cos(ang)[None, :]; s = libdevice.sin(ang)[None, :]
        y0 = x0 * c - x1 * s
        y1 = x1 * c + x0 * s
        if F8:
            y0 = y0 * tl.load(SS + 2 * p)[None, :] * inv
            y1 = y1 * tl.load(SS + 2 * p + 1)[None, :] * inv
            y0 = tl.minimum(tl.maximum(y0, -448.0), 448.0)
            y1 = tl.minimum(tl.maximum(y1, -448.0), 448.0)
        tl.store(row[:, None] + R + 2 * p[None, :], y0.to(OUT.dtype.element_ty), mask=hm[:, None])
        tl.store(row[:, None] + R + 2 * p[None, :] + 1, y1.to(OUT.dtype.element_ty), mask=hm[:, None])


@triton.jit
def _tail_merge_kernel(QF, s_qf, WK, WV, s_slot, s_head, WBASE, POS, RLEN, OL, LSER, WUV, OUT, sm_scale,
                       H: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
                       SINK: tl.constexpr, RING: tl.constexpr, TCP: tl.constexpr, BR: tl.constexpr, DT: tl.constexpr):
    # grid (B, HKV, D / DT): each program owns DT output dims; the (cheap) window scores are recomputed per program.
    # Sequence b's exact window is the ring at slots WBASE[b] + [0, SINK + RING) of a (slots, HKV, D) K / V layout
    # (slot stride s_slot, head stride s_head): position q at slot q (q < SINK) or SINK + q % RING. Slot j is valid once
    # the first position it holds is <= the current position POS[b]; in steady state (POS >= SINK + RING - 1) all are.
    # RLEN[b] == 0: no record entries yet, the row's FlashMLA output is ignored (whatever it holds).
    b = tl.program_id(0); g = tl.program_id(1); dt = tl.program_id(2)
    i = tl.arange(0, 16)
    hm = i < G
    h = g * G + i
    d = tl.arange(0, D)
    dd = dt * DT + tl.arange(0, DT)
    q = tl.load(QF + b * s_qf + h[:, None] * D + d[None, :], mask=hm[:, None], other=0.0)
    pos = tl.load(POS + b)
    base = tl.load(WBASE + b)
    j = tl.arange(0, TCP)
    q0 = j - SINK
    q0 = tl.where(q0 < SINK, q0 + RING, q0)                     # first position held by ring slot j
    km = (j < SINK + RING) & tl.where(j < SINK, j <= pos, q0 <= pos)
    k = tl.load(WK + (base + j[:, None]) * s_slot + g * s_head + d[None, :], mask=km[:, None], other=0.0)
    s = tl.dot(q, tl.trans(k)) * sm_scale
    s = tl.where(km[None, :], s, float("-inf"))
    m = tl.max(s, 1)
    p = tl.exp(s - m[:, None])
    l = tl.sum(p, 1)
    v = tl.load(WV + (base + j[:, None]) * s_slot + g * s_head + dd[None, :], mask=km[:, None], other=0.0)
    o_e = tl.dot(p.to(tl.bfloat16), v) / l[:, None]
    lse_e = m + tl.log(l)
    acc = tl.zeros((16, DT), dtype=tl.float32)
    for rc in range(0, R, BR):
        r_ = rc + tl.arange(0, BR)
        ol = tl.load(OL + (b * H + h[:, None]) * R + r_[None, :], mask=hm[:, None], other=0.0)
        w = tl.load(WUV + g * R * D + r_[:, None] * D + dd[None, :])
        acc += tl.dot(ol, w)
    lr = tl.load(LSER + b * H + h, mask=hm, other=0.0)
    wr = 1.0 / (1.0 + tl.exp(lse_e - lr))                     # record share of the softmax mass
    has = tl.load(RLEN + b) > 0
    out = tl.where(has, acc * wr[:, None] + o_e * (1.0 - wr[:, None]), o_e)
    tl.store(OUT + (b * H + h[:, None]) * D + dd[None, :], out.to(tl.bfloat16), mask=hm[:, None])


@triton.jit
def _rec_write_kernel(E0, E1, s_e0, s_e1, POS, THETA, REC, SLOT, R: tl.constexpr, RR: tl.constexpr,
                      F8: tl.constexpr, TWO: tl.constexpr):
    b = tl.program_id(0)
    slot = tl.load(SLOT + b)
    ok = slot >= 0                                 # -1 = no entry (position < sink, padding)
    dst = REC + tl.maximum(slot, 0) * (R + RR)
    j = tl.arange(0, R)
    p = tl.arange(0, RR // 2)
    c = tl.load(E0 + b * s_e0 + j).to(tl.float32)
    x0 = tl.load(E0 + b * s_e0 + R + 2 * p).to(tl.float32)
    x1 = tl.load(E0 + b * s_e0 + R + 2 * p + 1).to(tl.float32)
    if TWO:                                        # n = 2: the block's two layers each give a partial entry
        c += tl.load(E1 + b * s_e1 + j).to(tl.float32)
        x0 += tl.load(E1 + b * s_e1 + R + 2 * p).to(tl.float32)
        x1 += tl.load(E1 + b * s_e1 + R + 2 * p + 1).to(tl.float32)
    ang = tl.load(POS + b).to(tl.float32) * tl.load(THETA + p)
    cs = libdevice.cos(ang); sn = libdevice.sin(ang)
    y0 = x0 * cs - x1 * sn
    y1 = x1 * cs + x0 * sn
    if F8:
        c = tl.minimum(tl.maximum(c, -448.0), 448.0)
        y0 = tl.minimum(tl.maximum(y0, -448.0), 448.0)
        y1 = tl.minimum(tl.maximum(y1, -448.0), 448.0)
    tl.store(dst + j, c.to(REC.dtype.element_ty), mask=ok & (j < R))
    tl.store(dst + R + 2 * p, y0.to(REC.dtype.element_ty), mask=ok & (p < RR // 2))
    tl.store(dst + R + 2 * p + 1, y1.to(REC.dtype.element_ty), mask=ok & (p < RR // 2))


# ----------------------------------------------------------------------------- launchers
def q_prep(qkvr, wuk, pos, theta, ss, qs, out, f8, H=32, G=4, D=128, QR0=6144, BR=None):
    """qkvr (B, >= QR0 + H r) bf16 rows (q un-roped at [0, H D)); wuk (HKV, D, R); out (B, H, R + r)."""
    B = qkvr.shape[0]; HKV, _, R = wuk.shape; RR = theta.numel() * 2
    BR = BR or (64 if B < 8 else 128)
    _q_prep_kernel[(B, HKV, R // BR + 1)](qkvr, qkvr.stride(0), wuk, pos, theta, ss, qs, out,
                                          H=H, G=G, D=D, R=R, RR=RR, QR0=QR0, BR=BR, F8=f8, num_warps=4)
    return out


def tail_merge(qf, s_qf, wk, wv, wbase, pos, rlen, o_l, lse_r, wuv, out, sm_scale, sink, ring, H=32, G=4, D=128,
               DT=None):
    """Exact window attention merged with the record attention by LSE, + the value up-projection.
    qf: roped q rows (B, H D) with row stride s_qf. wk / wv: (slots, HKV, D) views (any slot / head strides, the same
    for both): sequence b's ring at slots wbase[b] + [0, sink + ring) (wbase (B,) int64). pos (B,) int64 current
    positions. rlen (B,) int32 record entries (0: o_l / lse_r of that row are ignored). o_l (B, 1, H, R), lse_r (B, H, 1)
    from FlashMLA; wuv (HKV, R, D); out (B, H D) bf16."""
    B = out.shape[0]; HKV, R, _ = wuv.shape
    assert wk.stride() == wv.stride() and wk.stride(-1) == 1
    DT = DT or (32 if B < 8 else D)
    TCP = triton.next_power_of_2(sink + ring)
    _tail_merge_kernel[(B, HKV, D // DT)](qf, s_qf, wk, wv, wk.stride(0), wk.stride(1), wbase, pos, rlen, o_l, lse_r,
                                          wuv, out, sm_scale, H=H, HKV=HKV, G=G, D=D, R=R, SINK=sink, RING=ring,
                                          TCP=TCP, BR=128, DT=DT, num_warps=4)
    return out


def rec_write(e0, e1, pos, theta, rec, slot, f8, R=512):
    """e0/e1 (B, R + r) row views (any row strides), summed; e1 = None for blocks of one layer (e0 is the whole entry).
    rec (nslots, R + r); slot (B,) int64, -1 = skip the row."""
    two = e1 is not None
    e1 = e1 if two else e0
    assert e0.stride(-1) == 1 and e1.stride(-1) == 1
    _rec_write_kernel[(e0.shape[0],)](e0, e1, e0.stride(0), e1.stride(0), pos, theta, rec, slot, R=R,
                                      RR=theta.numel() * 2, F8=f8, TWO=two, num_warps=4)
