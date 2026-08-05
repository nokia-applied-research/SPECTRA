"""Quantization primitives for SPECTRA, including the ROADMAP rate-distortion bit allocator.

The current shipped method (`spectra.method`) uses a heavy/light split with uniform bits in the
light block. The 3-5x recipe replaces uniform bits with
**per-latent-channel rate-distortion (water-filling) allocation**: because the G-weighted SVD
orders latent channels by variance, allocating bits proportional to 0.5*log2(variance) is
RD-optimal and provably beats uniform-bit scalar quant (KIVI) and random-rotation quant
(TurboQuant) at matched average rate. This module provides that allocator + a per-channel
fake-quant so it can be dropped into LatentProj.
"""
from __future__ import annotations
import math
import torch


def water_filling_bits(variances, avg_bits, bmin=0, bmax=8):
    """Reverse water-filling integer bit allocation across channels.

    For independent Gaussian sources, the rate-distortion-optimal real-valued allocation at a
    fixed average rate R is  b_i = R + 0.5 * log2(var_i / GM(var)),  where GM is the geometric
    mean. We then clamp to [bmin, bmax], round to integers, and greedily fix up the rounding so
    the realized average equals `avg_bits` as closely as possible.

    Args:
        variances: 1-D tensor of per-channel variances (e.g. SVD singular values squared).
        avg_bits:  target average bits per channel.
    Returns:
        1-D long tensor of per-channel bit-widths.
    """
    v = torch.as_tensor(variances, dtype=torch.float64).clamp_min(1e-12)
    log_gm = torch.log2(v).mean()
    b = avg_bits + 0.5 * (torch.log2(v) - log_gm)
    b = b.clamp(bmin, bmax)
    bi = torch.round(b).clamp(bmin, bmax).to(torch.long)
    # greedy fix-up toward the target average (add bits to highest-variance under-allocated
    # channels, remove from lowest-variance over-allocated channels)
    target_total = int(round(avg_bits * v.numel()))
    order = torch.argsort(v, descending=True)
    i = 0
    guard = 0
    while int(bi.sum()) < target_total and guard < 100000:
        c = order[i % order.numel()].item()
        if bi[c] < bmax:
            bi[c] += 1
        i += 1; guard += 1
    i = 0; guard = 0
    rorder = torch.flip(order, [0])
    while int(bi.sum()) > target_total and guard < 100000:
        c = rorder[i % rorder.numel()].item()
        if bi[c] > bmin:
            bi[c] -= 1
        i += 1; guard += 1
    return bi


@torch.no_grad()
def per_channel_fake_quant(c, bits_vec):
    """Asymmetric min-max fake-quant where channel j uses bits_vec[j] bits.
    c: (..., R); bits_vec: (R,) long. Channels with 0 bits are zeroed (dropped -> low-rank)."""
    *lead, R = c.shape
    out = torch.empty_like(c)
    flat = c.reshape(-1, R)
    for j in range(R):
        b = int(bits_vec[j])
        col = flat[:, j]
        if b <= 0:
            out.reshape(-1, R)[:, j] = 0
            continue
        if b >= 16:
            out.reshape(-1, R)[:, j] = col
            continue
        lo = col.min(); hi = col.max()
        qmax = (1 << b) - 1
        scale = (hi - lo).clamp_min(1e-8) / qmax
        q = torch.round((col - lo) / scale).clamp_(0, qmax)
        out.reshape(-1, R)[:, j] = q * scale + lo
    return out


@torch.no_grad()
def per_channel_quant_tokens(c, bits_vec, residual=0, group_tok=128):
    """Fast, vectorized per-channel asymmetric min-max fake-quant for a latent tensor.
    c: (..., T, R) (token dim = -2, channel dim = -1). Channel j uses bits_vec[j] bits.
    Min/max are taken per channel over TOKEN BLOCKS of `group_tok` (not globally): a single
    outlier token would otherwise blow up a channel's range and destroy resolution for all
    other tokens (LLM activations are heavy-tailed). Channels with 0 bits are dropped to 0
    (low-rank tail; valid for a mean-centered latent). The most recent `residual` tokens are
    kept fp16 (decode-step tokens, T==1, pass through losslessly)."""
    *lead, T, R = c.shape
    out = c.clone()
    nq = max(0, T - residual)
    if nq == 0:
        return out
    bv = bits_vec.clamp_min(0).to(torch.long)
    qmax = ((1 << bv) - 1).to(c.dtype)                     # (R,), 0 for dropped channels
    drop = (bv == 0)
    g = group_tok if (group_tok and group_tok > 0) else nq
    for s in range(0, nq, g):
        e = min(s + g, nq)
        blk = c[..., s:e, :]                               # (..., e-s, R)
        lo = blk.amin(dim=-2, keepdim=True); hi = blk.amax(dim=-2, keepdim=True)
        scale = (hi - lo) / qmax.clamp_min(1)
        q = torch.round((blk - lo) / scale.clamp_min(1e-8))
        q = torch.minimum(torch.clamp(q, min=0), qmax)
        deq = q * scale + lo
        if drop.any():
            deq = torch.where(drop, torch.zeros_like(deq), deq)
        out[..., s:e, :] = deq
    return out


@torch.no_grad()
def group_water_filling_quant(c, group_bits, group_size):
    """Per-token group quant (group over CHANNELS, like KIVI/LatentProjV2 -> cache-friendly)
    with a per-GROUP bit-width from water-filling. c: (..., R); group_bits: (n_groups,).
    Group g (channels [g*gs:(g+1)*gs]) uses group_bits[g] bits with per-token min/max over its
    gs channels. 0-bit groups are dropped to 0 (low-rank tail; valid for a centered latent)."""
    *lead, R = c.shape
    ng = int(group_bits.numel())
    gs = group_size
    if R != ng * gs:                                   # fallback: one group, mean bits
        b = int(group_bits.float().mean().round().clamp(0, 16))
        gb = torch.full((1,), b, device=c.device); return group_water_filling_quant(c, gb, R)
    x = c.reshape(*lead, ng, gs)
    lo = x.amin(-1, keepdim=True); hi = x.amax(-1, keepdim=True)        # (..., ng, 1)
    qmax = ((1 << group_bits.clamp_min(0).to(torch.long)) - 1).to(c.dtype)  # (ng,)
    qm = qmax.reshape(*([1] * len(lead)), ng, 1)
    scale = (hi - lo) / qm.clamp_min(1)
    q = torch.round((x - lo) / scale.clamp_min(1e-8))
    q = torch.minimum(torch.clamp(q, min=0), qm)
    deq = q * scale + lo
    drop = (qm == 0)
    if drop.any():
        deq = torch.where(drop.expand_as(deq), torch.zeros_like(deq), deq)
    return deq.reshape(*lead, R)


@torch.no_grad()
def static_quant_apply(c, half, scale, qmax):
    """Static calibrated per-channel asymmetric quant (cache-friendly, per-token, no dynamic
    min/max -> composes with the streaming KV cache). Channel j uses a fixed calibrated range
    [-half_j, +half_j] and qmax_j levels (all baked at build time from calibration std).
    c: (..., R); half/scale/qmax: (R,). Dropped channels have qmax==0 -> output 0."""
    q = torch.round((c + half) / scale.clamp_min(1e-12))
    q = torch.minimum(torch.clamp(q, min=0), qmax)
    out = q * scale - half
    drop = (qmax == 0)
    if drop.any():
        out = torch.where(drop, torch.zeros_like(out), out)
    return out


def static_quant_params(std, bits_vec, clip=4.0, dtype=torch.float16):
    """Build (half, scale, qmax) for static per-channel quant from per-channel std and bits.
    Range = +/- clip*std (centered latent); qmax = 2^bits - 1 levels."""
    std = torch.as_tensor(std).double()
    bv = torch.as_tensor(bits_vec).clamp_min(0).long()
    qmax = ((1 << bv) - 1).double()                 # 0 for dropped channels
    half = clip * std
    scale = (2 * half) / qmax.clamp_min(1)
    return half.to(dtype), scale.to(dtype), qmax.to(dtype)


def effective_bits_per_channel(bits_vec, group, fp16_scale_bits=32):
    """Honest average bits/element for a per-channel-allocated latent: mean(bits) + scale/zero
    overhead amortized over the group size (one fp16 scale+zero per group per active channel)."""
    bits_vec = torch.as_tensor(bits_vec, dtype=torch.float64)
    active = (bits_vec > 0).double().mean().clamp_min(1e-9)
    g = group if group and group > 0 else 1e9
    return float(bits_vec.mean() + fp16_scale_bits / g * active)
