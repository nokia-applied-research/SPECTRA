"""Shared math for SPECTRA-Flash, in HF Llama conventions (rotate_half RoPE on the full head dim; the record's
rope part uses adjacent pairs (2p, 2p+1) with its own angles theta_p)."""
from __future__ import annotations

import torch

F8_MAX = 448.0                      # float8_e4m3fn


def rope_pairs(x, pos, theta):
    """x (..., S, r) holds r/2 RoPE pairs (2p, 2p+1); pos (S,); theta (r/2,). Returns fp32."""
    ang = pos.float()[:, None] * theta[None, :]
    c, s = ang.cos(), ang.sin()
    x2 = x.float().unflatten(-1, (-1, 2))
    x0, x1 = x2[..., 0], x2[..., 1]
    return torch.stack([x0 * c - x1 * s, x1 * c + x0 * s], -1).flatten(-2)


def rope_full(x, cos, sin):
    """HF rotate_half RoPE. x (..., S, hd); cos/sin (S, hd)."""
    h = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., h:], x[..., :h]), dim=-1) * sin


def cos_sin_table(inv_freq, S, dev):
    ang = torch.arange(S, device=dev).float()[:, None] * inv_freq[None, :]
    emb = torch.cat([ang, ang], -1)
    return emb.cos(), emb.sin()


def fp8_round(x, scale):
    """Value after an fp8 (e4m3) round trip with a static per-channel scale."""
    y = (x.float() / scale).clamp_(-F8_MAX, F8_MAX).to(torch.float8_e4m3fn).float() * scale
    return y.to(x.dtype)


def repeat_kv(x, g):
    """(kvh, S, hd) -> (kvh * g, S, hd), each kv head repeated g times in place (HF order)."""
    return x.repeat_interleave(g, 0)


class LayerW:
    """fp32 copies of one attention layer's projections (transposed, x @ W), plus its head layout."""

    def __init__(self, attn):
        self.wq = attn.q_proj.weight.float().t().contiguous()
        self.wk = attn.k_proj.weight.float().t().contiguous()
        self.wv = attn.v_proj.weight.float().t().contiguous()
        self.wo = attn.o_proj.weight.float().t().contiguous()
        self.hd = attn.head_dim
        self.nq = self.wq.shape[1] // self.hd
        self.kvh = self.wk.shape[1] // self.hd
        self.gq = self.nq // self.kvh
        self.scaling = attn.scaling
