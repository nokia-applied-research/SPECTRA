"""Closed-form initialization of one block's record maps (the starting point of the distillation).

For a block of n layers with stacked inputs H = [h_g0 | ... | h_g0+n-1]:
  rope part : for each RoPE frequency, the (layer, kv head) copies of that frequency's key pair are mixed by a PCA
              (mixing across copies commutes with RoPE). Components are ranked by energy x f(theta L), where
              f(x) = 2 (1 - sin x / x) is the mean rotation error over distances up to L, and the top r/2 are kept
              as RoPE pairs s = H W_r with angles theta_p. The query side W_QR reads the matching query pairs.
  latent    : the keys' residual (not covered by the rope part) and the values are fitted jointly by a
              Gram-whitened SVD: c = H W_d (R dims), K_res ~ c W_UK, V ~ c W_UV.
"""
from __future__ import annotations

import torch


def _sqrt_pair(M, ridge=1e-3):
    """(M^1/2, M^-1/2) of a symmetric PSD matrix with a relative ridge."""
    M = 0.5 * (M + M.T)
    s = M.diagonal().mean().clamp_min(1e-8)
    w, Q = torch.linalg.eigh(M + torch.eye(M.shape[0], device=M.device) * ridge * s)
    ws = w.clamp_min(1e-8).sqrt()
    return (Q * ws) @ Q.T, (Q * (1 / ws)) @ Q.T


@torch.no_grad()
def closed_form_block(model, grp, Gj, Gl, R, r, query_weighted=True, rope_horizon=32768):
    """Gj: joint Gram of the stacked block inputs (n d x n d); Gl: per-layer Grams. Returns fp32 maps
    {Wd (n d, R), Wr (n d, r), theta (r/2), WuK[li] (R, kv), WuV[li] (R, kv), WQR[li] (d, H r), info}."""
    layers = model.model.layers; d = model.config.hidden_size; n = len(grp)
    at0 = layers[grp[0]].self_attn
    kv = at0.k_proj.weight.shape[0]; hd = at0.head_dim; kvh = kv // hd; half = hd // 2
    nq = model.config.num_attention_heads; gq = nq // kvh
    dev = Gj.device
    inv_freq = model.model.rotary_emb.inv_freq.float().to(dev)
    WK = lambda l: layers[l].self_attn.k_proj.weight.t().float()
    WV = lambda l: layers[l].self_attn.v_proj.weight.t().float()
    WQ = lambda l: layers[l].self_attn.q_proj.weight.t().float()
    nk = n * kv
    TK = torch.block_diag(*[WK(l) for l in grp]); TV = torch.block_diag(*[WV(l) for l in grp])
    Kcov = TK.T @ Gj @ TK
    qe = []                                              # query energy per (layer, kv head, frequency)
    for i, l in enumerate(grp):
        dq = (WQ(l) * (Gl[i] @ WQ(l))).sum(0).view(nq, hd)
        qe.append((dq[:, :half] + dq[:, half:]).view(kvh, gq, half).sum(1))
    qe = torch.stack(qe)
    rot_err = 2.0 * (1.0 - torch.sin(inv_freq * rope_horizon) / (inv_freq * rope_horizon))
    cands = []
    for i in range(half):
        idx0 = torch.tensor([li * kv + gg * hd + i for li in range(n) for gg in range(kvh)], device=dev)
        idx1 = idx0 + half
        M = Kcov[idx0][:, idx0] + Kcov[idx1][:, idx1]
        w = qe[:, :, i].reshape(-1).clamp_min(1e-12).sqrt() if query_weighted else torch.ones(n * kvh, device=dev)
        lam, V = torch.linalg.eigh(w[:, None] * M * w[None, :])
        for kk in range(1, min(r // 2, n * kvh) + 1):
            cands.append((lam[-kk].item() * rot_err[i].item(), i, V[:, -kk].clone(), w, idx0, idx1))
    tot = sum(c[0] for c in cands)
    sel = sorted(cands, key=lambda c: -c[0])[: r // 2]
    A = torch.zeros(nk, r, device=dev); Bm = torch.zeros(r, nk, device=dev)
    WQR = [torch.zeros(d, nq, r, device=dev) for _ in grp]
    theta = torch.zeros(r // 2, device=dev)
    for p, (_, i, v, w, idx0, idx1) in enumerate(sel):
        a = w * v; b = v / w
        A[idx0, 2 * p] = a; A[idx1, 2 * p + 1] = a
        Bm[2 * p, idx0] = b; Bm[2 * p + 1, idx1] = b
        theta[p] = inv_freq[i]
        for li, l in enumerate(grp):
            bh = b[li * kvh:(li + 1) * kvh].repeat_interleave(gq)
            wq = WQ(l).view(d, nq, hd)
            WQR[li][:, :, 2 * p] = wq[:, :, i] * bh
            WQR[li][:, :, 2 * p + 1] = wq[:, :, i + half] * bh
    TKres = TK @ (torch.eye(nk, device=dev) - A @ Bm)
    Gs = _sqrt_pair(Gj)
    kw = 1.0 / (Gs[0] @ TKres).norm().clamp_min(1e-8); vw = 1.0 / (Gs[0] @ TV).norm().clamp_min(1e-8)
    U, Sv, Vh = torch.linalg.svd(Gs[0] @ torch.cat([TKres * kw, TV * vw], dim=1), full_matrices=False)
    sr = Sv[:R].clamp_min(1e-8).sqrt()
    Wd = (Gs[1] @ U[:, :R]) * sr[None, :]; Wu = sr[:, None] * Vh[:R]
    WuK = Wu[:, :nk] / kw; WuV = Wu[:, nk:] / vw
    return dict(Wd=Wd, Wr=TK @ A, theta=theta,
                WuK=[WuK[:, li * kv:(li + 1) * kv].contiguous() for li in range(n)],
                WuV=[WuV[:, li * kv:(li + 1) * kv].contiguous() for li in range(n)],
                WQR=[x.reshape(d, nq * r).contiguous() for x in WQR],
                info=dict(rope_kept=sum(c[0] for c in sel) / max(tot, 1e-12), freqs=sorted({c[1] for c in sel}),
                          latent_energy=(Sv[:R] ** 2).sum().item() / (Sv ** 2).sum().item()))
