"""kv_tie_eora.py — Step 1+2 for kvshare.md: per-layer/head K=V tie-ability diagnosis
(rho_KV + 3-term decomposition) and OFF-CACHE closed-form compensation, validated by
real perplexity on a softmax LM.

Architecture-aware (v2): supports
  - GPT-2 family   : Conv1D, fused c_attn (QKV), c_proj. MHA. No RoPE.  (clean linear testbed)
  - Llama family   : nn.Linear q/k/v/o_proj (transposed layout). MHA (7B/13B). RoPE on Q,K.
GQA (num_kv_heads < num_heads) is detected and rejected with a clear message (later work).

Pipeline
--------
1. Calibrate per-layer activation Gram  G_l = E[a a^T]  (a = input to q/k/v projections,
   i.e. exactly what the projections multiply: post-(layer/RMS)norm hidden state).
2. DIAGNOSIS (kvshare.md 3.1/3.4): per head, M = W_K^T G W_V, report
   rho_KV = sum_{lambda>0} lambda^2 / ||M||_F^2  (positive-symmetric energy fraction),
   plus asymmetric / negative-alignment / departure-from-normality split.
3. COMPENSATION variants, realized by IN-PLACE weight surgery (model's real softmax,
   causal mask, and RoPE are used unchanged), each PPL-evaluated then restored:
     - base        : untouched.
     - tie         : per head W_K := W_V := W_KV = mean(W_K, W_V). No compensation.
     - tie+offcache: tie, then closed-form correct W_Q (score) and W_O (value). OFF-cache
                     (Q recomputed each step, O outside cache) -> still 50% cache. PROPOSAL.
     - tie+eorakv-r: tie, then EoRA rank-r recovery added back onto K and V (residual
                     re-enters the cache; reference for cost-per-recovered-PPL).

Closed forms (G-weighted least squares; G^{1/2} cancels analytically), per head:
  Q (score):  WQ_new = WQ (WK^T G WKV)(WKV^T G WKV)^{-1}
  O (value):  WO_new = (WKV^T G WKV)^{-1}(WKV^T G WV) WO
  EoRA recover ΔW: rank-r minimizer of ||G^{1/2}(ΔW - LR)||_F.

NOTE on RoPE (Llama): the closed forms above are derived for un-rotated scores. RoPE is a
position-dependent orthogonal map applied to Q and K (not V) at attention time. We keep the
RoPE-agnostic correction and measure its real-model PPL cost — quantifying the RoPE gap is a
deliberate goal of this step (cf. kvshare.md 3.7 softmax/relative-position risk).
"""
from __future__ import annotations
import argparse, json, math, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- utils
def _robust_eigh(A):
    """Symmetric eigendecomposition robust to cuSOLVER non-convergence: try on-device, then
    fall back to CPU LAPACK (syevd) which is far more robust for ill-conditioned Grams."""
    try:
        return torch.linalg.eigh(A)
    except (torch._C._LinAlgError, RuntimeError):
        w, Q = torch.linalg.eigh(A.detach().cpu().double())
        return w.to(A.device, A.dtype), Q.to(A.device, A.dtype)


_TSVD = torch.linalg.svd


def _robust_svd(A, full_matrices=False):
    """SVD robust to cuSOLVER non-convergence: fall back to CPU LAPACK (gesdd) in float64."""
    try:
        return _TSVD(A, full_matrices=full_matrices)  # noqa: robust-svd-internal
    except (torch._C._LinAlgError, RuntimeError):
        U, S, Vh = _TSVD(A.detach().cpu().double(), full_matrices=full_matrices)
        return U.to(A.device, A.dtype), S.to(A.device, A.dtype), Vh.to(A.device, A.dtype)


def _sym_inv(A, ridge):
    A = (A + A.t()) * 0.5
    d = torch.diagonal(A).mean().clamp_min(1e-12)
    A = A + torch.eye(A.shape[0], device=A.device, dtype=A.dtype) * ridge * d
    return torch.linalg.inv(A)


def _sym_sqrt_inv_sqrt(G, ridge):
    G = (G + G.t()) * 0.5
    d = torch.diagonal(G).mean().clamp_min(1e-12)
    G = G + torch.eye(G.shape[0], device=G.device, dtype=G.dtype) * ridge * d
    w, Q = _robust_eigh(G)
    w = w.clamp_min(1e-12)
    s, si = w.sqrt(), w.rsqrt()
    return (Q * s) @ Q.t(), (Q * si) @ Q.t()


# ----------------------------------------------------------- architecture abstraction
class AttnHeads:
    """Per-head Q/K/V/O access in the row-vector convention  q = a @ W_Q,
    with W_Q,W_K,W_V of shape (d, head_dim) and W_O of shape (head_dim, d).
    Handles GPT-2 (Conv1D, fused) and Llama (Linear, separate, transposed)."""
    def __init__(self, model):
        self.model = model
        cfg = model.config
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self.arch = "gpt2"
            self.layers = model.transformer.h
            self.L = cfg.n_layer
            self.H = cfg.n_head
            self.Hkv = cfg.n_head
            self.d = cfg.n_embd
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            self.arch = "llama"
            self.layers = model.model.layers
            self.L = cfg.num_hidden_layers
            self.H = cfg.num_attention_heads
            self.Hkv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
            self.d = cfg.hidden_size
        else:
            raise ValueError("unsupported architecture")
        self.hd = getattr(cfg, "head_dim", self.d // self.H)
        # GQA support: K and V projections produce kv_dim (smaller than d).
        # Q and O are still d-dim. tie/offcache/eorakv per-head variants are MHA-only.
        self.kv_dim = self.Hkv * self.hd
        self.is_gqa = (self.Hkv != self.H)

    def hslice(self, h):
        return slice(h * self.hd, (h + 1) * self.hd)

    # --- input module whose INPUT is the activation `a` that projections multiply
    def input_module(self, l):
        if self.arch == "gpt2":
            return self.layers[l].attn.c_attn
        return self.layers[l].self_attn.q_proj

    # --- per-head getters: return float64 (d,hd) or (hd,d) row-conv blocks (detached)
    def _get(self, l):
        if self.arch == "gpt2":
            ca = self.layers[l].attn.c_attn.weight          # (d, 3d) row-conv
            cp = self.layers[l].attn.c_proj.weight          # (d, d)  row-conv
            return ca, cp
        sa = self.layers[l].self_attn
        return sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight, sa.o_proj.weight

    def WQh(self, l, h):
        if self.arch == "gpt2":
            return self._get(l)[0][:, 0 * self.d:1 * self.d][:, self.hslice(h)].double()
        return self._get(l)[0][self.hslice(h), :].t().double()   # (in,hd)

    def WKh(self, l, h):
        if self.arch == "gpt2":
            return self._get(l)[0][:, 1 * self.d:2 * self.d][:, self.hslice(h)].double()
        return self._get(l)[1][self.hslice(h), :].t().double()

    def WVh(self, l, h):
        if self.arch == "gpt2":
            return self._get(l)[0][:, 2 * self.d:3 * self.d][:, self.hslice(h)].double()
        return self._get(l)[2][self.hslice(h), :].t().double()

    def WOh(self, l, h):
        if self.arch == "gpt2":
            return self._get(l)[1][self.hslice(h), :].double()   # (hd, d)
        return self._get(l)[3][:, self.hslice(h)].t().double()   # (hd, d)

    # --- per-head setters (write row-conv block back into module weight, with transpose)
    @torch.no_grad()
    def set_head(self, l, h, WQ=None, WK=None, WV=None, WO=None):
        sl = self.hslice(h)
        if self.arch == "gpt2":
            ca = self.layers[l].attn.c_attn.weight
            cp = self.layers[l].attn.c_proj.weight
            dt = ca.dtype
            if WQ is not None: ca[:, 0 * self.d:1 * self.d][:, sl] = WQ.to(dt)
            if WK is not None: ca[:, 1 * self.d:2 * self.d][:, sl] = WK.to(dt)
            if WV is not None: ca[:, 2 * self.d:3 * self.d][:, sl] = WV.to(dt)
            if WO is not None: cp[sl, :] = WO.to(dt)
        else:
            sa = self.layers[l].self_attn
            if WQ is not None: sa.q_proj.weight[sl, :] = WQ.t().to(sa.q_proj.weight.dtype)
            if WK is not None: sa.k_proj.weight[sl, :] = WK.t().to(sa.k_proj.weight.dtype)
            if WV is not None: sa.v_proj.weight[sl, :] = WV.t().to(sa.v_proj.weight.dtype)
            if WO is not None: sa.o_proj.weight[:, sl] = WO.t().to(sa.o_proj.weight.dtype)

    # --- snapshot / restore the attention weights of given layers (memory-cheap)
    def snapshot(self, layers):
        snap = {}
        for l in layers:
            mats = self._get(l)
            snap[l] = [m.detach().clone() for m in mats]
        return snap

    @torch.no_grad()
    def restore(self, snap):
        for l, mats in snap.items():
            cur = self._get(l)
            for c, m in zip(cur, mats):
                c.copy_(m)


# --------------------------------------------------------------------- calibration
@torch.no_grad()
def collect_grams(model, heads, windows, device):
    L, d = heads.L, heads.d
    G = [torch.zeros(d, d, dtype=torch.float64, device=device) for _ in range(L)]
    cnt = [0 for _ in range(L)]
    hooks = []

    def mk(l):
        def pre_hook(mod, args):
            a = args[0].reshape(-1, args[0].shape[-1]).double()
            G[l] += a.t() @ a
            cnt[l] += a.shape[0]
        return pre_hook

    for l in range(L):
        hooks.append(heads.input_module(l).register_forward_pre_hook(mk(l)))
    try:
        for ids in windows:
            model(ids.to(device).unsqueeze(0))
    finally:
        for h in hooks:
            h.remove()
    for l in range(L):
        G[l] /= max(cnt[l], 1)
    return G


# ------------------------------------------------------------------------ diagnosis
@torch.no_grad()
def diagnose(heads, G):
    L, H = heads.L, heads.H
    per_layer = []
    for l in range(L):
        Gl = G[l]
        rows = []
        for h in range(H):
            WK = heads.WKh(l, h)
            WV = heads.WVh(l, h)
            M = WK.t() @ Gl @ WV
            fro2 = (M * M).sum().clamp_min(1e-30)
            Msym = (M + M.t()) * 0.5
            Masym = (M - M.t()) * 0.5
            lam = torch.linalg.eigvalsh(Msym)
            pos, neg = lam[lam > 0], lam[lam < 0]
            eig = torch.linalg.eigvals(M)
            rows.append(dict(
                rho=float((pos * pos).sum() / fro2),
                asym=float((Masym * Masym).sum() / fro2),
                neg=float((neg * neg).sum() / fro2),
                dep=float((fro2 - (eig.abs() ** 2).sum()).clamp_min(0.0) / fro2),
            ))
        per_layer.append(dict(
            layer=l,
            rho_mean=round(sum(r["rho"] for r in rows) / H, 4),
            asym_mean=round(sum(r["asym"] for r in rows) / H, 4),
            neg_mean=round(sum(r["neg"] for r in rows) / H, 4),
            dep_mean=round(sum(r["dep"] for r in rows) / H, 4),
            rho_min=round(min(r["rho"] for r in rows), 4),
            rho_max=round(max(r["rho"] for r in rows), 4),
        ))
    overall = sum(p["rho_mean"] for p in per_layer) / L
    return dict(rho_overall=round(overall, 4), per_layer=per_layer)


# ---------------------------------------------------------------- weight surgery ops
def eora_lowrank(dW, Gs, Gis, r):
    """Rank-r minimizer of ||G^{1/2}(dW - LR)||_F, given precomputed Gs=G^{1/2}, Gis=G^{-1/2}."""
    A = Gs @ dW
    U, S, Vh = _robust_svd(A, full_matrices=False)
    r = min(r, S.shape[0])
    return Gis @ ((U[:, :r] * S[:r]) @ Vh[:r, :])


def rope_avg(C):
    """RoPE phase-average  E_Delta[R_Delta C R_Delta^T]  in closed form, assuming
    rotate_half pairing (p <-> p+hd/2) and full frequency decorrelation. Each 2D pair
    block keeps only its rotation-covariant part alpha*I + beta*J; cross-frequency
    blocks vanish. C: (hd, hd) -> (hd, hd)."""
    hd = C.shape[0]; h = hd // 2
    idx = torch.arange(h, device=C.device)
    cFF = torch.diagonal(C[:h, :h])      # C[p,p]
    cSS = torch.diagonal(C[h:, h:])      # C[q,q],  q=p+h
    cFS = torch.diagonal(C[:h, h:])      # C[p,q]
    cSF = torch.diagonal(C[h:, :h])      # C[q,p]
    alpha = 0.5 * (cFF + cSS)
    beta = 0.5 * (cSF - cFS)
    out = torch.zeros_like(C)
    out[idx, idx] = alpha
    out[idx + h, idx + h] = alpha
    out[idx, idx + h] = -beta
    out[idx + h, idx] = beta
    return out


def _oproj_module(heads, l):
    if heads.arch == "gpt2":
        return heads.layers[l].attn.c_proj
    return heads.layers[l].self_attn.o_proj


@torch.no_grad()
def _accumulate_streaming_stats(heads, orig_snap, tied_snap, calib_w, device):
    """Two-pass per window: TIED forward captures (Y, Z_tied); ORIG forward captures
    Z_orig. Accumulate per-layer  GYY += Y^T Y,  GYZ += Y^T (Z_orig - Z_tied)  in fp64.

    Memory: 2 * L * d^2 * 8 B (fp64 stats) + 2 * snapshots (orig + tied). NO per-token
    buffers, so calibration can scale to arbitrarily many windows.
    Cost per window: 2 forwards + 2 weight restores (8 GB memcopy each ≈ tens of ms)."""
    L, d = heads.L, heads.d
    din = heads.H * heads.hd            # o_proj INPUT dim (= concat of all Q heads' context)
    dout = d                            # o_proj OUTPUT dim (= hidden). din != dout for Qwen3.
    GYY = [torch.zeros(din, din, dtype=torch.float64, device=device) for _ in range(L)]
    GYZ = [torch.zeros(din, dout, dtype=torch.float64, device=device) for _ in range(L)]
    n_per = [0 for _ in range(L)]
    cache = {"y": [None] * L, "z_tied": [None] * L, "z_orig": [None] * L,
             "phase": "tied"}                                          # "tied" or "orig"

    pre_hooks, post_hooks = [], []

    def mk_pre(l):
        def hook(mod, args):
            if cache["phase"] == "tied":
                cache["y"][l] = args[0].reshape(-1, din).detach().double()
        return hook

    def mk_post(l):
        def hook(mod, args, out):
            o = out.reshape(-1, dout).detach().double()
            if cache["phase"] == "tied":
                cache["z_tied"][l] = o
            else:
                cache["z_orig"][l] = o
        return hook

    for l in range(L):
        op = _oproj_module(heads, l)
        pre_hooks.append(op.register_forward_pre_hook(mk_pre(l)))
        post_hooks.append(op.register_forward_hook(mk_post(l)))

    try:
        for ids in calib_w:
            ids_d = ids.to(device).unsqueeze(0)
            # TIED forward
            heads.restore(tied_snap)
            cache["phase"] = "tied"
            heads.model(ids_d)
            # ORIG forward
            heads.restore(orig_snap)
            cache["phase"] = "orig"
            heads.model(ids_d)
            # accumulate (Y, ΔZ) per layer for this window
            for l in range(L):
                y = cache["y"][l]
                dz = cache["z_orig"][l] - cache["z_tied"][l]
                GYY[l] += y.t() @ y
                GYZ[l] += y.t() @ dz
                n_per[l] += y.shape[0]
                cache["y"][l] = cache["z_tied"][l] = cache["z_orig"][l] = None
    finally:
        for h in pre_hooks + post_hooks:
            h.remove()
        heads.restore(tied_snap)                                       # leave model TIED

    return GYY, GYZ, n_per


def _get_full_rowconv_live(heads, l, kind):
    """Return (d_in, d_out) row-conv weight read from the LIVE model (float64 clone)."""
    d = heads.d
    if heads.arch == "gpt2":
        if kind == "Q": return heads.layers[l].attn.c_attn.weight[:, 0 * d:1 * d].clone().double()
        if kind == "K": return heads.layers[l].attn.c_attn.weight[:, 1 * d:2 * d].clone().double()
        if kind == "V": return heads.layers[l].attn.c_attn.weight[:, 2 * d:3 * d].clone().double()
        if kind == "O": return heads.layers[l].attn.c_proj.weight.clone().double()
    else:
        sa = heads.layers[l].self_attn
        if kind == "Q": return sa.q_proj.weight.t().clone().double()
        if kind == "K": return sa.k_proj.weight.t().clone().double()
        if kind == "V": return sa.v_proj.weight.t().clone().double()
        if kind == "O": return sa.o_proj.weight.t().clone().double()
    raise ValueError(kind)


@torch.no_grad()
def _set_full_rowconv_live(heads, l, kind, W_rowconv):
    """OVERWRITE the live module weight with the supplied (d_in, d_out) row-conv tensor."""
    d = heads.d
    if heads.arch == "gpt2":
        if kind == "Q":
            heads.layers[l].attn.c_attn.weight[:, 0 * d:1 * d] = W_rowconv.to(heads.layers[l].attn.c_attn.weight.dtype)
        elif kind == "K":
            heads.layers[l].attn.c_attn.weight[:, 1 * d:2 * d] = W_rowconv.to(heads.layers[l].attn.c_attn.weight.dtype)
        elif kind == "V":
            heads.layers[l].attn.c_attn.weight[:, 2 * d:3 * d] = W_rowconv.to(heads.layers[l].attn.c_attn.weight.dtype)
        elif kind == "O":
            heads.layers[l].attn.c_proj.weight.copy_(W_rowconv.to(heads.layers[l].attn.c_proj.weight.dtype))
        else:
            raise ValueError(kind)
    else:
        sa = heads.layers[l].self_attn
        if kind == "Q":   sa.q_proj.weight.copy_(W_rowconv.t().to(sa.q_proj.weight.dtype))
        elif kind == "K": sa.k_proj.weight.copy_(W_rowconv.t().to(sa.k_proj.weight.dtype))
        elif kind == "V": sa.v_proj.weight.copy_(W_rowconv.t().to(sa.v_proj.weight.dtype))
        elif kind == "O": sa.o_proj.weight.copy_(W_rowconv.t().to(sa.o_proj.weight.dtype))
        else: raise ValueError(kind)


@torch.no_grad()
def fit_apply_mla(heads, G, mla_layers, rank, ridge, device):
    """Post-hoc Multi-head Latent Attention: per layer, jointly factorize the stacked
    KV projection  [W_K | W_V]  (row-conv (d, 2d))  into  W_DOWN (d, r) @ [W_UK | W_UV]
    (r, 2d)  via G-weighted rank-r SVD (EoRA pattern):

        K = G^{1/2} [W_K | W_V]   (shape d, 2d)
        K = U Σ V^T,   K_r = U_r Σ_r V_r^T
        [W_K_new | W_V_new] = G^{-1/2} K_r   (rank ≤ r, equal to W_DOWN @ W_UP)

    At inference one would cache  c_t = h_t W_DOWN  (rank-r per token) instead of K,V.
    Per-token cache: r  vs  2d originally  →  cache fraction  = r / (2d).

    This PoC writes the (rank-r) reconstructions  W_K_new, W_V_new  back into the model
    so PPL can be measured directly. The cache fraction is purely analytic.

    RoPE: K is reconstructed first (= c_t @ W_UK), then the normal RoPE rotates it. No
    DeepSeek-style PE split in this PoC (we measure how much that costs us)."""
    L, d, kv = heads.L, heads.d, heads.kv_dim
    fro_errs, kept, spec_dWK = [], [], []
    for l in mla_layers:
        Gl = G[l]
        Csqrt, Cinv = _sym_sqrt_inv_sqrt(Gl, ridge)
        WK = _get_full_rowconv_live(heads, l, "K")        # MHA: (d,d), GQA: (d, kv_dim)
        WV = _get_full_rowconv_live(heads, l, "V")        # MHA: (d,d), GQA: (d, kv_dim)
        WKV = torch.cat([WK, WV], dim=1)                  # (d, 2*kv_dim)
        K = Csqrt @ WKV                                   # whitened image
        U, S, Vh = _robust_svd(K, full_matrices=False)
        r = min(rank, S.shape[0])
        K_r = (U[:, :r] * S[:r]) @ Vh[:r, :]              # rank-r best F-norm approx in G-norm
        WKV_recon = Cinv @ K_r                            # de-whiten back to weight space
        WK_new = WKV_recon[:, 0:kv].contiguous()
        WV_new = WKV_recon[:, kv:2*kv].contiguous()
        fro = float(torch.linalg.norm(K - K_r, ord="fro") / torch.linalg.norm(K, ord="fro"))
        fro_errs.append(fro)
        kept.append(float((S[:r] ** 2).sum() / (S ** 2).sum().clamp_min(1e-30)))
        dWK = WK - WK_new
        spec_dWK.append(float(torch.linalg.norm(dWK, ord="fro") / torch.linalg.norm(WK, ord="fro")))
        _set_full_rowconv_live(heads, l, "K", WK_new)
        _set_full_rowconv_live(heads, l, "V", WV_new)
    print(f"   [mla r={rank}] G-norm relerr med={sorted(fro_errs)[len(fro_errs)//2]:.3f} "
          f"max={max(fro_errs):.3f}  pred-energy kept med={sorted(kept)[len(kept)//2]:.3f}  "
          f"||ΔW_K||/||W_K|| med={sorted(spec_dWK)[len(spec_dWK)//2]:.3f}",
          flush=True)


@torch.no_grad()
def _lowrank_gweighted(W, Gl, rank, ridge):
    """G-weighted rank-r approximation of a single (d, m) weight: returns reconstruction
    minimizing ||G^{1/2}(W - hat)||_F.  rank<=0 means 'no compression' (return W)."""
    if rank is None or rank <= 0:
        return W
    Csqrt, Cinv = _sym_sqrt_inv_sqrt(Gl, ridge)
    A = Csqrt @ W
    U, S, Vh = _robust_svd(A, full_matrices=False)
    r = min(rank, S.shape[0])
    A_r = (U[:, :r] * S[:r]) @ Vh[:r, :]
    return Cinv @ A_r


@torch.no_grad()
def collect_grams_and_mean(model, heads, windows, device):
    """Like collect_grams but also returns per-layer activation mean E[h] (for the bias /
    mean-centering co-design). Returns (G_list, hmean_list)."""
    L, d = heads.L, heads.d
    G = [torch.zeros(d, d, dtype=torch.float64, device=device) for _ in range(L)]
    S = [torch.zeros(d, dtype=torch.float64, device=device) for _ in range(L)]
    cnt = [0 for _ in range(L)]
    hooks = []

    def mk(l):
        def pre_hook(mod, args):
            a = args[0].reshape(-1, args[0].shape[-1]).double()
            G[l] += a.t() @ a
            S[l] += a.sum(0)
            cnt[l] += a.shape[0]
        return pre_hook

    for l in range(L):
        hooks.append(heads.input_module(l).register_forward_pre_hook(mk(l)))
    try:
        for ids in windows:
            model(ids.to(device).unsqueeze(0))
    finally:
        for h in hooks:
            h.remove()
    for l in range(L):
        G[l] /= max(cnt[l], 1)
        S[l] /= max(cnt[l], 1)
    return G, S


@torch.no_grad()
def lowrank_gweighted_factors(W, Gl, rank, ridge):
    """G-weighted rank-r factorization of (d_in, m) weight W:  W ≈ W_DOWN @ W_UP,
    W_DOWN:(d_in, r), W_UP:(r, m), minimizing ||G^{1/2}(W - W_DOWN W_UP)||_F.
    rank<=0 -> full (W_DOWN=W, W_UP=I)."""
    d_in, m = W.shape
    if rank is None or rank <= 0:
        eye = torch.eye(m, device=W.device, dtype=W.dtype)
        return W.clone(), eye
    Csqrt, Cinv = _sym_sqrt_inv_sqrt(Gl, ridge)
    A = Csqrt @ W
    U, S, Vh = _robust_svd(A, full_matrices=False)
    r = min(rank, S.shape[0])
    sq = S[:r].sqrt()
    W_DOWN = Cinv @ (U[:, :r] * sq)            # (d_in, r)
    W_UP = (sq.unsqueeze(1) * Vh[:r, :])       # (r, m)
    return W_DOWN, W_UP


@torch.no_grad()
def fit_apply_mla_split(heads, G, mla_layers, rank_k, rank_v, ridge, device):
    """SEPARATE (decoupled) low-rank compression of K and V with independent ranks.
    rank_k / rank_v <= 0  => that side is kept FULL (uncompressed).
    Per-token cache = rank_k_eff + rank_v_eff  (full side counts as kv_dim).
    Lets us isolate K-only vs V-only compressibility/compensability."""
    d, kv = heads.d, heads.kv_dim
    for l in mla_layers:
        Gl = G[l]
        WK = _get_full_rowconv_live(heads, l, "K")        # (d, kv)
        WV = _get_full_rowconv_live(heads, l, "V")        # (d, kv)
        WK_new = _lowrank_gweighted(WK, Gl, rank_k, ridge)
        WV_new = _lowrank_gweighted(WV, Gl, rank_v, ridge)
        _set_full_rowconv_live(heads, l, "K", WK_new)
        _set_full_rowconv_live(heads, l, "V", WV_new)
    rk = "full" if (rank_k is None or rank_k <= 0) else str(rank_k)
    rv = "full" if (rank_v is None or rank_v <= 0) else str(rank_v)
    cache_k = kv if (rank_k is None or rank_k <= 0) else rank_k
    cache_v = kv if (rank_v is None or rank_v <= 0) else rank_v
    print(f"   [mla_split rK={rk} rV={rv}] per-token cache = {cache_k+cache_v} "
          f"(/{2*kv} baseline = {(cache_k+cache_v)/(2*kv):.4f}x)", flush=True)


def _get_full_rowconv(heads, l, snap, kind):
    """Return (d_in, d_out) row-conv weight from a snapshot dict. kind ∈ {Q,K,V,O}."""
    d = heads.d
    if heads.arch == "gpt2":
        ca, cp = snap[l]
        if kind == "Q": return ca[:, 0 * d:1 * d].clone().double()
        if kind == "K": return ca[:, 1 * d:2 * d].clone().double()
        if kind == "V": return ca[:, 2 * d:3 * d].clone().double()
        if kind == "O": return cp.clone().double()
    else:
        qw, kw, vw, ow = snap[l]
        if kind == "Q": return qw.t().clone().double()
        if kind == "K": return kw.t().clone().double()
        if kind == "V": return vw.t().clone().double()
        if kind == "O": return ow.t().clone().double()
    raise ValueError(kind)


@torch.no_grad()
def _set_full_rowconv_add(heads, l, kind, dW_rowconv):
    """In-place: weight_rowconv += dW_rowconv  (write back to module weight)."""
    d = heads.d
    if heads.arch == "gpt2":
        if kind == "Q":
            w = heads.layers[l].attn.c_attn.weight
            w[:, 0 * d:1 * d] += dW_rowconv.to(w.dtype)
        elif kind == "O":
            w = heads.layers[l].attn.c_proj.weight
            w += dW_rowconv.to(w.dtype)
        else:
            raise NotImplementedError(kind)
    else:
        if kind == "Q":
            w = heads.layers[l].self_attn.q_proj.weight
        elif kind == "O":
            w = heads.layers[l].self_attn.o_proj.weight
        else:
            raise NotImplementedError(kind)
        w += dW_rowconv.t().to(w.dtype)


@torch.no_grad()
def fit_apply_eora_wq(heads, G, orig_snap, tied_layers, rank, ridge):
    """Add a RANK-r residual to W_Q that recovers orig Q activations.

    Closed form (no extra forwards): the target moment is  X^T(X W_Q_orig^T - X W_Q_tied^T)
    = G ΔW_Q^T  where ΔW_Q = W_Q_orig - W_Q_tied (row-conv).  Whitened RRR identical
    in structure to EoRA-WO but the cross-moment is analytic from G, not from forwards.

    Off-cache: W_Q is never in the KV cache → 0.50× cache preserved."""
    norms, kept = [], []
    for l in tied_layers:
        Gl = G[l]
        WQ_orig = _get_full_rowconv(heads, l, orig_snap, "Q")        # (d, d)
        # Current model state is TIED. Read live WQ_tied via a one-tuple snap:
        if heads.arch == "gpt2":
            ca = heads.layers[l].attn.c_attn.weight
            WQ_tied = ca[:, 0:heads.d].clone().double()
        else:
            WQ_tied = heads.layers[l].self_attn.q_proj.weight.t().clone().double()
        dW = WQ_orig - WQ_tied                                       # (d, d) row-conv
        Csqrt, Cinv = _sym_sqrt_inv_sqrt(Gl, ridge)
        B = Gl @ dW                                                  # cross-moment
        W = (Cinv @ Cinv) @ B
        Mw = Csqrt @ W
        U, S, Vh = _robust_svd(Mw, full_matrices=False)
        r = min(rank, S.shape[0])
        Mw_r = (U[:, :r] * S[:r]) @ Vh[:r, :]
        dW_r = Cinv @ Mw_r
        norms.append(float(torch.linalg.norm(dW_r, ord="fro")))
        kept.append(float((S[:r] ** 2).sum() / (S ** 2).sum().clamp_min(1e-30)))
        _set_full_rowconv_add(heads, l, "Q", dW_r)
    print(f"   [eora_wq r={rank}] |dW|_F med={sorted(norms)[len(norms)//2]:.2f} "
          f"max={max(norms):.2f}  pred-energy kept top-r/total: "
          f"min={min(kept):.2f} med={sorted(kept)[len(kept)//2]:.2f}", flush=True)


@torch.no_grad()
def _rrr_precompute(heads, GYY, GYZ, n_per, ridge):
    """Per-layer factorization INDEPENDENT of rank — compute ONCE then sweep ranks
    cheaply via _rrr_apply(precomp, r). Returns list of (Cinv, U, S, Vh) per layer,
    stored in fp32 to fit in memory (~6 GB total on Llama2-7B)."""
    L = heads.L
    out = []
    for l in range(L):
        A = GYY[l] / max(n_per[l], 1)
        B = GYZ[l] / max(n_per[l], 1)
        Csqrt, Cinv = _sym_sqrt_inv_sqrt(A, ridge)
        W = (Cinv @ Cinv) @ B
        Mw = Csqrt @ W
        U, S, Vh = _robust_svd(Mw, full_matrices=False)
        out.append((Cinv.float(), U.float(), S.float(), Vh.float()))
    return out


@torch.no_grad()
def _rrr_apply(heads, precomp, rank):
    """Apply rank-r residual to W_O using precomputed factorization. Cheap (no SVD)."""
    L = len(precomp)
    ranks_norm, kept = [], []
    for l in range(L):
        Cinv, U, S, Vh = (t.double() for t in precomp[l])
        r = min(rank, S.shape[0])
        Mw_r = (U[:, :r] * S[:r]) @ Vh[:r, :]
        dW_r = Cinv @ Mw_r
        ranks_norm.append(float(torch.linalg.norm(dW_r, ord="fro")))
        kept.append(float((S[:r] ** 2).sum() / (S ** 2).sum().clamp_min(1e-30)))
        if heads.arch == "gpt2":
            cp = heads.layers[l].attn.c_proj.weight
            cp.copy_((cp.double() + dW_r).to(cp.dtype))
        else:
            op = heads.layers[l].self_attn.o_proj.weight
            op.copy_((op.double() + dW_r.t()).to(op.dtype))
    print(f"   [eora_wo r={rank}] |dW|_F med={sorted(ranks_norm)[L//2]:.2f} "
          f"max={max(ranks_norm):.2f}  pred-energy kept top-r/total: "
          f"min={min(kept):.2f} med={sorted(kept)[L//2]:.2f}", flush=True)


def _waterfill_budget(precomp, total_budget):
    """CARE-style water-filling on per-layer S² normalized energy. Returns r_per[l]
    summing to total_budget. At each step, allocates the next rank to the layer
    with the highest *remaining* normalized predictive energy."""
    L = len(precomp)
    cumS2 = []        # cumulative S² per layer
    totS2 = []
    cap = []
    for l in range(L):
        _, _, S, _ = precomp[l]
        S2 = (S.float() ** 2)
        cumS2.append(torch.cumsum(S2, dim=0).cpu().tolist())
        totS2.append(float(S2.sum()))
        cap.append(int(S.shape[0]))
    r_per = [0] * L
    # priority queue would be ideal but L=32 is tiny; linear scan is fine.
    for _ in range(total_budget):
        best_l, best_score = -1, -1.0
        for l in range(L):
            if r_per[l] >= cap[l]:
                continue
            cov = cumS2[l][r_per[l] - 1] if r_per[l] > 0 else 0.0
            rem = 1.0 - cov / max(totS2[l], 1e-30)
            if rem > best_score:
                best_score, best_l = rem, l
        if best_l < 0:
            break
        r_per[best_l] += 1
    return r_per


@torch.no_grad()
def _rrr_apply_adaptive(heads, precomp, total_budget):
    """Apply per-layer adaptive-rank residual via water-filling on S² energy."""
    r_per = _waterfill_budget(precomp, total_budget)
    L = len(precomp)
    for l in range(L):
        Cinv, U, S, Vh = (t.double() for t in precomp[l])
        r = r_per[l]
        if r <= 0:
            continue
        Mw_r = (U[:, :r] * S[:r]) @ Vh[:r, :]
        dW_r = Cinv @ Mw_r
        if heads.arch == "gpt2":
            cp = heads.layers[l].attn.c_proj.weight
            cp.copy_((cp.double() + dW_r).to(cp.dtype))
        else:
            op = heads.layers[l].self_attn.o_proj.weight
            op.copy_((op.double() + dW_r.t()).to(op.dtype))
    r_sorted = sorted(r_per)
    print(f"   [eora_wo adaptive budget={total_budget}] "
          f"r per layer: min={r_sorted[0]} med={r_sorted[L//2]} max={r_sorted[-1]} "
          f"avg={total_budget/L:.2f}  (full: {r_per})", flush=True)
    return r_per


@torch.no_grad()
def apply_variant(heads, G, kind, tie_layers, rank, ridge, device):
    """IN-PLACE weight surgery for the requested variant. Caller snapshots/restores.
    kinds: noop, tie, offcache, offcache_rope, offcache_q, offcache_rope_q, eorakv.

    _q suffix = SCORE-ONLY (Q corrected, W_O untouched). Use as the base for DOC, since
    DOC will solve the value-side via a per-layer ridge regression on attention output."""
    for l in tie_layers:
        Gl = G[l].to(device)
        Gs = Gis = None
        if kind == "eorakv":
            Gs, Gis = _sym_sqrt_inv_sqrt(Gl, ridge)   # per LAYER once (not per head)
        for h in range(heads.H):
            WQ = heads.WQh(l, h); WK = heads.WKh(l, h)
            WV = heads.WVh(l, h); WO = heads.WOh(l, h)
            if kind == "noop":                       # roundtrip self-test: get->set identity
                heads.set_head(l, h, WQ=WQ, WK=WK, WV=WV, WO=WO)
                continue
            WKV = 0.5 * (WK + WV)
            Kxx = _sym_inv(WKV.t() @ Gl @ WKV, ridge)
            if kind == "tie":
                heads.set_head(l, h, WK=WKV, WV=WKV)
            elif kind == "offcache":
                new_WQ = WQ @ (WK.t() @ Gl @ WKV) @ Kxx
                new_WO = Kxx @ (WKV.t() @ Gl @ WV) @ WO
                heads.set_head(l, h, WQ=new_WQ, WK=WKV, WV=WKV, WO=new_WO)
            elif kind == "offcache_rope":
                Ck = rope_avg(WK.t() @ Gl @ WKV)
                Cv = rope_avg(WKV.t() @ Gl @ WKV)     # symmetric -> beta=0 -> PD diagonal
                new_WQ = WQ @ Ck @ _sym_inv(Cv, ridge)
                new_WO = Kxx @ (WKV.t() @ Gl @ WV) @ WO   # value side: no RoPE, unchanged form
                heads.set_head(l, h, WQ=new_WQ, WK=WKV, WV=WKV, WO=new_WO)
            elif kind == "offcache_q":
                new_WQ = WQ @ (WK.t() @ Gl @ WKV) @ Kxx
                heads.set_head(l, h, WQ=new_WQ, WK=WKV, WV=WKV)        # W_O left for DOC
            elif kind == "offcache_rope_q":
                Ck = rope_avg(WK.t() @ Gl @ WKV)
                Cv = rope_avg(WKV.t() @ Gl @ WKV)
                new_WQ = WQ @ Ck @ _sym_inv(Cv, ridge)
                heads.set_head(l, h, WQ=new_WQ, WK=WKV, WV=WKV)        # W_O left for DOC
            elif kind == "eorakv":
                new_WK = WKV + eora_lowrank(WK - WKV, Gs, Gis, rank)
                new_WV = WKV + eora_lowrank(WV - WKV, Gs, Gis, rank)
                heads.set_head(l, h, WK=new_WK, WV=new_WV)
            else:
                raise ValueError(kind)


# ------------------------------------------------------------------------- ppl eval
@torch.no_grad()
def perplexity(model, windows, device):
    nll, ntok = 0.0, 0
    for ids in windows:
        ids = ids.to(device).unsqueeze(0)
        out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        nll += float(out.loss) * n
        ntok += n
    return math.exp(nll / max(ntok, 1))


def _pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / (math.sqrt(sxx * syy) + 1e-30)


def _spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):                              # average ties
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return _pearson(ranks(x), ranks(y))


@torch.no_grad()
def sweep_and_pareto(heads, G, base_ppl, eval_w, dev, diag, ridge, comp_kind):
    """(a) per-layer single-tie ΔPPL (no comp) vs rho -> predictive-power correlation.
       (b) greedy rho-ranked Pareto: tie top-k most tie-able layers (with comp_kind)."""
    L = heads.L
    print("[sweep] per-layer single-tie (no comp) vs rho ...", flush=True)
    rows = []
    for l in range(L):
        snap = heads.snapshot([l])
        try:
            apply_variant(heads, G, "tie", [l], 0, ridge, dev)
            ppl = perplexity(heads.model, eval_w, dev)
        finally:
            heads.restore(snap); del snap; torch.cuda.empty_cache()
        rho_l = diag["per_layer"][l]["rho_mean"]
        rows.append(dict(layer=l, rho=rho_l, ppl=round(ppl, 3), dppl=round(ppl - base_ppl, 3)))
        print(f"   L{l:02d} rho={rho_l:.3f}  tie1 ppl={ppl:9.3f}  dppl={ppl-base_ppl:+.3f}", flush=True)
    dps = [r["dppl"] for r in rows]
    # does ANY diagnostic component predict per-layer tie cost? (negative rho => predictive)
    corr = {}
    for key in ("rho", "asym", "neg", "dep"):
        col = [diag["per_layer"][r["layer"]][f"{key}_mean"] for r in rows]
        corr[key] = dict(pearson=round(_pearson(col, dps), 4),
                         spearman=round(_spearman(col, dps), 4))
    pear, spear = corr["rho"]["pearson"], corr["rho"]["spearman"]
    print(f"[sweep] corr(diag, dppl): " +
          "  ".join(f"{k}:r={v['pearson']:+.2f}/s={v['spearman']:+.2f}" for k, v in corr.items()) +
          "  (rho negative = predictive)", flush=True)

    order_best = [r["layer"] for r in sorted(rows, key=lambda r: r["rho"], reverse=True)]
    order_worst = list(reversed(order_best))
    order_cost = [r["layer"] for r in sorted(rows, key=lambda r: r["dppl"])]  # cheapest measured first
    ks = sorted({k for k in (4, 8, 16, 24, L) if k <= L})
    pareto = {"rho_best": [], "rho_worst": [], "measured_cost": []}
    for tag, order in (("rho_best", order_best), ("rho_worst", order_worst),
                       ("measured_cost", order_cost)):
        print(f"[pareto] tie top-k ({tag} order) + {comp_kind} ...", flush=True)
        for k in ks:
            layers = order[:k]
            snap = heads.snapshot(layers)
            try:
                apply_variant(heads, G, comp_kind, layers, 0, ridge, dev)
                ppl = perplexity(heads.model, eval_w, dev)
            finally:
                heads.restore(snap); del snap; torch.cuda.empty_cache()
            cache = 1.0 - 0.5 * k / L                 # tied layers store 1 KV instead of 2
            pareto[tag].append(dict(k=k, cache_frac=round(cache, 3), ppl=round(ppl, 3)))
            print(f"   {tag} k={k:2d} cache={cache:.3f} ppl={ppl:9.3f}", flush=True)
    return dict(per_layer=rows, pearson=round(pear, 4), spearman=round(spear, 4),
                corr=corr, pareto=pareto, comp_kind=comp_kind)


def make_windows(tok, text, seq_len, max_windows):
    ids = tok(text, return_tensors="pt").input_ids[0]
    wins = []
    for i in range(0, ids.shape[0] - 1, seq_len):
        w = ids[i:i + seq_len]
        if w.shape[0] < 16:
            break
        wins.append(w)
        if max_windows and len(wins) >= max_windows:
            break
    return wins


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--calib-windows", type=int, default=64)
    ap.add_argument("--eval-windows", type=int, default=128)
    ap.add_argument("--ranks", default="4,8,16")
    ap.add_argument("--tie-layers", default="all", help="'all' or csv of layer idx")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--wo-ranks", default="4,8,16,32,64",
                    help="csv of EoRA-on-W_O residual ranks to evaluate (off-cache, 0.5x cache)")
    ap.add_argument("--wo-ridge", type=float, default=1e-2,
                    help="ridge for the EoRA-on-W_O least squares")
    ap.add_argument("--mla-ranks", default="",
                    help="csv of MLA latent ranks (joint K+V down-projection). "
                         "Cache fraction per rank = r / (2 d_model).")
    ap.add_argument("--mla-ridge", type=float, default=1e-3)
    ap.add_argument("--mla-wo-bases", default="",
                    help="csv of MLA latent ranks; for each, run the WO sweep below.")
    ap.add_argument("--mla-wo-ranks", default="",
                    help="csv of EoRA-WO residual ranks to add ON TOP of MLA "
                         "(off-cache compensation, cache unchanged).")
    ap.add_argument("--mla-wo-adaptive-budgets", default="",
                    help="csv of TOTAL rank budgets across all layers for water-filling "
                         "adaptive WO (CARE-style). e.g. '64,128,256' compares to uniform "
                         "r=2,4,8 on a 32-layer model.")
    ap.add_argument("--kv-split-specs", default="",
                    help="K/V asymmetry experiment. csv of 'rK:rV' pairs (0=keep full). "
                         "For each, apply decoupled MLA then sweep --mla-wo-ranks EoRA-WO. "
                         "e.g. '0:128,128:0,256:256' = V-only/K-only/symmetric.")
    ap.add_argument("--sweep", action="store_true",
                    help="run per-layer rho->dppl correlation + greedy rho-ranked Pareto")
    ap.add_argument("--sweep-eval-windows", type=int, default=24)
    ap.add_argument("--out", default="kv_tie_eora.json")
    args = ap.parse_args()

    dev = args.device
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    dtype = getattr(torch, args.dtype)
    print(f"[load] {args.model} dtype={args.dtype} -> {dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    heads = AttnHeads(model)
    print(f"[cfg] arch={heads.arch} L={heads.L} H={heads.H} d={heads.d} head_dim={heads.hd}", flush=True)

    from datasets import load_dataset
    print("[data] wikitext-2-raw-v1", flush=True)
    ds_tr = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ds_te = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    calib_text = "\n\n".join(t for t in ds_tr["text"] if t.strip())
    eval_text = "\n\n".join(t for t in ds_te["text"] if t.strip())
    calib_w = make_windows(tok, calib_text, args.seq_len, args.calib_windows)
    eval_w = make_windows(tok, eval_text, args.seq_len, args.eval_windows)
    print(f"[data] calib windows={len(calib_w)} eval windows={len(eval_w)}", flush=True)

    t0 = time.time()
    print("[calib] collecting per-layer Gram G ...", flush=True)
    G = collect_grams(model, heads, calib_w, dev)
    print(f"[calib] done in {time.time()-t0:.1f}s", flush=True)

    print("[diag] per-head rho_KV ...", flush=True)
    diag = diagnose(heads, G)
    print(f"[diag] overall rho_KV = {diag['rho_overall']}", flush=True)
    for p in diag["per_layer"]:
        print(f"   L{p['layer']:02d} rho={p['rho_mean']:.3f} "
              f"(min {p['rho_min']:.3f}/max {p['rho_max']:.3f})  "
              f"asym={p['asym_mean']:.3f} neg={p['neg_mean']:.3f} dep={p['dep_mean']:.3f}",
              flush=True)

    tie_layers = list(range(heads.L)) if args.tie_layers == "all" else \
        [int(x) for x in args.tie_layers.split(",") if x.strip() != ""]
    ranks = [int(x) for x in args.ranks.split(",") if x.strip() != ""]

    results = {}

    # Per-variant runner. EoRA-WO uses STREAMING stats: snapshot orig + tied, do two
    # forwards per calib window, accumulate sufficient stats (GYY, GYZ) directly.
    # No per-token buffers → calibration can scale arbitrarily.
    def run(name, kind, rank=8, eora_wo=False, eora_wo_rank=8, eora_wo_ridge=None):
        tb = time.time()
        if kind == "base":
            ppl = perplexity(model, eval_w, dev)
            print(f"[ppl] {name:32s} = {ppl:10.3f}   ({time.time()-tb:.1f}s)", flush=True)
            return ppl
        orig_snap = heads.snapshot(tie_layers)
        try:
            apply_variant(heads, G, kind, tie_layers, rank, args.ridge, dev)
            if eora_wo:
                tied_snap = heads.snapshot(tie_layers)
                try:
                    GYY, GYZ, n_per = _accumulate_streaming_stats(
                        heads, orig_snap, tied_snap, calib_w, dev)
                    precomp = _rrr_precompute(heads, GYY, GYZ, n_per,
                                              eora_wo_ridge if eora_wo_ridge is not None else args.ridge)
                    del GYY, GYZ
                    _rrr_apply(heads, precomp, eora_wo_rank)
                    del precomp
                finally:
                    del tied_snap
                    torch.cuda.empty_cache()
            ppl = perplexity(model, eval_w, dev)
        finally:
            heads.restore(orig_snap)
            del orig_snap
            torch.cuda.empty_cache()
        print(f"[ppl] {name:32s} = {ppl:10.3f}   ({time.time()-tb:.1f}s)", flush=True)
        return ppl

    print("[run] variants ...", flush=True)
    results["base"] = run("base", "base")
    results["noop"] = run("noop (selftest)", "noop")
    if abs(results["noop"] - results["base"]) > 1e-3 * results["base"]:
        print(f"[WARN] noop selftest PPL {results['noop']:.4f} != base {results['base']:.4f} "
              f"-> get/set transpose mismatch!", flush=True)
    else:
        print("[ok] noop roundtrip == base (get/set transpose consistent)", flush=True)
    has_rope = heads.arch == "llama" or hasattr(model.config, "rope_theta")
    # MHA-only variants: K=V tying / per-head closed-form / EoRA-on-KV. Skip on GQA.
    if not heads.is_gqa:
        results["tie"] = run("tie", "tie")
        results["tie+offcache"] = run("tie+offcache", "offcache")
        if has_rope:
            results["tie+offcache_rope"] = run("tie+offcache_rope", "offcache_rope")
    else:
        print(f"[skip] tie/offcache/eorakv variants are MHA-only; "
              f"this model is GQA (H={heads.H}, Hkv={heads.Hkv}). Running MLA path.", flush=True)
        # Need a 'tie' baseline for the summary's "gap recovered" arithmetic; use base as
        # a trivial reference so gap=0 and recovered is None.
        results["tie"] = results["base"]
    # EoRA-on-W_O residual: per-head closed-form THEN add a rank-r residual to W_O
    # via whitened reduced-rank regression. Still off-cache → 0.50× cache.
    # STATS ARE COMPUTED ONCE — sweep ranks cheaply on top of fixed (GYY, GYZ).
    best_full_kind = "offcache_rope" if has_rope else "offcache"
    wo_ranks = [int(x) for x in args.wo_ranks.split(",") if x.strip()]
    if wo_ranks and not heads.is_gqa:
        print(f"[eora_wo] accumulating streaming stats for base={best_full_kind}, "
              f"calib={len(calib_w)} windows ...", flush=True)
        orig_snap = heads.snapshot(tie_layers)
        try:
            apply_variant(heads, G, best_full_kind, tie_layers, 0, args.ridge, dev)
            tied_snap = heads.snapshot(tie_layers)
            tx = time.time()
            GYY, GYZ, n_per = _accumulate_streaming_stats(
                heads, orig_snap, tied_snap, calib_w, dev)
            print(f"[eora_wo] stats done in {time.time()-tx:.1f}s "
                  f"({2*len(calib_w)} forwards)", flush=True)
            precomp = _rrr_precompute(heads, GYY, GYZ, n_per, args.wo_ridge)
            del GYY, GYZ
            print(f"[eora_wo] precompute done; sweeping {len(wo_ranks)} ranks", flush=True)
            for r in wo_ranks:
                tb = time.time()
                heads.restore(tied_snap)
                _rrr_apply(heads, precomp, r)
                ppl = perplexity(model, eval_w, dev)
                results[f"tie+{best_full_kind}+eora_wo_r{r}"] = ppl
                print(f"[ppl] tie+{best_full_kind}+eora_wo r{r:<4d}"
                      f"= {ppl:10.3f}   ({time.time()-tb:.1f}s)", flush=True)
        finally:
            heads.restore(orig_snap)
            del orig_snap
            try: del tied_snap, precomp
            except NameError: pass
            torch.cuda.empty_cache()

    if not heads.is_gqa:
        for r in ranks:
            results[f"tie+eorakv_r{r}"] = run(f"tie+eorakv_r{r}", "eorakv", rank=r)

    # Post-hoc MLA sweep (joint K+V down-projection via G-weighted rank-r factorization).
    # Cache fraction per token per layer = r / (2 * d_model). NOT tied to "tie" baseline —
    # this is a separate compression mechanism. PoC: pure MLA, no W_O compensation.
    mla_ranks = [int(x) for x in args.mla_ranks.split(",") if x.strip()]
    for mr in mla_ranks:
        tb = time.time()
        orig_snap = heads.snapshot(tie_layers)
        try:
            fit_apply_mla(heads, G, tie_layers, mr, args.mla_ridge, dev)
            ppl = perplexity(model, eval_w, dev)
        finally:
            heads.restore(orig_snap); del orig_snap; torch.cuda.empty_cache()
        results[f"mla_r{mr}"] = ppl
        print(f"[ppl] mla r{mr:<4d}  cache={mr / (2 * heads.kv_dim):.4f}× "
              f"= {ppl:10.3f}   ({time.time()-tb:.1f}s)", flush=True)

    # MLA + closed-form EoRA-WO compensation (the novel wedge vs CARE/TransMLA).
    # For each MLA base in --mla-wo-bases (csv): snapshot orig, apply MLA, snapshot tied,
    # accumulate stats ONCE, precompute SVD ONCE, sweep WO ranks cheaply.
    mla_wo_bases = [int(x) for x in args.mla_wo_bases.split(",") if x.strip()]
    mla_wo_ranks = [int(x) for x in args.mla_wo_ranks.split(",") if x.strip()]
    for mbase in mla_wo_bases:
        if not mla_wo_ranks:
            break
        cache_frac_mla = mbase / (2 * heads.kv_dim)
        print(f"\n[mla+eora_wo] base = MLA r={mbase} (cache {cache_frac_mla:.4f}×); "
              f"accumulating streaming stats, calib={len(calib_w)} windows ...", flush=True)
        orig_snap = heads.snapshot(tie_layers)
        try:
            fit_apply_mla(heads, G, tie_layers, mbase, args.mla_ridge, dev)
            mla_only_ppl = perplexity(model, eval_w, dev)
            results[f"mla_r{mbase}_base"] = mla_only_ppl
            print(f"[ppl] mla r{mbase} (base for WO comp) = {mla_only_ppl:10.3f}", flush=True)
            tied_snap = heads.snapshot(tie_layers)
            tx = time.time()
            GYY, GYZ, n_per = _accumulate_streaming_stats(
                heads, orig_snap, tied_snap, calib_w, dev)
            print(f"[mla+eora_wo] stats done in {time.time()-tx:.1f}s "
                  f"({2*len(calib_w)} forwards)", flush=True)
            precomp = _rrr_precompute(heads, GYY, GYZ, n_per, args.wo_ridge)
            del GYY, GYZ
            print(f"[mla+eora_wo] precompute done; sweeping {len(mla_wo_ranks)} ranks",
                  flush=True)
            for r in mla_wo_ranks:
                tb = time.time()
                heads.restore(tied_snap)
                _rrr_apply(heads, precomp, r)
                ppl = perplexity(model, eval_w, dev)
                results[f"mla_r{mbase}+eora_wo_r{r}"] = ppl
                print(f"[ppl] mla r{mbase} + eora_wo r{r:<4d} cache={cache_frac_mla:.4f}× "
                      f"= {ppl:10.3f}   ({time.time()-tb:.1f}s)", flush=True)
            # Adaptive water-filling at matched total budgets (compare to uniform).
            adaptive_budgets = [int(x) for x in args.mla_wo_adaptive_budgets.split(",")
                                if x.strip()]
            for tb_budget in adaptive_budgets:
                t0 = time.time()
                heads.restore(tied_snap)
                _rrr_apply_adaptive(heads, precomp, tb_budget)
                ppl = perplexity(model, eval_w, dev)
                results[f"mla_r{mbase}+eora_wo_adaptive_B{tb_budget}"] = ppl
                avg_r = tb_budget / heads.L
                print(f"[ppl] mla r{mbase} + eora_wo adaptive B={tb_budget} "
                      f"(avg r={avg_r:.2f}) cache={cache_frac_mla:.4f}× "
                      f"= {ppl:10.3f}   ({time.time()-t0:.1f}s)", flush=True)
        finally:
            heads.restore(orig_snap)
            del orig_snap
            try: del tied_snap, precomp
            except NameError: pass
            try: del GYY, GYZ
            except NameError: pass
            torch.cuda.empty_cache()

    # K/V asymmetry experiment: decoupled MLA (independent rK, rV) + EoRA-WO sweep.
    # Tests the core thesis: V error is off-cache compensable (linear), K error is not
    # (nonlinear through softmax). 0 = keep that side full.
    split_specs = [s for s in args.kv_split_specs.split(",") if s.strip()]
    for spec in split_specs:
        rk, rv = (int(x) for x in spec.split(":"))
        ck = heads.kv_dim if rk <= 0 else rk
        cv = heads.kv_dim if rv <= 0 else rv
        cache_frac = (ck + cv) / (2 * heads.kv_dim)
        tag = f"split_rK{rk}_rV{rv}"
        print(f"\n[kv-split] rK={rk} rV={rv}  cache={cache_frac:.4f}× ; "
              f"calib={len(calib_w)} ...", flush=True)
        orig_snap = heads.snapshot(tie_layers)
        try:
            fit_apply_mla_split(heads, G, tie_layers, rk, rv, args.mla_ridge, dev)
            ppl0 = perplexity(model, eval_w, dev)
            results[f"{tag}_base"] = ppl0
            print(f"[ppl] {tag} (no WO) cache={cache_frac:.4f}× = {ppl0:10.3f}", flush=True)
            tied_snap = heads.snapshot(tie_layers)
            GYY, GYZ, n_per = _accumulate_streaming_stats(
                heads, orig_snap, tied_snap, calib_w, dev)
            precomp = _rrr_precompute(heads, GYY, GYZ, n_per, args.wo_ridge)
            del GYY, GYZ
            for r in mla_wo_ranks:
                heads.restore(tied_snap)
                _rrr_apply(heads, precomp, r)
                ppl = perplexity(model, eval_w, dev)
                results[f"{tag}+eora_wo_r{r}"] = ppl
                print(f"[ppl] {tag} + eora_wo r{r:<3d} cache={cache_frac:.4f}× "
                      f"= {ppl:10.3f}", flush=True)
        finally:
            heads.restore(orig_snap)
            del orig_snap
            try: del tied_snap, precomp
            except NameError: pass
            try: del GYY, GYZ
            except NameError: pass
            torch.cuda.empty_cache()

    noop_ppl = results.pop("noop", None)               # keep selftest out of the result table
    base, tie = results["base"], results["tie"]
    gap = tie - base
    table = []
    for name, ppl in results.items():
        if name == "base":
            cache, recov = 1.0, None
        elif name.startswith("tie+eorakv_r"):
            r = int(name.split("_r")[1])
            cache = 0.5 * (1 + 2 * r / heads.hd)             # K and V each get rank-r residual
            recov = (tie - ppl) / gap if gap else None
        elif name.startswith("mla_r") and ("+eora_wo" in name):
            r_mla = int(name.split("mla_r")[1].split("+")[0])   # MLA rank determines cache
            cache = r_mla / (2 * heads.kv_dim)                  # vs original baseline (GQA-aware)
            recov = (tie - ppl) / gap if gap else None
        elif name.startswith("mla_r"):
            r = int(name.split("_r")[1].split("_")[0])          # handle "mla_r512" and "mla_r512_base"
            cache = r / (2 * heads.kv_dim)                      # joint MLA latent: r dim per token
            recov = (tie - ppl) / gap if gap else None
        elif name.startswith("split_rK"):
            rk = int(name.split("rK")[1].split("_")[0])
            rv = int(name.split("rV")[1].split("+")[0].split("_")[0])
            ck = heads.kv_dim if rk <= 0 else rk
            cv = heads.kv_dim if rv <= 0 else rv
            cache = (ck + cv) / (2 * heads.kv_dim)
            recov = (tie - ppl) / gap if gap else None
        else:
            cache = 0.5                                          # off-cache compensation → 0.5×
            recov = (tie - ppl) / gap if gap else None
        table.append(dict(variant=name, ppl=round(ppl, 3),
                          cache_frac=round(cache, 4),
                          recovered=None if recov is None else round(recov, 3)))

    sweep = None
    if args.sweep:
        sweep_w = eval_w[:args.sweep_eval_windows]
        comp = "offcache_rope" if has_rope else "offcache"
        sweep = sweep_and_pareto(heads, G, base, sweep_w, dev, diag, args.ridge, comp)

    out = dict(model=args.model, dtype=args.dtype,
               config=dict(arch=heads.arch, L=heads.L, H=heads.H, d=heads.d, head_dim=heads.hd),
               seq_len=args.seq_len, calib_windows=len(calib_w), eval_windows=len(eval_w),
               tie_layers=tie_layers, ridge=args.ridge, noop_selftest=noop_ppl,
               diagnosis=diag, ppl=results, table=table, sweep=sweep)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== SUMMARY ===", flush=True)
    print(f"{'variant':18s} {'ppl':>10s} {'cache':>7s} {'%gap recovered':>15s}", flush=True)
    for row in table:
        rec = "" if row["recovered"] is None else f"{100*row['recovered']:.1f}%"
        print(f"{row['variant']:18s} {row['ppl']:10.3f} {row['cache_frac']:7.2f} {rec:>15s}", flush=True)
    if gap:
        print(f"\n[done] base {base:.3f} -> tie {tie:.3f} (gap {gap:.3f}); "
              f"off-cache recovers {100*(tie-results['tie+offcache'])/gap:.1f}% at 0.50x cache",
              flush=True)
    if sweep:
        print(f"\n=== SWEEP: does any diagnostic predict per-layer tie cost? ===", flush=True)
        for k, v in sweep["corr"].items():
            print(f"   corr({k:4s}, dppl_tie1): pearson={v['pearson']:+.3f} "
                  f"spearman={v['spearman']:+.3f}", flush=True)
        print(f"   (rho expected NEGATIVE if self-adjointness predicts tie-ability)", flush=True)
        print(f"\n=== PARETO (tie top-k layers, +{sweep['comp_kind']}) ===", flush=True)
        print(f"{'k':>3s} {'cache':>7s} {'rho_best':>10s} {'rho_worst':>10s} {'measured':>10s}", flush=True)
        for rb, rw, mc in zip(sweep["pareto"]["rho_best"], sweep["pareto"]["rho_worst"],
                              sweep["pareto"]["measured_cost"]):
            print(f"{rb['k']:3d} {rb['cache_frac']:7.3f} {rb['ppl']:10.3f} "
                  f"{rw['ppl']:10.3f} {mc['ppl']:10.3f}", flush=True)
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
