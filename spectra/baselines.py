"""kv_baselines.py — faithful reimplementations of competing KV-cache compression methods,
all evaluated on the SAME NIAH harness (niah.py) for apples-to-apples comparison.

Two families:
  QUANTIZATION (operate on the actual post-RoPE KV cache via a custom Cache subclass):
    * kivi        Liu et al. 2024 — per-CHANNEL K (group over tokens), per-TOKEN V (group
                  over channels), recent `residual` tokens kept fp16.
    * turboquant  Zandieh et al. 2025 (ICLR'26) — random rotation -> per-coordinate
                  MSE-optimal (Lloyd-Max) scalar quant; per-vector norm kept fp16.
    * polarquant  Han/Kacham/Zandieh 2025 — random preconditioning -> recursive polar
                  transform -> per-level Lloyd-Max angle quant; per-vector radius kept fp16.
                  (No per-block scale/zero, that is the method's whole point.)
  TOKEN EVICTION (compress the prefill cache by selecting important tokens):
    * snapkv      Li et al. 2024 — vote with last `window` queries' attention, max-pool
                  cluster, top-(budget-window) per kv-head, keep + observation window.
    * pyramidkv   Cai/Zhang et al. 2024 — SnapKV selection with a per-LAYER pyramidal
                  budget (more in low layers, less in high layers).

Effective compression is accounted honestly per method (see eff_bits_* / budgets).
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


# =============================== scalar quant utils ==========================
def _lloyd_max(samples: torch.Tensor, n_levels: int, iters: int = 60):
    """1-D Lloyd-Max (k-means) centroids for the empirical distribution of `samples`."""
    s, _ = torch.sort(samples.flatten().float())
    q = torch.linspace(0, 1, n_levels + 2, device=s.device)[1:-1]
    c = torch.quantile(s, q)
    for _ in range(iters):
        bnd = (c[1:] + c[:-1]) / 2
        idx = torch.bucketize(s, bnd)
        new = c.clone()
        for k in range(n_levels):
            m = idx == k
            if m.any():
                new[k] = s[m].mean()
        if torch.allclose(new, c, atol=1e-7):
            c = new; break
        c = new
    return c


def _quant_to_codebook(x: torch.Tensor, centroids: torch.Tensor):
    bnd = (centroids[1:] + centroids[:-1]) / 2
    idx = torch.bucketize(x.flatten(), bnd).clamp_(0, centroids.numel() - 1)
    return centroids[idx].reshape(x.shape)


def rand_orthogonal(d, device, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    return Q.to(device=device, dtype=dtype)


# =============================== TurboQuant =================================
class TurboQuantizer:
    """Random rotation -> per-coordinate Lloyd-Max scalar quant. Norm stored fp16."""
    def __init__(self, d, bits, device, seed=0):
        self.d, self.bits = d, bits
        self.R = rand_orthogonal(d, device, torch.float32, seed)          # (d,d)
        # coords of a rotated unit-norm vector ~ N(0, 1/d); design codebook on N(0,1)
        g = torch.Generator(device="cpu").manual_seed(seed + 1)
        cb = _lloyd_max(torch.randn(400000, generator=g), 1 << bits).to(device)
        self.cb = cb                                                       # for N(0,1)

    @torch.no_grad()
    def __call__(self, x):
        # x: (..., d). rotate, per-vector normalize to unit per-coord variance, quant.
        xf = x.float()
        u = xf @ self.R.T                                                  # rotate
        s = u.norm(dim=-1, keepdim=True) / math.sqrt(self.d)              # per-vec scale
        s = s.clamp_min(1e-8)
        uq = _quant_to_codebook(u / s, self.cb) * s
        return (uq @ self.R).to(x.dtype)                                  # rotate back

    def eff_bits(self):
        return self.bits + 16.0 / self.d


# =============================== PolarQuant =================================
def _polar_fwd(x):
    """Recursive polar transform. x: (..., d), d a power of two.
    Returns (radius (...,1), [angles per level])."""
    angles = []
    cur = x
    while cur.shape[-1] > 1:
        a = cur[..., 0::2]; b = cur[..., 1::2]
        r = torch.hypot(a, b)
        angles.append(torch.atan2(b, a))
        cur = r
    return cur, angles


def _polar_inv(radius, angles):
    cur = radius
    for theta in reversed(angles):
        a = cur * torch.cos(theta); b = cur * torch.sin(theta)
        cur = torch.stack([a, b], dim=-1).reshape(*a.shape[:-1], a.shape[-1] * 2)
    return cur


class PolarQuantizer:
    """Random preconditioning -> recursive polar -> per-level Lloyd-Max angle quant.
    Radius stored fp16 (the only full-precision scalar; no per-block scale/zero)."""
    def __init__(self, d, bits, device, seed=0):
        assert (d & (d - 1)) == 0, "PolarQuant needs power-of-two head_dim"
        self.d, self.bits = d, bits
        self.P = rand_orthogonal(d, device, torch.float32, seed)
        g = torch.Generator(device="cpu").manual_seed(seed + 7)
        samp = (torch.randn(20000, d, generator=g)).to(device) @ self.P.T
        _, ang = _polar_fwd(samp)                                          # design codebooks
        self.cbs = [_lloyd_max(a, 1 << bits) for a in ang]                 # one per level

    @torch.no_grad()
    def __call__(self, x):
        xf = x.float() @ self.P.T
        r, ang = _polar_fwd(xf)
        angq = [_quant_to_codebook(a, cb) for a, cb in zip(ang, self.cbs)]
        rec = _polar_inv(r, angq)
        return (rec @ self.P).to(x.dtype)

    def eff_bits(self):
        return (16.0 + (self.d - 1) * self.bits) / self.d


# =============================== QuaRot-INT2 ================================
def _hadamard_matrix(d, device, dtype):
    """Normalized Sylvester-Hadamard matrix H (d a power of two), H @ H.T = I."""
    assert (d & (d - 1)) == 0, "Hadamard needs power-of-two head_dim"
    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    H = H / math.sqrt(d)
    return H.to(device=device, dtype=dtype)


class QuaRotQuantizer:
    """QuaRot-style KV quant: fixed Hadamard rotation (data-oblivious, outlier-spreading) ->
    per-token asymmetric min/max INT quant over the rotated channels -> rotate back.
    One fp16 scale+zero per token vector (group = head_dim)."""
    def __init__(self, d, bits, device, seed=0):
        self.d, self.bits = d, bits
        self.H = _hadamard_matrix(d, device, torch.float32)         # (d,d), symmetric-ish, orthonormal

    @torch.no_grad()
    def __call__(self, x):
        xf = x.float()
        u = xf @ self.H                                             # rotate (H orthonormal)
        lo = u.amin(-1, keepdim=True); hi = u.amax(-1, keepdim=True)
        qmax = (1 << self.bits) - 1
        scale = (hi - lo).clamp_min(1e-8) / qmax
        q = torch.round((u - lo) / scale).clamp_(0, qmax)
        uq = q * scale + lo
        return (uq @ self.H.T).to(x.dtype)                         # rotate back

    def eff_bits(self):
        return self.bits + 32.0 / self.d                           # fp16 scale+zero per token


# =============================== KIVI =======================================
def _quant_groups(x, bits, group):
    *lead, n = x.shape
    g = group if (group and group > 0 and n % group == 0) else n
    xr = x.reshape(*lead, n // g, g)
    lo = xr.amin(-1, keepdim=True); hi = xr.amax(-1, keepdim=True)
    qmax = (1 << bits) - 1
    scale = (hi - lo).clamp_min(1e-8) / qmax
    q = torch.round((xr - lo) / scale).clamp_(0, qmax)
    return (q * scale + lo).reshape(*lead, n)


class KiviQuantizer:
    """per-CHANNEL K (group tokens) / per-TOKEN V (group channels). Set is_key at call."""
    def __init__(self, bits, gk, gv):
        self.bits, self.gk, self.gv = bits, gk, gv

    @torch.no_grad()
    def quant_K(self, K):   # K: (n, d) -> per-channel, group over tokens
        return _quant_groups(K.t(), self.bits, self.gk).t()

    @torch.no_grad()
    def quant_V(self, V):   # V: (n, d) -> per-token, group over channels
        return _quant_groups(V, self.bits, self.gv)

    def eff_bits(self):
        return self.bits + 32.0 / min(self.gk, self.gv)


# =============================== QuantCache =================================
class QuantCache(DynamicCache):
    """DynamicCache that fake-quantizes the post-RoPE K/V the moment they are stored.
    The most recent `residual` tokens of the prefill are kept fp16 (KIVI convention;
    set residual=0 for turbo/polar). Decode-step tokens (len==1) are kept fp16."""
    def __init__(self, method, kq, vq=None, residual=0):
        super().__init__()
        self.method = method        # 'kivi' | 'turboquant' | 'polarquant'
        self.kq = kq; self.vq = vq if vq is not None else kq
        self.residual = residual

    def _apply(self, x, is_key):
        # x: (b, n_kv_head, seq, d)
        b, h, n, d = x.shape
        if n <= 1:
            return x  # decode token within residual -> keep fp16
        nq = max(0, n - self.residual)
        if nq == 0:
            return x
        head = x[:, :, :nq, :]; tail = x[:, :, nq:, :]
        if self.method == "kivi":
            flat = head.reshape(b * h, nq, d)
            out = torch.stack([(self.kq.quant_K(flat[i]) if is_key else self.vq.quant_V(flat[i]))
                               for i in range(b * h)], 0).reshape(b, h, nq, d)
        else:  # turboquant / polarquant: per-vector
            q = self.kq if is_key else self.vq
            out = q(head)
        return torch.cat([out, tail], dim=2)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        kq = self._apply(key_states, True)
        vq = self._apply(value_states, False)
        return super().update(kq, vq, layer_idx, cache_kwargs)


def make_quant_cache(method, d, bits, dev, gk=128, gv=32, residual=0):
    if method == "kivi":
        q = KiviQuantizer(bits, gk, gv); return QuantCache("kivi", q, residual=residual), q.eff_bits()
    if method == "turboquant":
        q = TurboQuantizer(d, bits, dev); return QuantCache("turboquant", q, residual=0), q.eff_bits()
    if method == "polarquant":
        q = PolarQuantizer(d, bits, dev); return QuantCache("polarquant", q, residual=0), q.eff_bits()
    if method == "quarot":
        q = QuaRotQuantizer(d, bits, dev); return QuantCache("quarot", q, residual=0), q.eff_bits()
    raise ValueError(method)


# =============================== SnapKV / PyramidKV =========================
def _rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def _apply_rope(q, cos, sin):
    # q: (b, h, s, d); cos/sin: (b, s, d)
    cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
    return q * cos + _rotate_half(q) * sin


@torch.no_grad()
def streaming_compress(model, heads, input_ids, budgets, sink=4):
    """StreamingLLM (Xiao et al. 2024): keep `sink` attention-sink tokens at the start +
    the most recent (budget-sink) tokens; evict the middle. No attention scores needed."""
    full = DynamicCache()
    out = model(input_ids, use_cache=True, past_key_values=full)
    cache = out.past_key_values
    n = input_ids.shape[1]
    new = DynamicCache()
    for l in range(heads.L):
        K = cache.layers[l].keys; V = cache.layers[l].values
        bud = max(1, budgets[l]); rec = max(1, bud - sink)
        if n <= bud:
            new.update(K, V, l); continue
        Kc = torch.cat([K[:, :, :sink, :], K[:, :, -rec:, :]], dim=2)
        Vc = torch.cat([V[:, :, :sink, :], V[:, :, -rec:, :]], dim=2)
        new.update(Kc, Vc, l)
    return new, n, out.logits[:, -1, :]


@torch.no_grad()
def snapkv_compress(model, heads, input_ids, budgets, window=32, kernel=5, pool=True):
    """Faithful SnapKV/PyramidKV/H2O prefill compression.
    budgets: list[int] per layer (PyramidKV) or constant (SnapKV/H2O).
    pool=True -> SnapKV/PyramidKV (max-pool clustering); pool=False -> H2O (raw heavy-hitters).
    Returns (DynamicCache compressed, n_tokens, last_logits)."""
    dev = input_ids.device
    L = heads.L
    qcap = {}
    hk = []

    def mk(l):
        def hook(mod, a_, out):
            qcap[l] = out[..., -window:, :].detach()   # (1, window, n_head*hd)
        return hook
    for l in range(L):
        hk.append(heads.layers[l].self_attn.q_proj.register_forward_hook(mk(l)))
    full = DynamicCache()
    out = model(input_ids, use_cache=True, past_key_values=full)
    for h in hk:
        h.remove()
    last_logits = out.logits[:, -1, :]
    cache = out.past_key_values
    n = input_ids.shape[1]
    Hq = heads.H            # query heads
    Hkv = heads.Hkv         # kv heads (GQA)
    hd = heads.hd
    grp = Hq // Hkv
    # rope cos/sin for the window positions
    pos = torch.arange(n, device=dev).unsqueeze(0)
    rotary = model.model.rotary_emb
    cos, sin = rotary(out.logits, pos)            # (1, n, hd)
    cos_w = cos[:, -window:, :]; sin_w = sin[:, -window:, :]

    new = DynamicCache()
    for l in range(L):
        K = cache.layers[l].keys      # (1, Hkv, n, hd) post-RoPE
        V = cache.layers[l].values
        q = qcap[l].view(1, window, Hq, hd).transpose(1, 2)        # (1,Hq,window,hd)
        q = _apply_rope(q, cos_w, sin_w)
        Krep = K.repeat_interleave(grp, dim=1)                     # (1,Hq,n,hd)
        scores = (q @ Krep.transpose(-1, -2)) / math.sqrt(hd)     # (1,Hq,window,n)
        # official SnapKV: causal mask within the observation window (last `window` keys),
        # then softmax in fp32.
        wmask = torch.full((window, window), float("-inf"), device=dev, dtype=scores.dtype).triu(1)
        scores[..., -window:] = scores[..., -window:] + wmask
        scores = scores.float().softmax(dim=-1).to(q.dtype)
        vote = scores[..., :n - window].sum(dim=2)                 # (1,Hq,n-window)
        vote = vote.view(1, Hkv, grp, n - window).sum(dim=2)      # (1,Hkv,n-window)
        pooled = F.avg_pool1d(vote, kernel_size=kernel, padding=kernel // 2, stride=1) if pool else vote
        bud = max(1, budgets[l]) - window
        bud = max(1, min(bud, n - window))
        idx = pooled.topk(bud, dim=-1).indices                    # (1,Hkv,bud)
        idx_e = idx.unsqueeze(-1).expand(-1, -1, -1, hd)
        Kp = K[:, :, :n - window, :].gather(2, idx_e)
        Vp = V[:, :, :n - window, :].gather(2, idx_e)
        Kc = torch.cat([Kp, K[:, :, n - window:, :]], dim=2)
        Vc = torch.cat([Vp, V[:, :, n - window:, :]], dim=2)
        new.update(Kc, Vc, l)
    return new, n, last_logits


def pyramid_budgets(C, L, beta=20):
    """KVCache-Factory pyramid: linear from max(low layers) to min(high layers), mean=C."""
    min_n = max(1, int(C // beta)); max_n = 2 * C - min_n
    if L == 1:
        return [C]
    step = (max_n - min_n) / (L - 1)
    return [int(round(max_n - i * step)) for i in range(L)]


@torch.no_grad()
def evict_generate(model, tok, heads, input_ids, gen_tokens, method, budget, window=32, kernel=5, sink=4):
    """SnapKV/PyramidKV/H2O/StreamingLLM: compress prefill then greedy-decode from compressed cache."""
    L = heads.L
    budgets = pyramid_budgets(budget, L) if method == "pyramidkv" else [budget] * L
    if method == "streamingllm":
        cache, n, last_logits = streaming_compress(model, heads, input_ids, budgets, sink=sink)
    elif method == "h2o":
        cache, n, last_logits = snapkv_compress(model, heads, input_ids, budgets, window, kernel, pool=False)
    else:  # snapkv, pyramidkv
        cache, n, last_logits = snapkv_compress(model, heads, input_ids, budgets, window, kernel, pool=True)
    dev = input_ids.device
    eos_ids = set()
    if tok.eos_token_id is not None:
        eos_ids.add(int(tok.eos_token_id))
    gc = getattr(model, "generation_config", None)
    if gc is not None and getattr(gc, "eos_token_id", None) is not None:
        e = gc.eos_token_id
        eos_ids.update(int(x) for x in (e if isinstance(e, (list, tuple)) else [e]))
    nxt = last_logits.argmax(-1, keepdim=True)
    gen = []
    # decode from the compressed cache: decouple the cache write index (compressed length)
    # from the RoPE position (true sequence position) so the new query's RoPE matches the
    # retained keys' original (prefill) RoPE; stop at EOS like model.generate().
    cache_len = cache.get_seq_length()
    cur_pos = n
    for _ in range(gen_tokens):
        if int(nxt.item()) in eos_ids:
            break
        gen.append(nxt)
        cp = torch.tensor([cache_len], device=dev)
        pid = torch.tensor([[cur_pos]], device=dev)
        out = model(nxt, past_key_values=cache, use_cache=True, cache_position=cp, position_ids=pid)
        cache = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        cache_len += 1; cur_pos += 1
    ids = torch.cat(gen, dim=1)[0] if gen else torch.empty(0, dtype=torch.long, device=dev)
    # effective token fraction vs full context (fp16 cache): seq/budget compression
    avg_bud = sum(budgets) / L
    return tok.decode(ids, skip_special_tokens=True), avg_bud


# =============================== KIVI (hook-style, pre-RoPE) ================
# Standalone KIVI fake-quant on k_proj/v_proj outputs (used by the NIAH --kivi-configs
# hook path, distinct from the post-RoPE QuantCache KIVI above). Kept for parity with the
# original kivi_baseline.py reference reimplementation.
def quant_groups(x, bits, group):
    *lead, n = x.shape
    g = group if (group and group > 0 and n % group == 0) else n
    xr = x.reshape(*lead, n // g, g)
    lo = xr.amin(-1, keepdim=True); hi = xr.amax(-1, keepdim=True)
    qmax = (1 << bits) - 1
    scale = (hi - lo).clamp_min(1e-8) / qmax
    q = torch.round((xr - lo) / scale).clamp_(0, qmax)
    return (q * scale + lo).reshape(*lead, n)


def fake_quant_K(K, bits, gk, resid):
    """K: (n, d). per-channel, grouped over tokens; last `resid` tokens kept fp16."""
    n, d = K.shape
    out = K.clone(); n_q = max(0, n - resid)
    if n_q == 0:
        return out
    out[:n_q] = quant_groups(K[:n_q].t(), bits, gk).t()
    return out


def fake_quant_V(V, bits, gv, resid):
    """V: (n, d). per-token, grouped over channels; last `resid` tokens kept fp16."""
    n, d = V.shape
    out = V.clone(); n_q = max(0, n - resid)
    if n_q == 0:
        return out
    out[:n_q] = quant_groups(V[:n_q], bits, gv)
    return out


def effective_bits(bits, group):
    return bits + 32.0 / (group if group and group > 0 else 1e9)
