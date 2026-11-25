"""SPECTRA-Flash for HuggingFace Transformers: `spectra.flash.enable(model, params)`.

The deployed attention of layer l in block g0 (n layers share one record per token; n = 1 or 2):
  prefill (S > 1, empty cache): exact causal attention over the whole prompt. Then the exact window of every layer
          (sink + the last `recent` tokens) and the block's record entries of every prompt position >= sink.
  decode  (S = 1): the absorbed query attends the record [sink, t - recent), the exact query attends the window, and
          the two partial softmaxes are merged by their LSE (one softmax over both). The current token's record entry
          is written at its final slot every step and is read once the token leaves the window.
Backends:
  "fast"  : FlashMLA (vLLM build) + the fused Triton kernels, the kernels of the deployed engine (Hopper).
  "torch" : the same math in PyTorch (any CUDA GPU), the reference.
fp8 (default, the deployed config) stores the record and the FlashMLA query in fp8 with the params' static scales.
Limits (v0.2): one CUDA device, bfloat16, unpadded batches (batch 1 or equal-length prompts), one prompt per cache
(no multi-turn continuation, beam search or cache cropping).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, CacheLayerMixin

from . import deploy
from .ops import F8_MAX, rope_pairs

BF, F8 = torch.bfloat16, torch.float8_e4m3fn
MPAGE, TPAGE = 64, 16                 # FlashMLA page (record), FA-style page (exact window)


def fast_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    try:
        from vllm.v1.attention.ops import flashmla
    except Exception as e:  # noqa: BLE001
        return False, f"vLLM FlashMLA not importable ({type(e).__name__})"
    ok, why = flashmla.is_flashmla_dense_supported()
    return bool(ok), why or ""


# ----------------------------------------------------------------------------------------------- runtime
class FlashRuntime:
    """Folded weights and settings, shared by every cache of one model."""

    def __init__(self, model, params: str, backend: str = "auto", fp8: bool | None = None, recent: int | None = None):
        cfg = model.config
        self.model = model
        self.dev = next(model.parameters()).device
        self.L, self.H, self.HKV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.DM, self.G = cfg.hidden_size, self.H // self.HKV
        self.n, self.blocks, qs, self.meta, self.params_dir = deploy.load(params, self.dev, self.HKV, self.D)
        dep = deploy.deploy_settings(self.meta, recent)
        self.sink, self.recent = dep["sink"], dep["recent"]
        self.fp8 = dep["fp8"] if fp8 is None else fp8
        if self.L % self.n:
            raise ValueError(f"{self.L} layers are not a multiple of the block size {self.n}")
        if set(self.blocks) != set(range(0, self.L, self.n)):
            raise ValueError("the params do not cover every block of this model")
        deploy.check_rope(self.blocks, model.model.rotary_emb.inv_freq)
        self.qscale = torch.tensor([qs[l] for l in range(self.L)], device=self.dev, dtype=torch.float32)
        self.sfx = "8" if self.fp8 else "16"
        self.R = self.blocks[0]["R"]; self.r = self.blocks[0]["r"]
        self.DK = self.R + self.r
        self.QR0 = self.H * self.D + 2 * self.HKV * self.D        # rope query rows start here in the fused GEMM
        self.E0 = self.QR0 + self.H * self.r                       # partial record entry rows start here
        self.SPS = -(-(self.sink + self.recent + 1) // TPAGE) * TPAGE   # window slots per sequence
        self.scale = self.D ** -0.5
        if backend == "auto":
            backend = "fast" if fast_available()[0] else "torch"
        if backend == "fast":
            ok, why = fast_available()
            if not ok:
                raise RuntimeError(f"backend 'fast' needs vLLM's FlashMLA on Hopper: {why}")
            from vllm.v1.attention.ops import flashmla
            from vllm.vllm_flash_attn import flash_attn_varlen_func
            from . import kernels
            self.MLA, self.FK, self.FA = flashmla, kernels, flash_attn_varlen_func
        elif backend != "torch":
            raise ValueError(f"backend must be auto | fast | torch, got {backend!r}")
        self.backend = backend
        self.one = torch.ones(1, device=self.dev, dtype=torch.float32)
        self._fuse()

    def _fuse(self):
        """Per layer one GEMM weight q | k | v | rope query | this layer's half of Wdr; q/k/v_proj become row views
        of it, so no weight is stored twice."""
        self.W = []
        for l, layer in enumerate(self.model.model.layers):
            at = layer.self_attn
            g0 = l - l % self.n; li = l - g0; blk = self.blocks[g0]
            if getattr(at, "_flash_fused", None) is None:
                base = [at.q_proj.weight.data, at.k_proj.weight.data, at.v_proj.weight.data]
                if any(p.bias is not None for p in (at.q_proj, at.k_proj, at.v_proj)):
                    raise NotImplementedError("q/k/v biases are not supported (Qwen2-style models)")
            else:
                base = [at._flash_fused[:self.QR0]]
            part = blk["Wdr" + self.sfx][:, li * self.DM:(li + 1) * self.DM]
            W = torch.cat(base + [blk["WQR"][li], part], 0).contiguous()
            del base
            hq, hk = self.H * self.D, self.HKV * self.D
            at.q_proj.weight = torch.nn.Parameter(W[:hq], requires_grad=False)
            at.k_proj.weight = torch.nn.Parameter(W[hq:hq + hk], requires_grad=False)
            at.v_proj.weight = torch.nn.Parameter(W[hq + hk:hq + 2 * hk], requires_grad=False)
            at._flash_fused = W
            self.W.append(W)
        torch.cuda.empty_cache()

    def new_cache(self) -> "FlashCache":
        return FlashCache(self)


# ----------------------------------------------------------------------------------------------- cache
class FlashLayer(CacheLayerMixin):
    """Per-layer bookkeeping HuggingFace sees (lengths for masks / positions); the data live in FlashCache."""
    is_sliding = False
    is_compileable = False
    is_croppable = False
    supports_early_init = False

    def __init__(self):
        super().__init__()
        self.seen = 0

    def lazy_initialization(self, key_states, value_states):
        pass

    def update(self, key_states, value_states, *args, **kwargs):
        raise RuntimeError("SPECTRA-Flash cache layers are written by the patched attention, not by Cache.update")

    def get_mask_sizes(self, query_length):
        return self.seen + query_length, 0

    def get_seq_length(self):
        return self.seen

    def get_max_length(self):
        return -1

    def reset(self):
        raise NotImplementedError("create a new cache (model.spectra_flash.new_cache()) instead of resetting one")

    def _unsupported(self, *a, **k):
        if self.seen:
            raise NotImplementedError("SPECTRA-Flash v0.2 does not support beam search / cache reordering")

    reorder_cache = batch_repeat_interleave = batch_select_indices = _unsupported

    def crop(self, *a, **k):
        raise NotImplementedError("SPECTRA-Flash v0.2 does not support cache cropping (assisted decoding)")


class FlashCache(Cache):
    """Record pages (one per block, FlashMLA layout) + exact windows (one per layer), B sequences each laid out
    contiguously: record slot q - sink of sequence b is row b * cap + (q - sink); window slot s is row b * SPS + s."""

    def __init__(self, rt: FlashRuntime):
        super().__init__(layers=[FlashLayer() for _ in range(rt.L)])
        self.rt = rt
        self.B = None
        self.cap = 0                    # record entries per sequence
        self.rc = None                  # (L / n, B * cap / MPAGE, MPAGE, 1, DK)
        self.tk = self.tv = None        # (L, B * SPS / TPAGE, TPAGE, HKV, D)
        self.step = None                # per decode step, shared by the layers
        self.h0 = None; self.e0 = None  # block's first-layer input (prefill) / partial entry (decode)

    # -- storage
    def _alloc(self, B, need):
        rt = self.rt
        cap = -(-max(need, MPAGE) // MPAGE) * MPAGE
        dt = F8 if rt.fp8 else BF
        rc = torch.zeros(rt.L // rt.n, B * cap // MPAGE, MPAGE, 1, rt.DK, dtype=dt, device=rt.dev)
        if self.rc is not None:                                    # grow: copy each sequence's entries
            old = self.rc.view(rt.L // rt.n, self.B, self.cap, rt.DK)
            rc.view(rt.L // rt.n, B, cap, rt.DK)[:, :, :self.cap] = old
        self.rc, self.cap = rc, cap
        self.bt = (torch.arange(B, device=rt.dev, dtype=torch.int32)[:, None] * (cap // MPAGE)
                   + torch.arange(cap // MPAGE, device=rt.dev, dtype=torch.int32)[None])

    def _init(self, B, S):
        rt = self.rt
        self.B = B
        self._alloc(B, S + 1024)
        self.tk = torch.zeros(rt.L, B * rt.SPS // TPAGE, TPAGE, rt.HKV, rt.D, dtype=BF, device=rt.dev)
        self.tv = torch.zeros_like(self.tk)

    def wslot(self, q: int) -> int:
        """Window slot of position q: sink positions first, then a ring of recent + 1 at sink + q mod (recent + 1),
        the layout of the engine (engine.Model.wslot; the slot order is tail_merge's summation order)."""
        s = self.rt.sink
        return q if q < s else s + q % (self.rt.recent + 1)

    def begin_step(self, p: int):
        """Decode of position p (same for the whole batch): lengths, slots and FlashMLA metadata."""
        rt, B = self.rt, self.B
        if p - rt.sink >= self.cap:
            self._alloc(B, 2 * self.cap)
        dev = rt.dev
        ar = torch.arange(B, device=dev)
        rlen = max(0, p - rt.recent - rt.sink)
        st = dict(p=p, rlen=rlen, pos=torch.full((B,), p, device=dev, dtype=torch.long),
                  wbase=ar * rt.SPS, tslot=ar * rt.SPS + self.wslot(p),
                  rwslot=ar * self.cap + (p - rt.sink) if p >= rt.sink else None,
                  rlen_t=torch.full((B,), rlen, device=dev, dtype=torch.int32))
        if rt.backend == "fast" and rlen > 0:
            md = rt.MLA.get_mla_metadata_dense_fp8 if rt.fp8 else rt.MLA.get_mla_metadata
            st["md"], st["ns"] = md(st["rlen_t"], rt.H, 1)
        self.step = st

    def memory(self) -> dict:
        """Bytes held (record pages + exact windows) and the dense bf16 K/V equivalent."""
        if self.B is None:
            return {}
        rt = self.rt
        rec = self.rc.numel() * self.rc.element_size(); win = 2 * self.tk.numel() * self.tk.element_size()
        seen = self.layers[0].seen
        dense = self.B * seen * rt.L * 2 * rt.HKV * rt.D * 2
        return dict(tokens=seen, batch=self.B, record_allocated=rec, windows=win, dense_equivalent=dense,
                    ratio_vs_dense=dense / max(1, rec + win))


# ----------------------------------------------------------------------------------------------- attention
def window_valid(p: int, sink: int, ring: int, device) -> torch.Tensor:
    """(sink + ring,) bool: ring slot j holds a processed position at decode position p (slot j's first position is
    j for j < sink, else the smallest q >= sink with q mod ring == j - sink). All true once p >= sink + ring - 1."""
    j = torch.arange(sink + ring, device=device)
    q0 = j - sink
    q0 = torch.where(q0 < sink, q0 + ring, q0)
    return torch.where(j < sink, j <= p, q0 <= p)


def _has_padding(mask):
    if mask is None or not torch.is_tensor(mask) or mask.dim() != 4:
        return False
    last = mask[:, :, -1, :]
    return bool((~last).any()) if mask.dtype == torch.bool else bool((last < -1).any())


def _entries(rt, blk, e, pos):
    """Record entries from H @ Wdr^T rows e (N, R + r) at positions pos: [c | RoPE(s)], fp32 (fp8 range clamped)."""
    ent = torch.cat([e[:, :rt.R].float(), rope_pairs(e[:, rt.R:], pos, blk["theta"])], -1)
    return ent.clamp(-F8_MAX, F8_MAX) if rt.fp8 else ent


def _prefill(rt, cache, l, li, blk, at, h, cos, sin, apply_rope):
    B, S, _ = h.shape
    H, HKV, D = rt.H, rt.HKV, rt.D
    qkv = F.linear(h, rt.W[l][:rt.QR0])
    q = qkv[..., :H * D].view(B, S, H, D).transpose(1, 2)
    k = qkv[..., H * D:(H + HKV) * D].view(B, S, HKV, D).transpose(1, 2)
    v = qkv[..., (H + HKV) * D:].view(B, S, HKV, D).transpose(1, 2)
    q, k = apply_rope(q, k, cos, sin)
    if rt.backend == "fast":                                            # FA3, as the deployed engine
        cu = torch.arange(B + 1, device=h.device, dtype=torch.int32) * S
        o = rt.FA(q.transpose(1, 2).reshape(B * S, H, D), k.transpose(1, 2).reshape(B * S, HKV, D),
                  v.transpose(1, 2).reshape(B * S, HKV, D), max_seqlen_q=S, cu_seqlens_q=cu, max_seqlen_k=S,
                  cu_seqlens_k=cu, causal=True, softmax_scale=rt.scale, fa_version=3).view(B, S, H * D)
    else:                                                               # kv heads expanded: SDPA's flash kernel
        o = F.scaled_dot_product_attention(q, k.repeat_interleave(rt.G, 1), v.repeat_interleave(rt.G, 1),
                                           is_causal=True, scale=rt.scale).transpose(1, 2).reshape(B, S, H * D)
    # exact window: [0, sink) and [max(sink, S - recent), S)
    keep = list(range(min(S, rt.sink))) + list(range(max(rt.sink, S - rt.recent), S))
    idx = torch.tensor(keep, device=h.device)
    slots = torch.tensor([cache.wslot(q_) for q_ in keep], device=h.device)
    tk = cache.tk[l].view(B, rt.SPS, HKV, D); tv = cache.tv[l].view(B, rt.SPS, HKV, D)
    tk[:, slots] = k.transpose(1, 2)[:, idx]; tv[:, slots] = v.transpose(1, 2)[:, idx]
    # record entries of positions [sink, S), written at the block's last layer from both layers' inputs
    if li == 0:
        cache.h0 = h
    if li == rt.n - 1 and S > rt.sink:
        Hh = torch.cat([cache.h0, h], -1)[:, rt.sink:] if rt.n == 2 else h[:, rt.sink:]
        cache.h0 = None
        pos = torch.arange(rt.sink, S, device=h.device)
        rec = cache.rc[l // rt.n].view(B, cache.cap, rt.DK)
        for b in range(B):
            e = F.linear(Hh[b], blk["Wdr" + rt.sfx])
            rec[b, :S - rt.sink] = _entries(rt, blk, e, pos).to(rec.dtype)
    return o


def _decode_fast(rt, cache, l, li, blk, h, cos, sin, apply_rope):
    st = cache.step; B = h.shape[0]
    H, HKV, D = rt.H, rt.HKV, rt.D
    qkvr = F.linear(h, rt.W[l])                                         # (B, QR0 + H r + R + r)
    qm = torch.empty(B, H, rt.DK, device=h.device, dtype=F8 if rt.fp8 else BF)
    rt.FK.q_prep(qkvr, blk["WuK" + rt.sfx][li], st["pos"], blk["theta"], blk["ss"], rt.qscale[l:l + 1], qm,
                 rt.fp8, H=H, G=rt.G, D=D, QR0=rt.QR0)
    q = qkvr[:, :H * D].view(B, H, 1, D); k = qkvr[:, H * D:(H + HKV) * D].view(B, HKV, 1, D)
    v = qkvr[:, (H + HKV) * D:rt.QR0].view(B, HKV, D)
    q, k = apply_rope(q, k, cos, sin)
    tkf = cache.tk[l].view(-1, HKV, D); tvf = cache.tv[l].view(-1, HKV, D)
    tkf[st["tslot"]] = k[:, :, 0]; tvf[st["tslot"]] = v
    if st["rlen"] > 0:
        bi = l // rt.n
        kw = dict(softmax_scale=rt.scale, causal=True)
        if rt.fp8:
            o_l, lse = rt.MLA.flash_mla_with_kvcache_fp8(qm.view(B, 1, H, rt.DK), cache.rc[bi], cache.bt, st["rlen_t"],
                                                         rt.R, st["md"], st["ns"], descale_q=rt.qscale[l:l + 1],
                                                         descale_k=rt.one, **kw)
        else:
            o_l, lse = rt.MLA.flash_mla_with_kvcache(qm.view(B, 1, H, rt.DK), cache.rc[bi], cache.bt, st["rlen_t"],
                                                     rt.R, st["md"], st["ns"], **kw)
    else:
        o_l = torch.zeros(B, 1, H, rt.R, device=h.device, dtype=BF)
        lse = torch.full((B, H, 1), float("-inf"), device=h.device, dtype=torch.float32)
    e = qkvr[:, rt.E0:]
    if li == 0 and rt.n > 1:
        cache.e0 = e
    elif st["rwslot"] is not None:
        rt.FK.rec_write(cache.e0 if rt.n > 1 else e, e if rt.n > 1 else None, st["pos"], blk["theta"],
                        cache.rc[l // rt.n].view(-1, rt.DK), st["rwslot"], rt.fp8, rt.R)
    out = torch.empty(B, H * D, device=h.device, dtype=BF)
    qf = q.reshape(B, H * D).contiguous()
    rt.FK.tail_merge(qf, qf.stride(0), cache.tk[l].view(-1, HKV, D), cache.tv[l].view(-1, HKV, D), st["wbase"],
                     st["pos"], st["rlen_t"], o_l, lse, blk["WuV" + rt.sfx][li], out, rt.scale, rt.sink, rt.recent + 1,
                     H=H, G=rt.G, D=D)
    return out


def _decode_torch(rt, cache, l, li, blk, h, cos, sin, apply_rope):
    """Reference: the fast path's math, one joint softmax, fp32 scores; fp8 values round-tripped as stored."""
    st = cache.step; B = h.shape[0]
    H, HKV, D, G = rt.H, rt.HKV, rt.D, rt.G
    qkvr = F.linear(h, rt.W[l])
    q_un = qkvr[:, :H * D].float().view(B, HKV, G, D)
    qa = torch.einsum("bkgd,kdr->bkgr", q_un, blk["WuK" + rt.sfx][li].float()).reshape(B, H, rt.R)
    qr = rope_pairs(qkvr[:, rt.QR0:rt.E0].view(B, H, rt.r), st["pos"][:1], blk["theta"])
    if rt.fp8:
        qr = qr * blk["ss"]
    qm = torch.cat([qa, qr], -1)
    if rt.fp8:                                                          # fp8 query with the static scale
        qs = rt.qscale[l]
        qm = (qm / qs).clamp(-F8_MAX, F8_MAX).to(F8).float() * qs
    q = qkvr[:, :H * D].view(B, H, 1, D); k = qkvr[:, H * D:(H + HKV) * D].view(B, HKV, 1, D)
    v = qkvr[:, (H + HKV) * D:rt.QR0].view(B, HKV, D)
    q, k = apply_rope(q, k, cos, sin)
    tkf = cache.tk[l].view(-1, HKV, D); tvf = cache.tv[l].view(-1, HKV, D)
    tkf[st["tslot"]] = k[:, :, 0]; tvf[st["tslot"]] = v
    n = rt.sink + rt.recent + 1                                         # the ring; slots not yet written are masked
    Kw = cache.tk[l].view(B, rt.SPS, HKV, D)[:, :n].float(); Vw = cache.tv[l].view(B, rt.SPS, HKV, D)[:, :n].float()
    s_e = torch.einsum("bkgd,btkd->bkgt", q.float().view(B, HKV, G, D), Kw).reshape(B, H, n) * rt.scale
    s_e = s_e.masked_fill(~window_valid(st["p"], rt.sink, rt.recent + 1, h.device), float("-inf"))
    rl = st["rlen"]
    if rl > 0:
        rec = cache.rc[l // rt.n].view(B, cache.cap, rt.DK)[:, :rl].float()
        s_r = torch.einsum("bhc,btc->bht", qm, rec) * rt.scale
    else:
        s_r = s_e.new_zeros(B, H, 0); rec = None
    P = torch.softmax(torch.cat([s_r, s_e], -1), -1)
    o = torch.einsum("bkgt,btkd->bkgd", P[..., rl:].view(B, HKV, G, n), Vw).reshape(B, H, D)
    if rl > 0:
        ol = torch.einsum("bht,btr->bhr", P[..., :rl], rec[..., :rt.R]).view(B, HKV, G, rt.R)
        o = o + torch.einsum("bkgr,krd->bkgd", ol, blk["WuV" + rt.sfx][li].float()).reshape(B, H, D)
    e = qkvr[:, rt.E0:]
    if li == 0 and rt.n > 1:
        cache.e0 = e
    elif st["rwslot"] is not None:
        ent = _entries(rt, blk, (cache.e0.float() + e.float()) if rt.n > 1 else e.float(), st["pos"])
        cache.rc[l // rt.n].view(-1, rt.DK)[st["rwslot"]] = ent.to(cache.rc.dtype)
    return o.reshape(B, H * D).to(BF)


def make_forward(rt, attn, l, orig_forward, apply_rope):
    g0 = l - l % rt.n; li = l - g0; blk = rt.blocks[g0]
    decode = _decode_fast if rt.backend == "fast" else _decode_torch

    def forward(hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        cache = past_key_values if past_key_values is not None else kwargs.get("past_key_value")
        if not isinstance(cache, FlashCache):
            return orig_forward(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask,
                                past_key_values=past_key_values, **kwargs)
        B, S, _ = hidden_states.shape
        seen = cache.layers[l].seen
        cos, sin = position_embeddings
        if seen == 0:
            if S > 1 and _has_padding(attention_mask):
                raise NotImplementedError("SPECTRA-Flash v0.2 needs unpadded batches (equal-length prompts or batch 1)")
            if l == 0:
                cache._init(B, S)
            o = _prefill(rt, cache, l, li, blk, attn, hidden_states, cos, sin, apply_rope)
        elif S == 1:
            if cache.B != B:
                raise ValueError(f"batch size changed from {cache.B} to {B}; use a new cache")
            if l == 0:
                cache.begin_step(seen)
            o = decode(rt, cache, l, li, blk, hidden_states[:, 0], cos, sin, apply_rope)[:, None]
        else:
            raise NotImplementedError("SPECTRA-Flash v0.2: one prompt per cache, then one token per step "
                                      "(no multi-turn continuation / chunked prefill in the HF path)")
        cache.layers[l].seen += S
        return attn.o_proj(o.to(hidden_states.dtype)), None

    return forward


# ----------------------------------------------------------------------------------------------- public API
def enable(model, params: str, backend: str = "auto", fp8: bool | None = None, recent: int | None = None, log=print):
    """Turn on SPECTRA-Flash for a HuggingFace Llama model (bfloat16, one CUDA device).

    params : a SPECTRA-Flash params dir or HF Hub repo id (params.safetensors + flash_config.json), converted
             from this base model. backend: "fast" (FlashMLA + Triton, Hopper), "torch" (reference) or "auto".
    After this call `model.generate(...)` uses the Flash cache; `model.spectra_flash.new_cache()` gives one
    for manual forward loops."""
    if getattr(model.config, "model_type", "") != "llama":
        raise NotImplementedError(f"SPECTRA-Flash v0.2 supports Llama models, got {model.config.model_type!r}")
    if next(model.parameters()).dtype != BF:
        log("[spectra-flash] casting the model to bfloat16 (the deployed dtype)")
        model.to(BF)
    model.eval()
    rt = FlashRuntime(model, params, backend=backend, fp8=fp8, recent=recent)
    from importlib import import_module
    apply_rope = getattr(import_module(type(model.model.layers[0].self_attn).__module__), "apply_rotary_pos_emb")
    for l, layer in enumerate(model.model.layers):
        at = layer.self_attn
        if not hasattr(at, "_spectra_orig_forward"):
            at._spectra_orig_forward = at.forward
        at.forward = make_forward(rt, at, l, at._spectra_orig_forward, apply_rope)
    model.spectra_flash = rt
    if not hasattr(model, "_spectra_orig_generate"):
        model._spectra_orig_generate = model.generate

        def generate(*args, **kwargs):
            if kwargs.get("past_key_values") is None and kwargs.get("use_cache", True) and \
                    getattr(model, "spectra_flash", None) is not None:
                kwargs["past_key_values"] = model.spectra_flash.new_cache()
            return model._spectra_orig_generate(*args, **kwargs)
        model.generate = generate
    ratio = (2 * rt.n * rt.HKV * rt.D * 2) / (rt.DK * (1 if rt.fp8 else 2))
    log(f"[spectra-flash] on: {rt.L // rt.n} blocks of {rt.n} layers, record {rt.R}+{rt.r} "
        f"{'fp8' if rt.fp8 else 'bf16'} ({ratio:.1f}x on compressed tokens), sink {rt.sink} + recent {rt.recent} exact, "
        f"backend {rt.backend}, params {rt.params_dir}")
    return model


def disable(model):
    """Restore the original attention and generate (the fused weights stay; their values are unchanged)."""
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "_spectra_orig_forward"):
            layer.self_attn.forward = layer.self_attn._spectra_orig_forward
    if hasattr(model, "_spectra_orig_generate"):
        model.generate = model._spectra_orig_generate
        del model._spectra_orig_generate
    model.spectra_flash = None
    return model
