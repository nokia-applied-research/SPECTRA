"""Minimal decode engine for SPECTRA-Flash: the end-to-end decode-speed benchmark and the on-policy sampler.

A Llama decode engine on vLLM's kernels (needs vLLM, Hopper): real weights (bf16), one CUDA graph per decode step
covering all layers, the final norm, lm_head and argmax. Dense and Flash share every op except attention:
  dense   : FA3 paged decode (page 16) over the full exact K/V cache (bf16).
  dense8  : the same with an fp8 K/V cache and fp8 query (benchmark only; the strongest dense baseline).
  flash16 : per block of n layers (n = 1 for SPECTRA-Flash) one FlashMLA record (512 latent + 64 RoPE, bf16, page 64) + an exact window
            (sink + recent) per layer; the two partial softmaxes merge by LSE (Triton tail_merge).
  flash8  : the deployed config: the record and the FlashMLA query in fp8 with the params' static scales.
Steady state: every step writes the current token's record entry at its final slot; FlashMLA reads the first
rlen = t - recent - sink entries, so an entry is used once its token leaves the window. The write is inside the
timed step. Window slots: sink positions first, then a ring of recent + 1 (the HF path and the vLLM plugin use the
same layout).
"""
from __future__ import annotations

import glob
import math
import os

import torch
import torch.nn.functional as F

from . import deploy
from . import kernels as FK

BF, F8 = torch.bfloat16, torch.float8_e4m3fn
PAGE, MPAGE = 16, 64
GiB = 2 ** 30
_K = {}


def _vllm():
    if not _K:
        from vllm import _custom_ops
        from vllm.v1.attention.ops import flashmla
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        _K.update(FA=flash_attn_varlen_func, MLA=flashmla, OPS=_custom_ops)
    return _K["FA"], _K["MLA"], _K["OPS"]


def rope(x, cos, sin):                         # x (N, heads, D); cos/sin (N, D) [HF rotate_half]
    h = x.shape[-1] // 2
    return x * cos[:, None] + torch.cat([-x[..., h:], x[..., :h]], -1) * sin[:, None]


def rope_inv_freq(cfg, dev) -> torch.Tensor:
    """RoPE inverse frequencies computed on `dev`, op for op as vLLM builds its rotary table. On CUDA the resulting
    table is bitwise equal to vLLM's cos_sin_cache and to the reference engine. HF computes the frequencies on the CPU,
    which differs by 1 ulp in a few frequencies (1e-3 rad at 32K): enough to flip greedy near-ties."""
    rp = dict(getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {})
    base = rp.get("rope_theta", getattr(cfg, "rope_theta", 10000.0))
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    inv = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.int64).to(dev).float() / D))
    kind = rp.get("rope_type", rp.get("type", "default"))
    if kind == "default":
        return inv
    if kind != "llama3":
        raise NotImplementedError(f"rope_type {kind!r}")
    factor, lf, hf = rp["factor"], rp["low_freq_factor"], rp["high_freq_factor"]
    orig = rp["original_max_position_embeddings"]
    wl = 2 * math.pi / inv
    out = torch.where(wl > orig / lf, inv / factor, inv)
    sm = (orig / wl - lf) / (hf - lf)
    med = ~(wl < orig / hf) & ~(wl > orig / lf)
    return torch.where(med, (1 - sm) * out / factor + sm * out, out)


def rope_pairs(x, pos, theta):                 # x (N, heads, r) as r/2 adjacent pairs; returns fp32
    ang = pos.float()[:, None, None] * theta[None, None, :]
    c, s = ang.cos(), ang.sin()
    x2 = x.float().unflatten(-1, (-1, 2))
    x0, x1 = x2[..., 0], x2[..., 1]
    return torch.stack([x0 * c - x1 * s, x1 * c + x0 * s], -1).flatten(-2)


# ----------------------------------------------------------------------------------------------- weights
class Model:
    """Base Llama weights (from a local dir or the HF Hub) + optionally the folded Flash params."""

    def __init__(self, base: str, params: str | None = None, dev="cuda", recent: int | None = None, max_pos=136 * 1024):
        from safetensors.torch import load_file
        from transformers import AutoConfig
        if params is not None:
            params = deploy.resolve(params)              # fail before the base weights are loaded
        path = base if os.path.isdir(base) else __import__("huggingface_hub").snapshot_download(
            base, allow_patterns=["*.safetensors", "*.json"])
        cfg = AutoConfig.from_pretrained(path)
        if cfg.model_type != "llama":
            raise NotImplementedError(f"engine supports Llama models, got {cfg.model_type!r}")
        self.cfg = cfg
        self.H, self.HKV, self.NL, self.DM = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.num_hidden_layers, cfg.hidden_size
        self.D = getattr(cfg, "head_dim", None) or self.DM // self.H
        self.G, self.EPS, self.SC = self.H // self.HKV, cfg.rms_norm_eps, self.D ** -0.5
        self.NQKV = (self.H + 2 * self.HKV) * self.D
        dev = torch.device(dev)
        sd = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            sd.update(load_file(f, device=str(dev)))
        self.emb = sd.pop("model.embed_tokens.weight"); self.norm = sd.pop("model.norm.weight")
        self.head = sd.pop("lm_head.weight") if "lm_head.weight" in sd else self.emb
        self.inv = rope_inv_freq(cfg, dev)
        ang = torch.arange(max_pos, device=dev).float()[:, None] * self.inv[None, :]
        self.cos_sin = torch.cat([ang.cos(), ang.sin()], -1).to(BF)                  # vLLM rotary table
        self.flash = params is not None
        if self.flash:
            self.n, self.blocks, qs, self.meta, self.params_dir = deploy.load(params, dev, self.HKV, self.D)
            dep = deploy.deploy_settings(self.meta, recent)
            self.sink, self.recent = dep["sink"], dep["recent"]
            deploy.check_rope(self.blocks, self.inv)
            self.qscale = torch.tensor([qs[l] for l in range(self.NL)], device=dev, dtype=torch.float32)
            self.R, self.r = self.blocks[0]["R"], self.blocks[0]["r"]
            self.DK = self.R + self.r
        else:
            self.n, self.sink, self.recent = 2, 4, recent or 123
        self.L = []
        for i in range(self.NL):
            p = f"model.layers.{i}."
            g = lambda k: sd.pop(p + k)
            lw = dict(ln1=g("input_layernorm.weight"), ln2=g("post_attention_layernorm.weight"),
                      wo=g("self_attn.o_proj.weight"), wd=g("mlp.down_proj.weight"))
            lw["wgu"] = torch.cat([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")], 0)
            base_qkv = torch.cat([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"),
                                  g("self_attn.v_proj.weight")], 0)
            if not self.flash:
                lw["wqkv16"] = base_qkv
            else:                  # q | k | v | rope query | this layer's half of Wdr (one GEMM per layer)
                g0 = i - i % self.n; li = i - g0; blk = self.blocks[g0]
                for sfx in ("16", "8"):
                    part = blk["Wdr" + sfx][:, li * self.DM:(li + 1) * self.DM]
                    lw["wqkv" + sfx] = torch.cat([base_qkv, blk["WQR"][li], part], 0)
            del base_qkv
            self.L.append(lw)
        if any("layers" in k for k in sd):
            raise ValueError("unused layer weights: not a plain Llama checkpoint")
        del sd
        torch.cuda.empty_cache()

    @property
    def SPS(self):                                # window slots per sequence
        return -(-(self.sink + self.recent + 1) // PAGE) * PAGE

    def wslot(self, q: int) -> int:
        """Window slot of position q: sink positions first, then a ring of recent + 1 slots at sink + q mod (recent + 1).
        The layout of the reference engine behind the reported numbers: the slot order is the summation order inside
        tail_merge, so another (equally valid) ring offset changes greedy tokens at near-ties."""
        return q if q < self.sink else self.sink + q % (self.recent + 1)


# ----------------------------------------------------------------------------------------------- caches
class Cache:
    def __init__(self, M: Model, variant: str, B: int, Tcap: int):
        dev = M.emb.device
        self.variant, self.B, self.Tcap = variant, B, Tcap
        ar = lambda n_: torch.arange(B * n_, device=dev, dtype=torch.int32).view(B, n_)
        if variant.startswith("dense"):
            self.npb = math.ceil(Tcap / PAGE)
            self.kc = torch.empty(M.NL, B * self.npb, PAGE, M.HKV, M.D, dtype=BF if variant == "dense" else F8, device=dev)
            self.vc = torch.empty_like(self.kc)
            self.bt = ar(self.npb)
        else:
            self.npm = math.ceil(Tcap / MPAGE); self.npt = M.SPS // PAGE
            self.rc = torch.empty(M.NL // M.n, B * self.npm, MPAGE, 1, M.DK, dtype=BF if variant == "flash16" else F8,
                                  device=dev)
            self.rbt = ar(self.npm)
            self.tkc = torch.empty(M.NL, B * self.npt, PAGE, M.HKV, M.D, dtype=BF, device=dev)
            self.tvc = torch.empty_like(self.tkc)

    @staticmethod
    def nbytes(M: Model, variant: str, B: int, Tcap: int) -> int:
        if variant.startswith("dense"):
            return 2 * M.NL * B * math.ceil(Tcap / PAGE) * PAGE * M.HKV * M.D * (2 if variant == "dense" else 1)
        rec = (M.NL // M.n) * B * math.ceil(Tcap / MPAGE) * MPAGE * M.DK * (2 if variant == "flash16" else 1)
        return rec + 2 * M.NL * B * M.SPS * M.HKV * M.D * 2

    def fill(self):                               # benchmark: constant values (kernel time does not depend on them)
        for t in [getattr(self, k) for k in ("kc", "vc", "rc", "tkc", "tvc") if hasattr(self, k)]:
            if t.dtype == F8:
                t.view(torch.uint8).fill_(0x20)   # e4m3 0.125
            else:
                t.fill_(0.01)


# ----------------------------------------------------------------------------------------------- engine
class Engine:
    def __init__(self, M: Model, C: Cache):
        self.FA, self.MLA, self.OPS = _vllm()
        self.M, self.C, self.B = M, C, C.B
        dev = M.emb.device; B = C.B
        z = lambda dt=torch.long: torch.zeros(B, dtype=dt, device=dev)
        self.ids, self.pos, self.out = z(), z(), z()
        self.dslot, self.dlen = z(), z(torch.int32)
        self.tslot, self.tlen, self.rlen, self.rwslot = z(), z(torch.int32), z(torch.int32), z()
        self.rlen_k = torch.ones(B, dtype=torch.int32, device=dev)   # max(rlen, 1): FlashMLA faults on empty rows
        self.cu = torch.arange(B + 1, device=dev, dtype=torch.int32)
        self.ar = torch.arange(B, device=dev)
        self.wbase = self.ar * M.SPS                          # first window slot of each sequence
        self.one = torch.ones(1, device=dev, dtype=torch.float32)
        self.qd8 = torch.ones(B, M.HKV, device=dev, dtype=torch.float32)
        self.flash = C.variant.startswith("flash"); self.f8 = C.variant in ("flash8", "dense8")
        if self.flash and not M.flash:
            raise ValueError("flash variants need a params dir")
        self.keep_logits = False
        self.logits = torch.zeros(B, M.head.shape[0], device=dev, dtype=BF)
        self.graph = None; self._e0 = None

    def set_meta(self):                           # FlashMLA tile metadata for the current rlen (inside the graph)
        fn = self.MLA.get_mla_metadata_dense_fp8 if self.C.variant == "flash8" else self.MLA.get_mla_metadata
        self.md, self.ns = fn(self.rlen_k, self.M.H, 1)

    def set_decode(self, p: int):
        """Steady-state decode of position p for every sequence (sequence b at its own contiguous slots)."""
        M, C = self.M, self.C
        self.pos.fill_(p)
        if self.flash:
            self.tslot.copy_(self.ar * M.SPS + M.wslot(p)); self.tlen.fill_(min(p + 1, M.sink + M.recent + 1))
            self.rlen.fill_(max(0, p - M.recent - M.sink)); self.rlen_k.fill_(max(1, p - M.recent - M.sink))
            # p < sink has no entry; slot 0 (position sink's) is rewritten once position sink is decoded
            self.rwslot.copy_(self.ar * C.npm * MPAGE + max(0, p - M.sink))
        else:
            self.dslot.copy_(self.ar * C.npb * PAGE + p); self.dlen.fill_(p + 1)

    # ---- one decode step for B sequences (graph-capturable), with vLLM's fused ops as vLLM runs Llama
    def step(self):
        M, B, OPS = self.M, self.B, self.OPS
        if self.flash:
            self.set_meta()
        res = M.emb[self.ids]
        h = torch.empty_like(res); OPS.rms_norm(h, res, M.L[0]["ln1"], M.EPS)
        act = torch.empty(B, M.L[0]["wd"].shape[1], device=h.device, dtype=BF)
        for l in range(M.NL):
            lw = M.L[l]
            if l > 0:
                OPS.fused_add_rms_norm(h, res, lw["ln1"], M.EPS)
            o = self.attn_flash(l, lw, h) if self.flash else self.attn_dense(l, lw, h)
            h = F.linear(o, lw["wo"])
            OPS.fused_add_rms_norm(h, res, lw["ln2"], M.EPS)
            torch.ops._C.silu_and_mul(act, F.linear(h, lw["wgu"]))
            h = F.linear(act, lw["wd"])
        OPS.fused_add_rms_norm(h, res, M.norm, M.EPS)
        lg = F.linear(h, M.head)
        if self.keep_logits:
            self.logits.copy_(lg)
        self.out.copy_(lg.argmax(-1))
        return self.out

    def attn_dense(self, l, lw, h):
        M, C, B, OPS = self.M, self.C, self.B, self.OPS
        H, HKV, D = M.H, M.HKV, M.D
        qkv = F.linear(h, lw["wqkv16"][:M.NQKV])
        q, k, v = qkv[:, :H * D], qkv[:, H * D:(H + HKV) * D], qkv[:, (H + HKV) * D:]
        OPS.rotary_embedding(self.pos, q, k, D, M.cos_sin, True)
        kw = {}
        if self.f8:
            OPS.reshape_and_cache_flash(k.view(B, HKV, D), v.view(B, HKV, D), C.kc[l].view(torch.uint8),
                                        C.vc[l].view(torch.uint8), self.dslot, "fp8", self.one, self.one)
            q = q.reshape(B, H, D).to(F8)
            kw = dict(q_descale=self.qd8, k_descale=self.qd8, v_descale=self.qd8)
        else:
            OPS.reshape_and_cache_flash(k.view(B, HKV, D), v.view(B, HKV, D), C.kc[l], C.vc[l], self.dslot, "auto",
                                        self.one, self.one)
            q = q.view(B, H, D)
        o = self.FA(q, C.kc[l], C.vc[l], max_seqlen_q=1, cu_seqlens_q=self.cu, max_seqlen_k=C.Tcap,
                    seqused_k=self.dlen, causal=True, block_table=C.bt, softmax_scale=M.SC, fa_version=3, **kw)
        return o.view(B, H * D)

    def attn_flash(self, l, lw, h):
        M, C, B, OPS = self.M, self.C, self.B, self.OPS
        H, HKV, D = M.H, M.HKV, M.D
        n = M.n; g0 = l - l % n; li = l - g0; bi = g0 // n; blk = M.blocks[g0]
        f8 = self.f8; sfx = "8" if f8 else "16"
        qkvr = F.linear(h, lw["wqkv" + sfx])          # q k v | rope query | partial record entry
        qm = torch.empty(B, H, M.DK, device=h.device, dtype=F8 if f8 else BF)
        FK.q_prep(qkvr, blk["WuK" + sfx][li], self.pos, blk["theta"], blk["ss"], M.qscale[l:l + 1], qm, f8,
                  H=H, G=M.G, D=D, QR0=M.NQKV)
        q, k, v = qkvr[:, :H * D], qkvr[:, H * D:(H + HKV) * D], qkvr[:, (H + HKV) * D:M.NQKV]
        OPS.rotary_embedding(self.pos, q, k, D, M.cos_sin, True)                    # in place, after q_prep
        OPS.reshape_and_cache_flash(k.view(B, HKV, D), v.view(B, HKV, D), C.tkc[l], C.tvc[l], self.tslot, "auto",
                                    self.one, self.one)
        if f8:
            o_l, lse_r = self.MLA.flash_mla_with_kvcache_fp8(qm.view(B, 1, H, M.DK), C.rc[bi], C.rbt, self.rlen_k, M.R,
                                                             self.md, self.ns, softmax_scale=M.SC, causal=True,
                                                             descale_q=M.qscale[l:l + 1], descale_k=self.one)
        else:
            o_l, lse_r = self.MLA.flash_mla_with_kvcache(qm.view(B, 1, H, M.DK), C.rc[bi], C.rbt, self.rlen_k, M.R,
                                                         self.md, self.ns, softmax_scale=M.SC, causal=True)
        e = qkvr[:, M.NQKV + H * M.r:]
        if n == 1:                               # entry at its final slot; read once it leaves the window
            FK.rec_write(e, None, self.pos, blk["theta"], C.rc[bi].view(-1, M.DK), self.rwslot, f8, M.R)
        elif li == 0:
            self._e0 = e
        else:
            FK.rec_write(self._e0, e, self.pos, blk["theta"], C.rc[bi].view(-1, M.DK), self.rwslot, f8, M.R)
        out = torch.empty(B, H * D, device=h.device, dtype=BF)
        FK.tail_merge(q, qkvr.stride(0), C.tkc[l].view(-1, HKV, D), C.tvc[l].view(-1, HKV, D), self.wbase, self.pos,
                      self.rlen, o_l, lse_r, blk["WuV" + sfx][li], out, M.SC, M.sink, M.recent + 1, H=H, G=M.G, D=D)
        return out

    # ---- prefill (B = 1, eager): exact attention over the prompt, then the caches in the steady layout
    def prefill(self, ids):
        M, C = self.M, self.C
        H, HKV, D = M.H, M.HKV, M.D
        T = ids.numel(); dev = ids.device
        pos = torch.arange(T, device=dev)
        ang = pos.float()[:, None] * M.inv[None, :]
        emb = torch.cat([ang, ang], -1); cos, sin = emb.cos().to(BF), emb.sin().to(BF)
        cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
        x = M.emb[ids]; hb = {}
        rms = lambda t, w: w * (t.float() * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + M.EPS)).to(t.dtype)
        keep = list(range(min(T, M.sink))) + list(range(max(M.sink, T - M.recent), T))
        kidx = torch.tensor(keep, device=dev); kslot = torch.tensor([M.wslot(q_) for q_ in keep], device=dev)
        for l in range(M.NL):
            lw = M.L[l]
            xn = rms(x, lw["ln1"])
            qkv = F.linear(xn, lw["wqkv16"][:M.NQKV])
            q = rope(qkv[:, :H * D].view(T, H, D), cos, sin)
            k = rope(qkv[:, H * D:(H + HKV) * D].view(T, HKV, D), cos, sin)
            v = qkv[:, (H + HKV) * D:].reshape(T, HKV, D)
            o = self.FA(q, k, v, max_seqlen_q=T, cu_seqlens_q=cu, max_seqlen_k=T, cu_seqlens_k=cu, causal=True,
                        softmax_scale=M.SC, fa_version=3)
            if not self.flash:
                C.kc[l].view(-1, HKV, D)[:T] = k; C.vc[l].view(-1, HKV, D)[:T] = v
            else:
                n = M.n; g0 = l - l % n; li = l - g0; bi = g0 // n; blk = M.blocks[g0]
                C.tkc[l].view(-1, HKV, D)[kslot] = k[kidx]; C.tvc[l].view(-1, HKV, D)[kslot] = v[kidx]
                hb[li] = xn
                if li == n - 1 and T > M.sink:
                    Hh = torch.cat([hb[j] for j in range(n)], -1)[M.sink:]
                    e = F.linear(Hh, blk["Wdr8" if self.f8 else "Wdr16"])
                    ent = torch.cat([e[:, :M.R].float(),
                                     rope_pairs(e[:, M.R:].view(-1, 1, M.r), pos[M.sink:], blk["theta"])[:, 0]], -1)
                    dst = C.rc[bi].view(-1, M.DK)
                    if self.f8:
                        dst.view(torch.uint8)[:ent.shape[0]] = ent.clamp(-448, 448).to(F8).view(torch.uint8)
                    else:
                        dst[:ent.shape[0]] = ent.to(BF)
                    hb = {}
            x = x + F.linear(o.reshape(T, H * D), lw["wo"])
            a, b = F.linear(rms(x, lw["ln2"]), lw["wgu"]).chunk(2, -1)
            x = x + F.linear(F.silu(a) * b, lw["wd"])
        lg = F.linear(rms(x[-1:], M.norm), M.head)
        self.prefill_logits = lg
        return lg.argmax(-1)

    def capture(self):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step()

    def prime(self):
        """Capture the decode graph once (B = 1); later prefills overwrite every slot the warm-ups touched."""
        M = self.M
        p = min(M.sink + M.recent + 200, self.C.Tcap - 2)
        self.ids.fill_(1); self.set_decode(p)
        self.capture()

    def generate(self, ids, max_new: int, eos=(), sample=None):
        """B = 1 greedy (or `sample(logits) -> token` with keep_logits) decode through the captured graph.
        Returns the generated tokens (the first comes from the prefill); stops after an eos token."""
        if self.graph is None:
            self.prime()
        T = ids.numel()
        first = self.prefill(ids)
        toks = [int(sample(self.prefill_logits[0]) if sample else first.item())]
        while len(toks) < max_new and toks[-1] not in eos:
            p = T + len(toks) - 1
            self.ids.fill_(toks[-1]); self.set_decode(p)
            self.graph.replay()
            toks.append(int(sample(self.logits[0]) if sample else self.out.item()))
        return toks


# ----------------------------------------------------------------------------------------------- benchmark
def setup_bench(M, variant, B, T):
    """Steady-state decode at context T: B sequences, the current token at position T."""
    C = Cache(M, variant, B, T + 64); C.fill()
    E = Engine(M, C)
    E.ids.fill_(1); E.set_decode(T)
    return E


def bench(M, variants, Bs, Ts, steps=50, margin_gib=4.0, log=print):
    """Decode step time (CUDA graph replay, median of `steps`) over Ts x Bs x variants; skips what does not fit."""
    total = torch.cuda.get_device_properties(M.emb.device).total_memory
    out = []
    for T in Ts:
        for B in Bs:
            for v in variants:
                need = Cache.nbytes(M, v, B, T + 64)
                have = total - torch.cuda.memory_allocated() - margin_gib * GiB
                if need > have:
                    log(f"[bench] T={T} B={B} {v}: skip (cache {need / GiB:.1f} GiB > {have / GiB:.1f} GiB)")
                    out.append(dict(T=T, B=B, variant=v, skip=True, cache_gib=need / GiB))
                    continue
                E = setup_bench(M, v, B, T)
                try:
                    E.capture()
                    for _ in range(5):
                        E.graph.replay()
                    torch.cuda.synchronize()
                    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(steps)]
                    for a, b in ev:
                        a.record(); E.graph.replay(); b.record()
                    torch.cuda.synchronize()
                    ts = sorted(a.elapsed_time(b) for a, b in ev)
                    ms = ts[len(ts) // 2]
                    rec = dict(T=T, B=B, variant=v, ms=ms, ms_mean=sum(ts) / len(ts), tok_s=B * 1000 / ms,
                               cache_gib=need / GiB)
                    log(f"[bench] T={T} B={B} {v}: {ms:.2f} ms/step  {rec['tok_s']:.0f} tok/s  cache {need / GiB:.1f} GiB")
                except Exception as e:  # noqa: BLE001
                    rec = dict(T=T, B=B, variant=v, error=repr(e)[:300])
                    log(f"[bench] T={T} B={B} {v}: ERROR {rec['error']}")
                out.append(rec)
                del E
                torch.cuda.empty_cache()
    return out


def needle_ids(tok, T: int, depth: float = 0.5, key: str = "48213"):
    """Passkey prompt of about T tokens (chat template): filler, the key at `depth`, the question."""
    fill = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    n = max(1, (T - 200) // len(tok(fill, add_special_tokens=False).input_ids))
    k = int(n * depth)
    text = fill * k + f"The pass key is {key}. Remember it. {key} is the pass key. " + fill * (n - k)
    msg = [{"role": "user", "content": text + "\nWhat is the pass key? Answer with the number only."}]
    out = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt")
    out = out["input_ids"] if not torch.is_tensor(out) else out            # BatchEncoding in newer transformers
    return out.view(-1), key


def needle_check(M, tok, variant, T=32768, log=print) -> bool:
    """Correctness gate before timing: prefill + graph-replayed decode must retrieve the passkey."""
    ids, key = needle_ids(tok, T)
    ids = ids.to(M.emb.device)
    E = Engine(M, Cache(M, variant, 1, ids.numel() + 64))
    toks = E.generate(ids, 12, eos=set(tok.convert_tokens_to_ids(["<|eot_id|>", "<|end_of_text|>"])))
    ans = tok.decode(toks, skip_special_tokens=True)
    ok = key in ans
    log(f"[needle] {variant} at {ids.numel()} tokens: {'PASS' if ok else 'FAIL'} ({ans.strip()!r})")
    del E
    torch.cuda.empty_cache()
    return ok
