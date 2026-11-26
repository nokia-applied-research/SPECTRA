"""SPECTRA-Flash for Qwen3.5 hybrid models (`qwen3_5`, e.g. Qwen/Qwen3.5-9B) in vLLM 0.29.

Qwen3.5 keeps a KV cache only in its full-attention layers (every 4th layer); the Gated DeltaNet layers and their
state stay exactly as vLLM runs them. The full-attention layers are grouped into blocks of n consecutive ones (n = 1:
SPECTRA-Flash; n = 2: layers 3 + 7, 11 + 15, ...: BellowsKV-Flash); each block shares one FlashMLA record (512 latent
+ 64 RoPE dims, fp8; entry = sum over the block's layers of h_l @ its rows of [W_d | W_r], written by the block's last
layer), and each layer has its own exact window (sink 4 + recent 123), the caches of `caches.py`. The module keeps vLLM's
`Qwen3NextAttention` parameters (qkv_proj with the fused gate, q_norm / k_norm, partial RoPE, o_proj), so the checkpoint
loads unchanged, and replaces its `Attention` with the op `vllm::spectra_flash_q35`:

    q, gate, k, v = qkv_proj(h);  qn = q_norm(q);  kn = k_norm(k);  e = h @ Wdr8^T     (record entry, fp8 units)
    qr = qn W_QR (per head, the rope query of the record);  qq = [qn | qr]
    op: q_prep(qq) for decode rows -> partial RoPE on qn, kn -> window write -> decode: FlashMLA + tail_merge;
        prefill: exact causal attention (FA3) -> record write (e)
    out = attn * sigmoid(gate);  o_proj(out)

Selected by `--hf-overrides '{"spectra_flash": "<params dir or Hub repo>"}'` with `--language-model-only` (text-only
positions); the params are a SPECTRA-Flash params dir (params.py) with `arch` "qwen3_5". The hybrid allocator needs
`--block-size` = a multiple of 135 with block_size x 4096 B >= one GDN state (540 on the 9B), so the record (64 x 576 B
pages) and window (135 x 4096 B rings) pages divide the unified page without padding. The query side of the record is
the exact normalized query (qn W_UK, RoPE(qn W_QR)); the deployed kernel quantizes it to fp8 with a static per-layer
scale.
"""
from __future__ import annotations

import json
import os
import re

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import direct_register_custom_op

from .. import deploy
from .. import params as P
from .caches import SETTINGS, RecordCache, WindowCache, ring_capacity
from .op import _chunk_attention, _kernels, _prefill_rows, _ring_past

logger = init_logger("vllm.spectra")
OP_NAME = "spectra_flash_q35"
QUALIFIED_OP_NAME = f"vllm::{OP_NAME}"
_STEP: dict = {}
_ORIG: dict = {}
_SHARED: dict = {}


def _src(hf_config):
    return getattr(hf_config, "spectra_flash", None) or os.environ.get("SPECTRA_FLASH")


def q35_settings(vllm_config) -> dict:
    if SETTINGS:
        return SETTINGS
    hf = vllm_config.model_config.hf_config
    d = deploy.resolve(_src(hf))
    meta = json.load(open(os.path.join(d, "flash_config.json")))
    if meta.get("arch") != "qwen3_5":
        raise ValueError(f"{d}: SPECTRA-Flash params for {meta.get('model')}, not a Qwen3.5 model")
    tc = getattr(hf, "text_config", hf)
    cfg = meta["config"]; n = cfg["group_size"]
    full = [i for i, t in enumerate(tc.layer_types) if t == "full_attention"]
    groups = [list(g) for g in meta["blocks"]]
    if groups != [full[i:i + n] for i in range(0, len(full), n)]:
        raise ValueError(f"{d}: blocks {groups} are not consecutive groups of {n} full-attention layers {full}")
    layout = {l: (g[0], li) for g in groups for li, l in enumerate(g)}     # layer -> (block's first layer, index)
    R, r = cfg["r_latent"], cfg["r_rope"]
    dep = deploy.deploy_settings(meta)
    hkv, hd = tc.num_key_value_heads, tc.head_dim
    SETTINGS.update(dir=d, n=n, R=R, r=r, DK=R + r, sink=dep["sink"], recent=dep["recent"], layout=layout,
                    cap=ring_capacity(dep["sink"], dep["recent"], hkv, hd, R + r))
    return SETTINGS


def _check(vllm_config) -> None:
    from vllm.distributed import get_tensor_model_parallel_world_size
    if get_tensor_model_parallel_world_size() != 1:
        raise NotImplementedError("SPECTRA-Flash (Qwen3.5) supports tensor_parallel_size=1 only")
    if vllm_config.model_config.dtype != torch.bfloat16:
        raise NotImplementedError("SPECTRA-Flash runs in bfloat16")
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("SPECTRA-Flash needs --no-enable-prefix-caching")


class _NoAttention(torch.nn.Module):
    """Stands in for vLLM's `Attention` while the base class is built: no backend selection, no cache spec."""

    def __init__(self, *args, **kwargs):
        super().__init__()


def make_attention_class(base):
    class SpectraFlashQwenAttention(base):
        """Qwen3NextAttention with the record + window caches and the spectra_flash_q35 op instead of `Attention`."""

        def __init__(self, *args, **kwargs):
            import vllm.model_executor.models.qwen3_next as q3n
            saved = q3n.Attention
            q3n.Attention = _NoAttention                 # the base would select a backend for block_size 540 and fail
            try:
                super().__init__(*args, **kwargs)
            finally:
                q3n.Attention = saved
            prefix = kwargs.get("prefix", "")
            vllm_config = get_current_vllm_config()
            _check(vllm_config)
            st = q35_settings(vllm_config)
            ctx = vllm_config.compilation_config.static_forward_context
            ctx.pop(f"{prefix}.attn", None)
            del self.attn
            self.layer_name = f"{prefix}.attn"
            ctx[self.layer_name] = self
            l = extract_layer_index(prefix)
            tc = getattr(vllm_config.model_config.hf_config, "text_config", vllm_config.model_config.hf_config)
            full = [i for i, t in enumerate(tc.layer_types) if t == "full_attention"]
            self.spf_l, self.first = l, (l == full[0])
            self.g0, self.li = st["layout"][l]
            self.n = st["n"]
            self.sink, self.recent, self.R, self.r, self.DK = st["sink"], st["recent"], st["R"], st["r"], st["DK"]
            H, HKV, D, d = self.num_heads, self.num_kv_heads, self.head_dim, self.hidden_size
            self.G = H // HKV
            self.spf_window_cache = WindowCache(f"{prefix}.spf_window", HKV, D, st["cap"])
            if self.li == 0:                              # the block's first layer owns its record
                self.spf_record_cache = RecordCache(f"{prefix}.spf_record", self.DK)
                rec = self.spf_record_cache
            else:                                         # later layers of the block read the same record
                rec = ctx[re.sub(rf"\.{l}\.", f".{self.g0}.", prefix, count=1) + ".spf_record"]
            object.__setattr__(self, "spf_record", rec)
            object.__setattr__(self, "spf_window", self.spf_window_cache)
            if self.n > 1:                                # first layer's partial entries, kept for the last layer
                max_tokens = max(vllm_config.scheduler_config.max_num_batched_tokens,
                                 vllm_config.compilation_config.max_cudagraph_capture_size or 0)
                dev = self.qkv_proj.weight.device
                key = ("e0", str(dev))
                buf = _SHARED.get(key)
                if buf is None or buf.shape[0] < max_tokens:
                    buf = torch.zeros(max_tokens, self.DK, dtype=torch.bfloat16, device=dev)
                    _SHARED[key] = buf
                object.__setattr__(self, "e0buf", buf)
            bf = torch.bfloat16
            self.register_buffer("Wdr8", torch.zeros(self.DK, d, dtype=bf), persistent=False)
            self.register_buffer("WQR", torch.zeros(H, D, self.r, dtype=bf), persistent=False)
            self.register_buffer("WuK", torch.zeros(HKV, D, self.R, dtype=bf), persistent=False)
            self.register_buffer("WuV", torch.zeros(HKV, self.R, D, dtype=bf), persistent=False)
            self.register_buffer("theta", torch.zeros(self.r // 2, dtype=torch.float32), persistent=False)
            self.register_buffer("ss", torch.ones(self.r, dtype=torch.float32), persistent=False)
            self.register_buffer("qscale", torch.ones(1, dtype=torch.float32), persistent=False)
            self.register_buffer("one", torch.ones(1, dtype=torch.float32), persistent=False)
            self.spf_loaded = False

        def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
            T = hidden_states.shape[0]; H, HKV, D = self.num_heads, self.num_kv_heads, self.head_dim
            if positions.ndim == 2:
                positions = positions[0]
            qkv, _ = self.qkv_proj(hidden_states)
            q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            q, gate = torch.chunk(q_gate.view(T, H, 2 * D), 2, dim=-1)
            qn = self.q_norm(q.reshape(T, H, D))
            kn = self.k_norm(k.reshape(T, HKV, D)).reshape(T, HKV * D)
            qr = torch.einsum("thd,hdr->thr", qn, self.WQR)
            qq = torch.cat([qn.reshape(T, H * D), qr.reshape(T, H * self.r)], -1)
            e = F.linear(hidden_states, self.Wdr8)
            out = torch.empty(T, H * D, dtype=hidden_states.dtype, device=hidden_states.device)
            torch.ops.vllm.spectra_flash_q35(qq, kn, v.contiguous(), e, positions, out, self.layer_name)
            out = out * torch.sigmoid(gate.reshape(T, H * D))
            output, _ = self.o_proj(out)
            return output

    SpectraFlashQwenAttention.__name__ = "SpectraFlashQwenAttention"
    return SpectraFlashQwenAttention


# ----------------------------------------------------------------------------------------------- the op
def run_layer_q35(layer, qq, kn, v, e, positions, output) -> None:
    md = get_forward_context().attn_metadata
    if isinstance(md, list):
        md = md[0]
    rec, win = layer.spf_record, layer.spf_window
    if not isinstance(md, dict) or rec.pages is None or win.k is None:
        output.zero_()
        return
    FK, MLA, FA, OPS = _kernels()
    rmd, wmd = md[rec.prefix], md[win.prefix]
    T = qq.shape[0]
    H, HKV, D, DK, G = layer.num_heads, layer.num_kv_heads, layer.head_dim, layer.DK, layer.G
    nd, B = rmd.num_decodes, rmd.num_reqs
    q = qq[:, :H * D]
    pre = _prefill_rows(rmd, nd, B)
    if nd > 0:
        qm = torch.empty(nd, H, DK, device=qq.device, dtype=torch.float8_e4m3fn)
        FK.q_prep(qq[:nd], layer.WuK, positions[:nd], layer.theta, layer.ss, layer.qscale, qm, True, H=H, G=G, D=D,
                  QR0=H * D)
    qmc = {}
    for b, a0, e0, C in pre:
        if C - layer.recent - layer.sink > 0:
            qmc[b] = torch.empty(e0 - a0, H, DK, device=qq.device, dtype=torch.float8_e4m3fn)
            FK.q_prep(qq[a0:e0], layer.WuK, positions[a0:e0], layer.theta, layer.ss, layer.qscale, qmc[b], True,
                      H=H, G=G, D=D, QR0=H * D)
    OPS.rotary_embedding(positions[:T], q, kn, D, layer.rotary_emb.cos_sin_cache, True)     # partial NeoX RoPE
    past = {b: _ring_past(win, wmd.wbase[b], C, layer.sink, layer.recent + 1) for b, a0, e0, C in pre if C > 0}
    OPS.reshape_and_cache_flash(kn.view(T, HKV, D), v.view(T, HKV, D), win.k_cache, win.v_cache, wmd.wslot[:T], "auto",
                                layer.one, layer.one)
    if nd > 0:
        if layer.first or "md" not in _STEP:
            _STEP["md"], _STEP["ns"] = MLA.get_mla_metadata_dense_fp8(rmd.rlen_k[:nd], H, 1)
        o_l, lse = MLA.flash_mla_with_kvcache_fp8(qm.view(nd, 1, H, DK), rec.pages, rmd.block_table[:nd],
                                                  rmd.rlen_k[:nd], layer.R, _STEP["md"], _STEP["ns"],
                                                  softmax_scale=layer.scaling, causal=True, descale_q=layer.qscale,
                                                  descale_k=layer.one)
        FK.tail_merge(q[:nd], qq.stride(0), win.k, win.v, wmd.wbase[:nd], positions[:nd], rmd.rlen[:nd], o_l, lse,
                      layer.WuV, output[:nd], layer.scaling, layer.sink, layer.recent + 1, H=H, G=G, D=D)
    end = nd
    for b, a0, e0, C in pre:
        S = e0 - a0
        if C == 0:
            cu = torch.tensor([0, S], device=qq.device, dtype=torch.int32)
            # straight into `output`: vLLM's memory profile runs without attention metadata, so a temporary of the
            # prompt's size here (1 GiB at 128K) is unaccounted and fragments the allocator
            FA(q[a0:e0].reshape(S, H, D), kn[a0:e0].view(S, HKV, D), v[a0:e0].view(S, HKV, D), max_seqlen_q=S,
               cu_seqlens_q=cu, max_seqlen_k=S, cu_seqlens_k=cu, causal=True, softmax_scale=layer.scaling,
               fa_version=3, out=output[a0:e0].view(S, H, D))
        else:
            _chunk_attention(layer, FA, MLA, rec, rmd, b, q[a0:e0].reshape(S, H * D), kn[a0:e0], v[a0:e0],
                             qmc.get(b), past[b], C, output[a0:e0])
        end = max(end, e0)
    if end < T:
        output[end:].zero_()
    # the block's record entries: e = this layer's partial (h_l @ its rows of [W_d | W_r]); the block's last layer sums
    # them and writes. Earlier layers of the block attend only to entries of earlier steps (record < t - recent).
    if layer.n == 1:
        FK.rec_write(e, None, positions[:T], layer.theta, rec.rows, rmd.rslot[:T], True, layer.R)
    elif layer.li == 0:
        layer.e0buf[:T].copy_(e)
    elif layer.li < layer.n - 1:
        layer.e0buf[:T].add_(e)
    else:
        FK.rec_write(layer.e0buf[:T], e, positions[:T], layer.theta, rec.rows, rmd.rslot[:T], True, layer.R)


def spectra_flash_q35(qq: torch.Tensor, kn: torch.Tensor, v: torch.Tensor, e: torch.Tensor, positions: torch.Tensor,
                      output: torch.Tensor, layer_name: str) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    run_layer_q35(layer, qq, kn, v, e, positions, output)


def spectra_flash_q35_fake(qq, kn, v, e, positions, output, layer_name) -> None:
    return


def register_op() -> None:
    if hasattr(torch.ops.vllm, OP_NAME):
        return
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
    direct_register_custom_op(op_name=OP_NAME, op_func=eager_break_during_capture(spectra_flash_q35),
                              mutates_args=["qq", "kn", "output"], fake_impl=spectra_flash_q35_fake)


# ----------------------------------------------------------------------------------------------- params
@torch.no_grad()
def load_q35_params(model) -> None:
    cls = _ORIG.get("flash_cls")
    mods = [m for m in model.modules() if cls is not None and isinstance(m, cls)]
    if not mods:
        return
    st = SETTINGS
    by_layer = {l: (bp, i) for bp in P.load(st["dir"])["blocks"].values() for i, l in enumerate(bp["layers"])}
    for m in mods:
        bp, i = by_layer[m.spf_l]
        cs, ss = bp["cs"].float(), bp["ss"].float()
        H, HKV, D, R, d = m.num_heads, m.num_kv_heads, m.head_dim, m.R, m.hidden_size
        dev = m.Wdr8.device
        Wdr = torch.cat([bp["Wd"].float() / cs[None, :], bp["Wr"].float() / ss[None, :]], 1)   # (n d, R + r)
        m.Wdr8.copy_(Wdr[i * d:(i + 1) * d].t().to(dev, torch.bfloat16))                     # this layer's rows
        m.WQR.copy_(bp["WQR"][i].float().to(dev, torch.bfloat16))                               # (H, D, r)
        WuK = bp["WuK"][i].float().view(R, HKV, D).permute(1, 2, 0) * cs[None, None, :]         # (HKV, D, R)
        WuV = bp["WuV"][i].float().view(R, HKV, D).permute(1, 0, 2) * cs[None, :, None]         # (HKV, R, D)
        m.WuK.copy_(WuK.to(dev, torch.bfloat16)); m.WuV.copy_(WuV.to(dev, torch.bfloat16))
        m.theta.copy_(bp["theta"].float().to(dev)); m.ss.copy_(ss.to(dev)); m.qscale.fill_(bp["qscale"][i])
        m.spf_loaded = True
    logger.info("[spectra-flash] Qwen3.5 params %s: %d full-attention layers %s in blocks of n=%d, record %d+%d fp8, "
                "sink %d + recent %d, ring %d slots", st["dir"], len(mods), [m.spf_l for m in mods], st["n"], st["R"],
                st["r"], st["sink"], st["recent"], st["cap"])


def patch() -> None:
    """Swap Qwen3.5's full-attention class (only models whose hf_config names Flash params build the Flash class)
    and load the params after the checkpoint. Idempotent; every vLLM process runs it through the plugin."""
    if _ORIG:
        return
    import vllm.model_executor.models.qwen3_5 as q35
    base = getattr(q35, "_flash_base_attention", None) or q35.Qwen3NextAttention     # vLLM's class
    q35._flash_base_attention = base
    prev = q35.Qwen3NextAttention          # vLLM's class, or another Flash plugin's factory (BellowsKV-Flash)
    flash_cls = make_attention_class(base)

    def factory(*args, **kwargs):
        if _src(get_current_vllm_config().model_config.hf_config):
            return flash_cls(*args, **kwargs)
        return prev(*args, **kwargs)
    q35.Qwen3NextAttention = factory
    for cls_name in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"):
        cls = getattr(q35, cls_name, None)
        if cls is None or "load_weights" not in cls.__dict__:
            continue
        orig = cls.load_weights

        def load_weights(self, weights, _orig=orig):
            out = _orig(self, weights)
            load_q35_params(self)
            return out
        cls.load_weights = load_weights
    _ORIG.update(base=base, flash_cls=flash_cls)
    register_op()
