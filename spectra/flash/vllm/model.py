"""Llama with SPECTRA-Flash attention for vLLM 0.29: `SpectraFlashLlamaForCausalLM`.

A stock Llama checkpoint loads unchanged (the layers keep LlamaAttention's qkv_proj / o_proj / rotary table). After it,
`load_weights` loads the Flash params (local dir or HF Hub repo, from hf_config.spectra_flash, set by --hf-overrides),
folds them exactly as the engine does and extends each layer's qkv GEMM with its rope-query rows and its half of the
block's record map: one GEMM gives q | k | v | rope query | partial record entry (deploy.py). Each layer owns its exact
window cache; each block's first layer owns the block's record cache (caches.py).
"""
from __future__ import annotations

import json
import os
import re

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM
from vllm.model_executor.models.utils import extract_layer_index

from .. import deploy
from .caches import SETTINGS, RecordCache, WindowCache, ring_capacity
from .op import register_op

logger = init_logger("vllm.spectra")
register_op()
_SHARED: dict = {}


def flash_settings(vllm_config: VllmConfig) -> dict:
    """The deploy settings of the params named by hf_config.spectra_flash (read once per process)."""
    if SETTINGS:
        return SETTINGS
    hf = vllm_config.model_config.hf_config
    src = getattr(hf, "spectra_flash", None) or os.environ.get("SPECTRA_FLASH")
    if not src:
        raise ValueError('SPECTRA-Flash needs its params: --hf-overrides \'{"architectures": '
                         '["SpectraFlashLlamaForCausalLM"], "spectra_flash": "<params dir or HF Hub repo>"}\'')
    d = deploy.resolve(src)
    meta = json.load(open(os.path.join(d, "flash_config.json")))
    cfg = meta["config"]
    dep = deploy.deploy_settings(meta)
    if not dep["fp8"]:
        raise NotImplementedError("the vLLM plugin runs the deployed fp8 record only")
    hkv = hf.num_key_value_heads
    hd = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
    n, R, r = cfg["group_size"], cfg["r_latent"], cfg["r_rope"]
    if n not in (1, 2):
        raise NotImplementedError(f"the vLLM plugin supports blocks of 1 or 2 layers (params have {n})")
    SETTINGS.update(dir=d, n=n, R=R, r=r, DK=R + r, sink=dep["sink"], recent=dep["recent"],
                    cap=ring_capacity(dep["sink"], dep["recent"], hkv, hd, R + r))
    return SETTINGS


def _check_config(vllm_config: VllmConfig) -> None:
    if get_tensor_model_parallel_world_size() != 1:
        raise NotImplementedError("SPECTRA-Flash on vLLM supports tensor_parallel_size=1 only")
    if vllm_config.model_config.dtype != torch.bfloat16:
        raise NotImplementedError("SPECTRA-Flash runs in bfloat16: start vLLM with --dtype bfloat16")
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("SPECTRA-Flash needs --no-enable-prefix-caching")
    if vllm_config.speculative_config is not None:
        raise NotImplementedError("SPECTRA-Flash does not support speculative decoding")
    if vllm_config.kv_transfer_config is not None:
        raise NotImplementedError("SPECTRA-Flash does not support KV connectors")
    sc = vllm_config.scheduler_config
    if sc.enable_chunked_prefill and sc.max_num_batched_tokens < vllm_config.model_config.max_model_len:
        logger.warning("[spectra-flash] chunked prefill is on (chunks of up to %d tokens): a later chunk of a prompt "
                       "sees the earlier chunks through the record and the exact window, an approximation. The "
                       "evaluated config prefills each prompt in one step: --no-enable-chunked-prefill and "
                       "--max-num-batched-tokens >= --max-model-len", sc.max_num_batched_tokens)


class SpectraFlashAttention(LlamaAttention):
    """LlamaAttention whose `Attention` is replaced by the spectra_flash op and the layer's Flash caches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prefix = kwargs.get("prefix")
        if prefix is None:
            raise ValueError("SpectraFlashAttention needs prefix=")
        vllm_config = get_current_vllm_config()
        st = flash_settings(vllm_config)
        n = st["n"]
        l = extract_layer_index(prefix)
        g0, li = l - l % n, l % n
        if g0 + n > vllm_config.model_config.hf_config.num_hidden_layers:
            raise NotImplementedError(f"layer {l} is outside a full block of {n} layers")
        ctx = vllm_config.compilation_config.static_forward_context
        ctx.pop(f"{prefix}.attn", None)                        # drop the stock Attention built by the base class
        del self.attn
        self.layer_name = f"{prefix}.attn"
        ctx[self.layer_name] = self                            # op lookup only (not an AttentionLayerBase)
        self.spf_l, self.g0, self.li, self.n = l, g0, li, n
        self.sink, self.recent, self.R, self.r, self.DK = st["sink"], st["recent"], st["R"], st["r"], st["DK"]
        H, HKV, D, d = self.num_heads, self.num_kv_heads, self.head_dim, self.hidden_size
        self.G = H // HKV
        self.NQKV = self.q_size + 2 * self.kv_size
        self.E0 = self.NQKV + H * self.r
        self.spf_window_cache = WindowCache(f"{prefix}.spf_window", HKV, D, st["cap"])
        if li == 0:
            self.spf_record_cache = RecordCache(f"{prefix}.spf_record", self.DK)
            rec = self.spf_record_cache
        else:
            rec = ctx[re.sub(rf"\.{l}\.", f".{g0}.", prefix, count=1) + ".spf_record"]
        object.__setattr__(self, "spf_record", rec)            # plain references for the op
        object.__setattr__(self, "spf_window", self.spf_window_cache)
        bf = torch.bfloat16
        self.register_buffer("W", torch.zeros(self.E0 + self.DK, d, dtype=bf), persistent=False)
        self.register_buffer("WuK", torch.zeros(HKV, D, self.R, dtype=bf), persistent=False)
        self.register_buffer("WuV", torch.zeros(HKV, self.R, D, dtype=bf), persistent=False)
        self.register_buffer("theta", torch.zeros(self.r // 2, dtype=torch.float32), persistent=False)
        self.register_buffer("ss", torch.ones(self.r, dtype=torch.float32), persistent=False)
        self.register_buffer("qscale", torch.ones(1, dtype=torch.float32), persistent=False)
        self.register_buffer("one", torch.ones(1, dtype=torch.float32), persistent=False)
        max_tokens = max(vllm_config.scheduler_config.max_num_batched_tokens,
                         vllm_config.compilation_config.max_cudagraph_capture_size or 0)
        key = ("e0", str(self.W.device))
        buf = _SHARED.get(key)
        if buf is None or buf.shape[0] < max_tokens:
            buf = torch.zeros(max_tokens, self.DK, dtype=bf, device=self.W.device)
            _SHARED[key] = buf
        object.__setattr__(self, "e0buf", buf)                  # block's first-layer partial entries (persistent)
        self.spf_loaded = False

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkvr = F.linear(hidden_states, self.W)                  # q | k | v | rope query | partial record entry
        out = torch.empty(qkvr.shape[0], self.num_heads * self.head_dim, dtype=qkvr.dtype, device=qkvr.device)
        torch.ops.vllm.spectra_flash(qkvr, positions, out, self.layer_name)
        output, _ = self.o_proj(out)
        return output


class SpectraFlashDecoderLayer(LlamaDecoderLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "", config=None, attn_layer_type=None):
        super().__init__(vllm_config=vllm_config, prefix=prefix, config=config,
                         attn_layer_type=SpectraFlashAttention)


@torch.no_grad()
def load_flash_params(model) -> None:
    from ..engine import rope_inv_freq
    st = SETTINGS
    layers = [m.self_attn for m in model.model.layers if isinstance(getattr(m, "self_attn", None),
                                                                     SpectraFlashAttention)]
    a0 = layers[0]
    dev, d = a0.W.device, a0.hidden_size
    n, blocks, qs, meta, _ = deploy.load(st["dir"], dev, a0.num_kv_heads, a0.head_dim)
    deploy.check_rope(blocks, rope_inv_freq(model.config, dev))
    for at in layers:
        blk, li, l = blocks[at.g0], at.li, at.spf_l
        qkv = at.qkv_proj.weight
        at.W.copy_(torch.cat([qkv.to(at.W.dtype), blk["WQR"][li], blk["Wdr8"][:, li * d:(li + 1) * d]], 0))
        at.qkv_proj.weight.data = at.W[:at.NQKV]               # no second copy of q / k / v
        at.WuK.copy_(blk["WuK8"][li]); at.WuV.copy_(blk["WuV8"][li])
        at.theta.copy_(blk["theta"]); at.ss.copy_(blk["ss"]); at.qscale.fill_(float(qs[l]))
        at.spf_loaded = True
    del blocks
    torch.cuda.empty_cache()
    logger.info("[spectra-flash] params %s: %d blocks of %d layers, record %d+%d fp8, sink %d + recent %d exact, "
                "ring %d slots", st["dir"], len(layers) // n, n, st["R"], st["r"], st["sink"], st["recent"], st["cap"])


class SpectraFlashLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", layer_type=None):
        _check_config(vllm_config)
        flash_settings(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=SpectraFlashDecoderLayer)

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        load_flash_params(self)
        return loaded
