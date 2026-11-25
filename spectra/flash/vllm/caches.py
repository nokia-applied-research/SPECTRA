"""The caches of SPECTRA-Flash in vLLM 0.29: per block of n layers (n = 1) one FlashMLA record, per layer one exact window.

  RecordCache (one per block): FullAttentionSpec, 64-token pages of R + r = 576 B per token (512 latent + 64 RoPE dims
      in fp8; declared bf16 so every cache group has one dtype, and re-viewed as fp8). FlashMLA reads it through the
      group's block table at 64-token kernel blocks. The entry of position q sits at record index q - sink.
  WindowCache (one per layer): CircularBufferSpec, one ring block per request holding the bf16 K|V of the sink
      positions and of the last `recent` tokens plus the current one: position q at slot q (q < sink) or
      sink + q mod (recent + 1), the engine's layout (so the window attention sums in the engine's order).
The ring capacity is the smallest multiple of `unit` >= sink + recent + 1, where `unit` makes the ring page a whole
number of record pages. vLLM then scales the record's block size up to the ring page, so neither cache is padded.

Both are `AttentionLayerBase` side caches registered in `static_forward_context` (the QSA / Inkling / DSv4 pattern):
the runner allocates them, builds their metadata with the builders below and binds the tensors; the per-layer op
(`spectra.flash.vllm.op`) reads them through the forward context. Everything the decode path reads per step lives in
persistent GPU buffers owned by the builders, so a decode step can be replayed in a FULL CUDA graph.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (AttentionBackend, AttentionCGSupport, AttentionMetadata,
                                       AttentionMetadataBuilder, CommonAttentionMetadata, MultipleOf)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec, KVCacheSpec

logger = init_logger("vllm.spectra")
_LOGGED: set = set()

PAGE = 64                                  # FlashMLA page (tokens)
SETTINGS: dict = {}                        # sink, recent, ring capacity, ... (set by the model, read by the builders)


def ring_capacity(sink: int, recent: int, hkv: int, hd: int, dk: int) -> int:
    rec_page = PAGE * dk                   # bytes (fp8 entries)
    tok = hkv * 2 * hd * 2                 # bytes per window token (bf16 K|V)
    unit = rec_page // math.gcd(rec_page, tok)
    cap = -(-(sink + recent + 1) // unit) * unit
    assert cap * tok % rec_page == 0
    return cap


def _register(module: nn.Module, prefix: str, vllm_config: VllmConfig) -> None:
    ctx = vllm_config.compilation_config.static_forward_context
    if prefix in ctx:
        raise ValueError(f"Duplicate layer name: {prefix}")
    ctx[prefix] = module


# ----------------------------------------------------------------------------------------------- metadata
@dataclass
class RecordMetadata(AttentionMetadata):
    block_table: torch.Tensor          # (B, kernel blocks) 64-token record pages
    rlen: torch.Tensor                 # (B,) int32 record entries a decode row reads (persistent)
    rlen_k: torch.Tensor               # (B,) int32 max(rlen, 1): FlashMLA faults on empty rows (persistent)
    rslot: torch.Tensor                # (T_max,) int64 record row of each token's entry, -1 = none (persistent)
    num_reqs: int
    num_decodes: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor      # (B + 1,)
    query_start_loc_cpu: torch.Tensor  # (B + 1,) CPU
    seq_lens_cpu: torch.Tensor | None  # (B,) CPU (exact for prefill rows)


@dataclass
class WindowMetadata(AttentionMetadata):
    wbase: torch.Tensor                # (B,) int64 first slot of each request's ring (persistent)
    wslot: torch.Tensor                # (T_max,) int64 ring slot each token is written to, -1 = not kept (persistent)
    num_reqs: int


def _token_positions(cam: CommonAttentionMetadata, B: int, T: int, ar: torch.Tensor):
    """(request of each token, position of each token, seq_len of each request) for the T actual tokens."""
    qsl = cam.query_start_loc[:B + 1].long()
    L = cam.seq_lens[:B].long()
    tok = ar[:T]
    req = torch.searchsorted(qsl[1:].contiguous(), tok, right=True).clamp_(max=max(B - 1, 0))
    pos = (L - (qsl[1:] - qsl[:-1]))[req] + (tok - qsl[req])
    return req, pos, L


def _cpu_lens(cam: CommonAttentionMetadata):
    seq_cpu = cam.seq_lens_cpu_upper_bound
    if seq_cpu is None:
        seq_cpu = cam.seq_lens[:cam.num_reqs].cpu()
    return cam.query_start_loc_cpu[:cam.num_reqs + 1], seq_cpu[:cam.num_reqs]


class _Builder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec: KVCacheSpec, layer_names: list[str], vllm_config: VllmConfig,
                 device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1)                   # decode rows first
        if not SETTINGS:
            raise RuntimeError("SPECTRA-Flash builder created before the model set its settings")
        self.sink, self.recent = SETTINGS["sink"], SETTINGS["recent"]
        self.max_reqs = vllm_config.scheduler_config.max_num_seqs
        cc = vllm_config.compilation_config
        self.max_tokens = max(vllm_config.scheduler_config.max_num_batched_tokens, cc.max_cudagraph_capture_size or 0)
        self.ar = torch.arange(self.max_tokens, device=device)
        self._hw_tok = 0                                        # high-water marks of rows written last step
        self._hw_req = 0

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata):
        common_attn_metadata.seq_lens.fill_(1)                  # cheap capture; real lengths arrive at replay
        return self.build(0, common_attn_metadata)


class RecordBuilder(_Builder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.rlen = torch.zeros(self.max_reqs, dtype=torch.int32, device=device)
        self.rlen_k = torch.ones(self.max_reqs, dtype=torch.int32, device=device)
        self.rslot = torch.full((self.max_tokens,), -1, dtype=torch.int64, device=device)

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> RecordMetadata:
        cam = common_attn_metadata
        B, T = cam.num_reqs, cam.num_actual_tokens
        nd, _, _, _ = split_decodes_and_prefills(cam, decode_threshold=1)
        s = self.sink
        if T > 0:
            req, pos, L = _token_positions(cam, B, T, self.ar)
            idx = (pos - s).clamp(min=0)
            kbt = cam.block_table_tensor
            slot = kbt[req, idx // PAGE].long() * PAGE + idx % PAGE
            self.rslot[:T].copy_(torch.where(pos >= s, slot, torch.full_like(slot, -1)))
            # decode rows (p = L - 1) read the entries [sink, p - recent)
            rl = (L - 1 - self.recent - s).clamp(min=0).to(torch.int32)
            self.rlen[:B].copy_(rl)
            self.rlen_k[:B].copy_(rl.clamp(min=1))
        if self._hw_tok > T:
            self.rslot[T:self._hw_tok].fill_(-1)
        if self._hw_req > B:
            self.rlen[B:self._hw_req].zero_(); self.rlen_k[B:self._hw_req].fill_(1)
        self._hw_tok, self._hw_req = T, B
        qsl_cpu, seq_cpu = _cpu_lens(cam)
        return RecordMetadata(block_table=cam.block_table_tensor, rlen=self.rlen, rlen_k=self.rlen_k, rslot=self.rslot,
                              num_reqs=B, num_decodes=nd, num_actual_tokens=T, query_start_loc=cam.query_start_loc,
                              query_start_loc_cpu=qsl_cpu, seq_lens_cpu=seq_cpu)


class WindowBuilder(_Builder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.cap = kv_cache_spec.block_size                     # ring capacity (slots)
        self.wbase = torch.zeros(self.max_reqs, dtype=torch.int64, device=device)
        self.wslot = torch.full((self.max_tokens,), -1, dtype=torch.int64, device=device)

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> WindowMetadata:
        cam = common_attn_metadata
        B, T = cam.num_reqs, cam.num_actual_tokens
        s, ring = self.sink, self.recent + 1
        base = cam.block_table_tensor[:B, 0].long() * self.cap
        self.wbase[:B].copy_(base)
        if T > 0:
            req, pos, L = _token_positions(cam, B, T, self.ar)
            keep = (pos < s) | (pos >= L[req] - self.recent)    # the sinks and the last `recent` tokens (+ current)
            ws = torch.where(pos < s, pos, s + pos % ring)
            self.wslot[:T].copy_(torch.where(keep, base[req] + ws, torch.full_like(ws, -1)))
        if self._hw_tok > T:
            self.wslot[T:self._hw_tok].fill_(-1)
        if self._hw_req > B:
            self.wbase[B:self._hw_req].zero_()
        self._hw_tok, self._hw_req = T, B
        return WindowMetadata(wbase=self.wbase, wslot=self.wslot, num_reqs=B)


# ----------------------------------------------------------------------------------------------- backends
class _NoImplBackend(AttentionBackend):
    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("SPECTRA-Flash caches have no attention impl; the spectra_flash op runs them")

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class RecordBackend(_NoImplBackend):
    @staticmethod
    def get_name() -> str:
        return "SPECTRA_FLASH_RECORD"

    @staticmethod
    def get_builder_cls():
        return RecordBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [PAGE]


class WindowBackend(_NoImplBackend):
    @staticmethod
    def get_name() -> str:
        return "SPECTRA_FLASH_WINDOW"

    @staticmethod
    def get_builder_cls():
        return WindowBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(1)]                                   # the ring is one block: never split


# ----------------------------------------------------------------------------------------------- cache owners
class RecordCache(nn.Module, AttentionLayerBase):
    """Owner of one block's record. After binding: `pages` (kernel blocks, 64, 1, 576) fp8 for FlashMLA and `rows`
    (kernel blocks * 64, 576) fp8 for the entry writes."""

    supports_dcp = False

    def __init__(self, prefix: str, dk: int):
        super().__init__()
        vllm_config = get_current_vllm_config()
        self.prefix, self.dk = prefix, dk
        self.kv_cache = torch.tensor([])
        self.pages = self.rows = None
        _register(self, prefix, vllm_config)

    def forward(self):  # pragma: no cover - never called
        ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return RecordBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        half = self.dk // 4                                      # (half + half) bf16 = dk bytes per token
        return FullAttentionSpec(block_size=PAGE, num_kv_heads=1, head_size=half, head_size_v=half,
                                 dtype=torch.bfloat16, num_head_slots=1, state_content_bytes=self.dk)

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache
        if kv_cache.numel() == 0:
            self.pages = self.rows = None
            return
        u8 = kv_cache.view(torch.uint8)                          # (kernel blocks, 1, 64, dk)
        nb = u8.shape[0]
        assert u8.shape[1:] == (1, PAGE, self.dk), u8.shape
        assert u8.stride(3) == 1 and u8.stride(2) == self.dk and u8.stride(0) == PAGE * self.dk, u8.stride()
        f8 = u8.view(torch.float8_e4m3fn)
        self.rows = torch.as_strided(f8, (nb * PAGE, self.dk), (self.dk, 1), f8.storage_offset())
        self.pages = self.rows.view(nb, PAGE, 1, self.dk)
        if "rec" not in _LOGGED:
            _LOGGED.add("rec")
            logger.info("[spectra-flash] record cache %s: %d pages of %d tokens x %d B (view %s, strides %s)",
                        self.prefix, nb, PAGE, self.dk, tuple(kv_cache.shape), tuple(u8.stride()))


class WindowCache(nn.Module, AttentionLayerBase):
    """Owner of one layer's exact window ring. After binding: `k` / `v` (slots, HKV, D) bf16 views of the K|V cache."""

    supports_dcp = False

    def __init__(self, prefix: str, num_kv_heads: int, head_size: int, capacity: int):
        super().__init__()
        vllm_config = get_current_vllm_config()
        self.prefix = prefix
        self.num_kv_heads, self.head_size, self.capacity = num_kv_heads, head_size, capacity
        self.kv_cache = torch.tensor([])
        self.k = self.v = self.k_cache = self.v_cache = None
        _register(self, prefix, vllm_config)

    def forward(self):  # pragma: no cover
        ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return WindowBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return CircularBufferSpec(block_size=self.capacity, num_kv_heads=self.num_kv_heads, head_size=self.head_size,
                                  head_size_v=self.head_size, dtype=torch.bfloat16)

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # logical (blocks, HKV, capacity, 2 D) [B, H, N, C]; physically slot-major (blocks, capacity, HKV, 2 D)
        self.kv_cache = kv_cache
        if kv_cache.numel() == 0:
            self.k = self.v = self.k_cache = self.v_cache = None
            return
        D, nb = self.head_size, kv_cache.shape[0]
        bnhc = kv_cache.transpose(1, 2)                           # (blocks, capacity, HKV, 2 D)
        assert bnhc.stride(3) == 1 and bnhc.stride(2) == 2 * D, bnhc.stride()
        assert bnhc.stride(1) == self.num_kv_heads * 2 * D and bnhc.stride(0) == self.capacity * bnhc.stride(1), \
            bnhc.stride()
        self.k_cache, self.v_cache = bnhc[..., :D], bnhc[..., D:]  # reshape_and_cache_flash targets
        flat = torch.as_strided(kv_cache, (nb * self.capacity, self.num_kv_heads, 2 * D),
                                (bnhc.stride(1), bnhc.stride(2), 1), kv_cache.storage_offset())
        self.k, self.v = flat[..., :D], flat[..., D:]             # tail_merge reads (slot, head, dim)
        if "win" not in _LOGGED:
            _LOGGED.add("win")
            logger.info("[spectra-flash] window cache %s: %d rings of %d slots (view %s, strides %s)",
                        self.prefix, nb, self.capacity, tuple(kv_cache.shape), tuple(kv_cache.stride()))
