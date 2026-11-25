"""SPECTRA-Flash settings. Defaults are the reference conversion (Llama-3.1-8B-Instruct, 2026-11)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .calibration import CODE_SOURCES

# The reference conversion used the open codeparrot split for code (the-stack-smol skipped), so that is pinned here.
FLASH_CODE_SOURCES = tuple(s for s in CODE_SOURCES if s[0] == "codeparrot/codeparrot-clean-valid")


# deployed exact window: 4 sink + 123 recent + the current token = 128 keys, one tail_merge chunk, so it is free at
# decode. Written into every converted params dir; the conversion itself trains with `recent` (64 for Lite).
DEPLOY_RECENT = 123


@dataclass
class FlashConfig:
    # deployment shape (FlashMLA-native record: 512 latent + 64 rope dims, fp8)
    group_size: int = 1            # n: layers sharing one record (1: one record per layer)
    r_latent: int = 512            # latent dims (FlashMLA's value / first 512 of the 576 key dims)
    r_rope: int = 64               # rope dims (32 RoPE pairs)
    sink: int = 4                  # first tokens kept exact
    recent: int = 64               # most recent tokens kept exact
    # calibration (WikiText-2 train + Python code, long windows)
    calib_seq: int = 32768
    calib_windows: int = 16        # training windows (half text, half code)
    calib_heldout: int = 2         # one text + one code window, for the error report only
    # block-wise distillation (long-key, short-query; base model frozen)
    steps: int = 1500
    queries: int = 1024            # query positions per step; each attends all keys of its window
    lr: float = 1e-4
    rope_horizon: int = 32768      # distance scale of the closed-form rope ranking
    query_weighted: bool = True    # weight the rope ranking by query energy
    heldout_queries: int = 1024
    seed: int = 0                  # per block: seed + first layer of the block
    # fp8 scales (static): headroom over the calibration max
    record_headroom: float = 2.0
    query_headroom: float = 2.0
    # decode-phase calibration data (off by default): the model greedily continues some windows, so the record also
    # learns to compress the model's own generated tokens (in deployment they enter the record once they leave the window)
    decode_frac: float = 0.0       # share of training windows (text and code alike) that end in a generated span
    decode_tokens: int = 1024      # generated tokens at the end of such a window (window length unchanged)
    decode_query_frac: float = 0.5 # share of each step's queries drawn from the generated span, when there is one
    decode_batch: int = 4          # windows generated together

    @classmethod
    def preset(cls, name: str) -> "FlashConfig":
        """lite: the reference conversion of the released SPECTRA-Flash params."""
        if name == "lite":
            return cls()
        raise ValueError(f"unknown preset {name!r} (lite)")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def record_bytes_per_token(self, fp8: bool = True) -> int:
        """Bytes per token per block of n layers (record only; the exact sink/recent window is extra)."""
        return (self.r_latent + self.r_rope) * (1 if fp8 else 2)
