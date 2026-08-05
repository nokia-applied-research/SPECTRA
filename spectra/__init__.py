"""SPECTRA — SPECtral Transform with Rate Allocation, for the LLM KV cache.

SPECTRA = SPECtral Transform with Rate Allocation.

Training-free transform coding: refract K/V into a data-aware, energy-ordered latent
spectrum (G-weighted SVD), allocate bits across the spectrum by rate-distortion, and
degrade gracefully to low rank at high compression. W_O is healed in closed form
(reduced-rank regression). No fine-tuning. Reaches sub-2-bit *effective rates* (the regime
where uniform scalar/rotation quantizers collapse).

Easy API (one line):
    import spectra
    model, tok = spectra.compress("meta-llama/Llama-2-7b-hf", ratio="9x")
    spectra.save(model, "out_dir"); model, tok = spectra.load("out_dir")

Lower-level API:
    from spectra import AttnHeads, build_spectra, LatentProjV2
    from spectra.quant import water_filling_bits        # RD-optimal per-channel bit allocation
    from spectra import baselines                         # KIVI/TurboQuant/PolarQuant/SnapKV/...
    from spectra.eval import longbench, niah             # long-context evaluation harnesses
"""
__version__ = "0.1.0"

from .core import AttnHeads
from .method import build_spectra, build_factor, LatentProjV2, cache_frac
from .api import compress, save, load, PRESETS

__all__ = [
    "compress", "save", "load", "PRESETS",
    "AttnHeads", "build_spectra", "build_factor", "LatentProjV2", "cache_frac",
]
