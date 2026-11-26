"""SPECTRA: SPECtral Transform with Rate Allocation, for the LLM KV cache.

Training-free transform coding: each layer's keys and values are cached as an energy-ordered latent (a G-weighted
SVD of k_proj / v_proj), bits are allocated across the latent by reverse water-filling, and o_proj is healed in
closed form. No fine-tuning.

    import spectra
    model, tok = spectra.compress("NousResearch/Meta-Llama-3.1-8B-Instruct", preset="8.8x")
    spectra.save(model, "out_dir"); model, tok = spectra.load("out_dir")

Lower-level pieces:
    from spectra import AttnHeads, build_spectra, LatentProjV2
    from spectra.quant import water_filling_bits          # rate-distortion bit allocation
    from spectra.accounting import honest_ratio           # compression ratio with the exact window counted
    from spectra.eval import longbench, niah              # the paper's evaluation protocols
"""
__version__ = "0.3.0"

from .core import AttnHeads
from .method import build_spectra, build_factor, set_exact_window, LatentProjV2, LatentProjWF, cache_frac
from .api import compress, save, load, PRESETS

__all__ = [
    "compress", "save", "load", "PRESETS",
    "AttnHeads", "build_spectra", "build_factor", "set_exact_window", "LatentProjV2", "LatentProjWF", "cache_frac",
]
