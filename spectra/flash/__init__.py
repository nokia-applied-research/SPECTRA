"""SPECTRA-Flash: one FlashMLA-native record per layer and token (512 latent + 64 RoPE dims, fp8).

Post-training conversion: the base model stays frozen; the record maps are initialized in closed form (an
activation-weighted SVD, as in SPECTRA) and fitted by block-wise distillation on calibration text (WikiText-2 train +
Python code), 2.6 GPU hours for an 8B model. Decode then reads one fp8 record per layer through FlashMLA, plus a small
exact window (4 sink + 123 recent tokens).

    spectra.flash.enable(model, "<params dir or HF Hub repo>")      # 🤗 Transformers
    spectra-flash convert --model meta-llama/Llama-3.1-8B-Instruct --out spectra-flash-llama31-8b --gpus 0,1,2,3
"""
from .config import FlashConfig
from . import params


def enable(model, params, **kwargs):
    """`spectra.flash.hf.enable`: SPECTRA-Flash attention for a HuggingFace Llama model (see its docstring)."""
    from .hf import enable as _enable
    return _enable(model, params, **kwargs)


def disable(model):
    from .hf import disable as _disable
    return _disable(model)


__all__ = ["FlashConfig", "params", "enable", "disable"]
