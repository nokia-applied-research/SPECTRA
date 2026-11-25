"""SPECTRA-Flash for vLLM 0.29 (out-of-tree plugin, no fork).

A stock Llama-3.1 checkpoint plus the Flash params, selected per model with --hf-overrides:

  vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 32768 --no-enable-prefix-caching \\
      --no-enable-chunked-prefill --max-num-batched-tokens 32768 \\
      --hf-overrides '{"architectures": ["SpectraFlashLlamaForCausalLM"], "spectra_flash": "<params dir | repo>"}'

The `vllm.general_plugins` entry point (`spectra.flash.vllm:plugin`) calls `register()` in every vLLM process. It only adds
an architecture name and a custom op, so other models are unaffected.

Prefill, two modes:
  exact   (the evaluated config): each prompt in one step, --no-enable-chunked-prefill and --max-num-batched-tokens at
          least the longest prompt;
  chunked (vLLM's default scheduling): a prompt longer than --max-num-batched-tokens runs in chunks, and each later
          chunk sees the earlier ones through the record and the exact window (an approximation).
v0.2 limits: tensor_parallel_size 1, bfloat16, Hopper (FlashMLA fp8), no prefix caching, no speculative decoding, no KV
connectors.
"""
from __future__ import annotations

ARCH = "SpectraFlashLlamaForCausalLM"
_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.config.compilation import CompilationConfig
    from vllm.model_executor.models.registry import ModelRegistry

    from .op import QUALIFIED_OP_NAME, register_op

    register_op()
    if QUALIFIED_OP_NAME not in CompilationConfig._attention_ops:
        CompilationConfig._attention_ops.append(QUALIFIED_OP_NAME)       # a piecewise splitting op
    ModelRegistry.register_model(ARCH, "spectra.flash.vllm.model:SpectraFlashLlamaForCausalLM")
    _REGISTERED = True


def plugin() -> None:
    """vllm.general_plugins entry point: registers the SPECTRA-Flash architecture (only a name, selected with
    --hf-overrides). Never breaks vLLM for other models."""
    try:
        register()
    except Exception as e:  # noqa: BLE001
        import logging
        logging.getLogger("vllm.spectra").warning("SPECTRA-Flash plugin not registered: %r", e)
