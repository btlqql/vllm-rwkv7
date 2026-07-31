"""vLLM general-plugin entry point."""

from __future__ import annotations

from vllm_rwkv7 import SUPPORTED_ARCHITECTURES

MODEL_TARGET = "vllm_rwkv7.model:RWKV7ForCausalLM"


def register() -> None:
    """Register RWKV-7 lazily and safely when called in multiple processes."""

    from vllm import ModelRegistry

    registered = set(ModelRegistry.get_supported_archs())
    for architecture in SUPPORTED_ARCHITECTURES:
        if architecture not in registered:
            ModelRegistry.register_model(architecture, MODEL_TARGET)
