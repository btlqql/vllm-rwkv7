"""vLLM general-plugin entry point."""

from __future__ import annotations

from vllm_rwkv7 import SUPPORTED_ARCHITECTURES

MODEL_TARGET = "vllm_rwkv7.model:RWKV7ForCausalLM"


def register() -> None:
    """Register RWKV-7 lazily and safely when called in multiple processes."""

    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, MambaModelConfig

    from vllm_rwkv7.hf_config import register_transformers_configs

    register_transformers_configs()

    registered = set(ModelRegistry.get_supported_archs())
    for architecture in SUPPORTED_ARCHITECTURES:
        if architecture in registered:
            continue
        config_owner = MODELS_CONFIG_MAP.get(architecture)
        if config_owner is not None and config_owner is not MambaModelConfig:
            raise RuntimeError(
                f"vLLM architecture {architecture!r} already has a foreign configuration hook"
            )
        ModelRegistry.register_model(architecture, MODEL_TARGET)
        MODELS_CONFIG_MAP[architecture] = MambaModelConfig
