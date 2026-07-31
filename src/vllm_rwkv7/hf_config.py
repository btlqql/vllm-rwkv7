"""Local Transformers configuration classes for public RWKV-7 checkpoints."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig


def _initialize_rwkv7_config(
    self: PretrainedConfig,
    *,
    vocab_size: int = 65536,
    hidden_size: int = 768,
    num_hidden_layers: int = 12,
    num_attention_heads: int | None = None,
    num_heads: int | None = None,
    head_dim: int = 64,
    intermediate_size: int | None = None,
    norm_eps: float = 1e-5,
    hidden_act: str = "sqrelu",
    norm_bias: bool = True,
    norm_first: bool = True,
    decay_low_rank_dim: int = 64,
    gate_low_rank_dim: int = 128,
    a_low_rank_dim: int = 64,
    v_low_rank_dim: int = 32,
    tie_word_embeddings: bool = False,
    **kwargs: Any,
) -> None:
    """Initialize one of the model-type-specific local config classes."""

    PretrainedConfig.__init__(self, tie_word_embeddings=tie_word_embeddings, **kwargs)

    if (
        num_attention_heads is not None
        and num_heads is not None
        and num_attention_heads != num_heads
    ):
        raise ValueError("num_attention_heads and num_heads must match when both are set")
    resolved_heads = num_attention_heads if num_attention_heads is not None else num_heads
    if resolved_heads is None:
        if hidden_size % head_dim:
            raise ValueError(
                "num_attention_heads cannot be inferred when hidden_size is not "
                "divisible by head_dim"
            )
        resolved_heads = hidden_size // head_dim

    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = resolved_heads
    self.num_heads = resolved_heads
    self.head_dim = head_dim
    self.intermediate_size = intermediate_size or hidden_size * 4
    self.norm_eps = norm_eps
    self.layer_norm_eps = norm_eps
    self.hidden_act = hidden_act
    self.norm_bias = norm_bias
    self.norm_first = norm_first
    self.decay_low_rank_dim = decay_low_rank_dim
    self.gate_low_rank_dim = gate_low_rank_dim
    self.a_low_rank_dim = a_low_rank_dim
    self.v_low_rank_dim = v_low_rank_dim


class RWKV7Config(PretrainedConfig):
    """Configuration for checkpoints whose model type is ``rwkv7``."""

    model_type = "rwkv7"
    __init__ = _initialize_rwkv7_config


class RWKV7AdapterConfig(PretrainedConfig):
    """Configuration for checkpoints whose model type is ``rwkv7_hf_adapter``."""

    model_type = "rwkv7_hf_adapter"
    __init__ = _initialize_rwkv7_config


class NativeRWKV7Config(PretrainedConfig):
    """Configuration for clean-room native fixtures."""

    model_type = "rwkv7_native"
    __init__ = _initialize_rwkv7_config


CONFIG_CLASSES = {
    RWKV7Config.model_type: RWKV7Config,
    RWKV7AdapterConfig.model_type: RWKV7AdapterConfig,
    NativeRWKV7Config.model_type: NativeRWKV7Config,
}


def register_transformers_configs() -> None:
    """Register configs without replacing an implementation owned elsewhere."""

    for model_type, config_class in CONFIG_CLASSES.items():
        try:
            AutoConfig.for_model(model_type)
        except ValueError:
            AutoConfig.register(model_type, config_class)
