"""Normalize the public Hugging Face RWKV-7 configuration contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ConfigSource = Mapping[str, Any] | object
_MISSING = object()


def _read(source: ConfigSource, name: str, default: Any = _MISSING) -> Any:
    if isinstance(source, Mapping):
        value = source.get(name, _MISSING)
    else:
        value = getattr(source, name, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise KeyError(name)
        return default
    return value


def _positive_int(name: str, value: Any) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from error
    if resolved <= 0:
        raise ValueError(f"{name} must be a positive integer, got {resolved}")
    return resolved


@dataclass(frozen=True, slots=True)
class RWKV7ModelConfig:
    """Validated model fields consumed by the vLLM implementation.

    `num_attention_heads` is the canonical public name. `num_heads` is exposed
    as a read-only compatibility property for older converted checkpoints.
    """

    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    attention_hidden_size: int
    intermediate_size: int
    layer_norm_epsilon: float
    decay_low_rank_dim: int
    gate_low_rank_dim: int
    a_low_rank_dim: int
    v_low_rank_dim: int
    tie_word_embeddings: bool

    @property
    def num_heads(self) -> int:
        return self.num_attention_heads

    @classmethod
    def from_hf_config(cls, source: ConfigSource) -> RWKV7ModelConfig:
        hidden_size = _positive_int("hidden_size", _read(source, "hidden_size", 768))
        head_dim = _positive_int("head_dim", _read(source, "head_dim", 64))

        canonical_heads = _read(source, "num_attention_heads", None)
        legacy_heads = _read(source, "num_heads", None)
        if (
            canonical_heads is not None
            and legacy_heads is not None
            and int(canonical_heads) != int(legacy_heads)
        ):
            raise ValueError(
                "num_attention_heads and legacy num_heads must match when both are set"
            )

        heads_value = canonical_heads if canonical_heads is not None else legacy_heads
        if heads_value is None:
            if hidden_size % head_dim:
                raise ValueError(
                    "attention_hidden_size cannot be inferred: hidden_size must be "
                    "divisible by head_dim when head counts are omitted"
                )
            heads_value = hidden_size // head_dim
        num_attention_heads = _positive_int("num_attention_heads", heads_value)

        attention_hidden_size = _positive_int(
            "attention_hidden_size",
            _read(source, "attention_hidden_size", hidden_size),
        )
        expected_attention_size = num_attention_heads * head_dim
        if attention_hidden_size != expected_attention_size:
            raise ValueError(
                "attention_hidden_size must equal num_attention_heads * head_dim; "
                f"got {attention_hidden_size} != {num_attention_heads} * {head_dim}"
            )

        layer_norm_epsilon = _read(source, "layer_norm_epsilon", None)
        if layer_norm_epsilon is None:
            layer_norm_epsilon = _read(source, "layer_norm_eps", 1e-5)
        layer_norm_epsilon = float(layer_norm_epsilon)
        if layer_norm_epsilon <= 0:
            raise ValueError("layer_norm_epsilon must be positive")

        return cls(
            model_type=str(_read(source, "model_type", "rwkv7_native")),
            vocab_size=_positive_int("vocab_size", _read(source, "vocab_size", 65536)),
            hidden_size=hidden_size,
            num_hidden_layers=_positive_int(
                "num_hidden_layers", _read(source, "num_hidden_layers", 12)
            ),
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            attention_hidden_size=attention_hidden_size,
            intermediate_size=_positive_int(
                "intermediate_size", _read(source, "intermediate_size", hidden_size * 4)
            ),
            layer_norm_epsilon=layer_norm_epsilon,
            decay_low_rank_dim=_positive_int(
                "decay_low_rank_dim", _read(source, "decay_low_rank_dim", 64)
            ),
            gate_low_rank_dim=_positive_int(
                "gate_low_rank_dim", _read(source, "gate_low_rank_dim", 128)
            ),
            a_low_rank_dim=_positive_int("a_low_rank_dim", _read(source, "a_low_rank_dim", 64)),
            v_low_rank_dim=_positive_int("v_low_rank_dim", _read(source, "v_low_rank_dim", 32)),
            tie_word_embeddings=bool(_read(source, "tie_word_embeddings", False)),
        )
