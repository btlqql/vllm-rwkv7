from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_rwkv7.config import RWKV7ModelConfig


def test_canonical_transformers_head_names_are_preserved() -> None:
    config = RWKV7ModelConfig.from_hf_config(
        {
            "model_type": "rwkv7_native",
            "vocab_size": 100,
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "head_dim": 64,
            "attention_hidden_size": 128,
            "intermediate_size": 512,
        }
    )

    assert config.num_attention_heads == 2
    assert config.num_heads == 2
    assert config.head_dim == 64
    assert config.attention_hidden_size == 128
    assert config.hidden_size == 256


def test_legacy_num_heads_alias_is_normalized() -> None:
    config = RWKV7ModelConfig.from_hf_config(
        SimpleNamespace(hidden_size=128, head_dim=32, num_heads=4)
    )

    assert config.num_attention_heads == 4
    assert config.num_heads == 4
    assert config.attention_hidden_size == 128


def test_defaults_follow_hf_adapter_contract() -> None:
    config = RWKV7ModelConfig.from_hf_config({})

    assert config.vocab_size == 65536
    assert config.hidden_size == 768
    assert config.num_hidden_layers == 12
    assert config.num_attention_heads == 12
    assert config.head_dim == 64
    assert config.intermediate_size == 3072
    assert config.layer_norm_epsilon == pytest.approx(1e-5)


def test_conflicting_head_aliases_are_rejected() -> None:
    with pytest.raises(ValueError, match="num_attention_heads.*num_heads"):
        RWKV7ModelConfig.from_hf_config(
            {
                "hidden_size": 128,
                "head_dim": 32,
                "num_attention_heads": 4,
                "num_heads": 2,
            }
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"hidden_size": 0}, "hidden_size"),
        (
            {"hidden_size": 128, "head_dim": 32, "num_attention_heads": 3},
            "attention_hidden_size",
        ),
        (
            {
                "hidden_size": 128,
                "head_dim": 32,
                "num_attention_heads": 4,
                "attention_hidden_size": 96,
            },
            "num_attention_heads.*head_dim",
        ),
    ],
)
def test_invalid_shapes_fail_early(values: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RWKV7ModelConfig.from_hf_config(values)
