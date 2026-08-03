# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.quantize_rwkv7_compressed_tensors import (
    convert_model,
    should_quantize,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.layers.0.attn.r_proj.weight", True),
        ("model.layers.23.attn.o_proj.weight", True),
        ("model.layers.4.ffn.key.weight", True),
        ("model.layers.4.ffn.value.weight", True),
        ("model.layers.0.attn.k_proj.weight", False),
        ("model.layers.0.attn.v_proj.weight", False),
        ("model.layers.0.attn.a_lora.lora.0.weight", False),
        ("lm_head.weight", False),
    ],
)
def test_rwkv7_quantization_policy_preserves_state_update_weights(name, expected):
    assert should_quantize(name, quantize_lm_head=False) is expected


def test_rwkv7_ffn_policy_preserves_attention_projections():
    assert not should_quantize(
        "model.layers.0.attn.r_proj.weight",
        quantize_lm_head=False,
        quantize_attention_projections=False,
    )
    assert should_quantize(
        "model.layers.0.ffn.key.weight",
        quantize_lm_head=False,
        quantize_attention_projections=False,
    )
    assert not should_quantize(
        "model.layers.0.ffn.key.weight",
        quantize_lm_head=False,
        quantize_attention_projections=False,
        excluded_layers=(0, 23),
    )
    assert should_quantize(
        "model.layers.1.ffn.key.weight",
        quantize_lm_head=False,
        quantize_attention_projections=False,
        ffn_projections=("key",),
    )
    assert not should_quantize(
        "model.layers.1.ffn.value.weight",
        quantize_lm_head=False,
        quantize_attention_projections=False,
        ffn_projections=("key",),
    )


@pytest.mark.parametrize(
    ("bits", "group_size", "packed_columns", "scale_columns"),
    [(4, 8, 2, 2), (8, 8, 4, 1)],
)
def test_converter_writes_loadable_pack_quantized_checkpoint(
    tmp_path, bits, group_size, packed_columns, scale_columns
):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"architectures": ["RWKV7ForCausalLM"]}), encoding="utf-8"
    )
    target_name = "model.layers.0.attn.r_proj.weight"
    preserved_name = "model.layers.0.attn.k_proj.weight"
    tensors = {
        target_name: torch.linspace(-1, 1, steps=64).reshape(4, 16).half(),
        preserved_name: torch.arange(64).reshape(4, 16).half(),
        "lm_head.weight": torch.arange(32).reshape(2, 16).half(),
    }
    save_file(tensors, source / "model.safetensors")

    manifest = convert_model(
        source,
        output,
        bits=bits,
        group_size=group_size,
        max_shard_size=256,
    )

    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    packed_name = target_name.removesuffix("weight") + "weight_packed"
    scale_name = target_name.removesuffix("weight") + "weight_scale"
    shape_name = target_name.removesuffix("weight") + "weight_shape"
    assert target_name not in index["weight_map"]
    assert preserved_name in index["weight_map"]
    assert manifest["quantized_modules"] == 1

    loaded = {}
    for shard_name in set(index["weight_map"].values()):
        with safe_open(output / shard_name, framework="pt", device="cpu") as shard:
            loaded.update({name: shard.get_tensor(name) for name in list(shard.keys())})
    assert loaded[packed_name].dtype == torch.int32
    assert loaded[packed_name].shape == (4, packed_columns)
    assert loaded[scale_name].shape == (4, scale_columns)
    torch.testing.assert_close(loaded[shape_name], torch.tensor([4, 16]))
    torch.testing.assert_close(loaded[preserved_name], tensors[preserved_name])

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    weights_config = config["quantization_config"]["config_groups"]["group_0"][
        "weights"
    ]
    assert config["quantization_config"]["format"] == "pack-quantized"
    assert config["quantization_config"]["ignore"] == ["lm_head"]
    assert config["quantization_config"]["config_groups"]["group_0"]["targets"] == [
        r"re:^model\.layers\.\d+\.ffn\.(?:key|value)$",
        r"re:^model\.layers\.\d+\.attn\.(?:r_proj|o_proj)$",
    ]
    assert weights_config["num_bits"] == bits


def test_converter_writes_ffn_only_targets(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"architectures": ["RWKV7ForCausalLM"]}), encoding="utf-8"
    )
    tensors = {
        "model.layers.0.ffn.key.weight": torch.randn(4, 16).half(),
        "model.layers.0.attn.r_proj.weight": torch.randn(4, 16).half(),
        "model.layers.1.ffn.key.weight": torch.randn(4, 16).half(),
    }
    save_file(tensors, source / "model.safetensors")

    manifest = convert_model(
        source,
        output,
        bits=8,
        quantize_attention_projections=False,
        excluded_layers=(0,),
        max_shard_size=256,
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    targets = config["quantization_config"]["config_groups"]["group_0"]["targets"]
    assert targets == [r"re:^model\.layers\.\d+\.ffn\.(?:key|value)$"]
    assert manifest["quantized_modules"] == 1
    assert manifest["excluded_layers"] == [0]
    assert config["quantization_config"]["ignore"] == [
        "lm_head",
        "model.layers.0.ffn.key",
        "model.layers.0.ffn.value",
    ]

    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert "model.layers.0.ffn.key.weight" in index["weight_map"]
    assert "model.layers.1.ffn.key.weight_packed" in index["weight_map"]
    assert "model.layers.0.attn.r_proj.weight" in index["weight_map"]


def test_converter_writes_projection_role_target(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"architectures": ["RWKV7ForCausalLM"]}), encoding="utf-8"
    )
    tensors = {
        "model.layers.0.ffn.key.weight": torch.randn(4, 16).half(),
        "model.layers.0.ffn.value.weight": torch.randn(4, 16).half(),
    }
    save_file(tensors, source / "model.safetensors")

    manifest = convert_model(
        source,
        output,
        bits=4,
        group_size=8,
        quantize_attention_projections=False,
        ffn_projections=("key",),
        max_shard_size=256,
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    targets = config["quantization_config"]["config_groups"]["group_0"]["targets"]
    assert targets == [r"re:^model\.layers\.\d+\.ffn\.(?:key)$"]
    assert manifest["ffn_projections"] == ["key"]
    assert manifest["quantized_modules"] == 1


def test_converter_rejects_excluded_layer_outside_model(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"architectures": ["RWKV7ForCausalLM"], "num_hidden_layers": 1}),
        encoding="utf-8",
    )
    save_file(
        {"model.layers.0.ffn.key.weight": torch.randn(4, 16).half()},
        source / "model.safetensors",
    )

    with pytest.raises(ValueError, match="below 1"):
        convert_model(
            source,
            tmp_path / "output",
            bits=8,
            excluded_layers=(1,),
        )
