#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create a streaming compressed-tensors W8A16 or W4A16 RWKV7 checkpoint."""

import argparse
import json
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from compressed_tensors.compressors.pack_quantized import (
    PackedQuantizationCompressor,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils import calculate_qparams
from safetensors import safe_open
from safetensors.torch import save_file

RWKV7_FFN_PATTERN = re.compile(
    r"^model\.layers\.\d+\.ffn\.(?P<projection>key|value)\.weight$"
)
RWKV7_ATTENTION_PATTERN = re.compile(
    r"^model\.layers\.\d+\.attn\.(?:r_proj|o_proj)\.weight$"
)
RWKV7_ATTENTION_TARGET = r"re:^model\.layers\.\d+\.attn\.(?:r_proj|o_proj)$"
RWKV7_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")
WEIGHT_FILES = {"model.safetensors", "model.safetensors.index.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(4, 8), required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--quantize-lm-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Quantize lm_head in addition to the stable RWKV7 projections.",
    )
    parser.add_argument(
        "--quantize-attention-projections",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantize r_proj/o_proj; disable for the higher-accuracy FFN policy.",
    )
    parser.add_argument(
        "--ffn-projections",
        nargs="+",
        choices=("key", "value"),
        default=("key", "value"),
        help="FFN projection roles to quantize.",
    )
    parser.add_argument(
        "--exclude-layer",
        action="append",
        type=int,
        default=[],
        help="Keep a zero-based transformer layer in the model dtype; repeatable.",
    )
    parser.add_argument(
        "--max-shard-size-mib",
        type=int,
        default=512,
        help="Maximum accumulated tensor bytes per output shard.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_scheme(
    bits: int,
    group_size: int = 128,
    *,
    targets: list[str] | None = None,
) -> QuantizationScheme:
    if bits == 4:
        if group_size <= 0:
            raise ValueError("W4 group size must be positive")
        strategy = QuantizationStrategy.GROUP
        selected_group_size: int | None = group_size
    elif bits == 8:
        strategy = QuantizationStrategy.CHANNEL
        selected_group_size = None
    else:
        raise ValueError(f"Only W4 and W8 are supported, got {bits}")

    return QuantizationScheme(
        targets=targets or ["Linear"],
        weights=QuantizationArgs(
            num_bits=bits,
            type=QuantizationType.INT,
            symmetric=True,
            group_size=selected_group_size,
            strategy=strategy,
            dynamic=False,
            observer="memoryless_minmax",
        ),
    )


def make_quantization_config(
    scheme: QuantizationScheme,
    *,
    quantize_lm_head: bool,
    ignore: list[str] | None = None,
) -> QuantizationConfig:
    ignored_targets = [] if quantize_lm_head else ["lm_head"]
    ignored_targets.extend(ignore or [])
    return QuantizationConfig(
        config_groups={"group_0": scheme},
        format="pack-quantized",
        quantization_status=QuantizationStatus.COMPRESSED,
        ignore=ignored_targets,
    )


def should_quantize(
    name: str,
    *,
    quantize_lm_head: bool,
    quantize_attention_projections: bool = True,
    ffn_projections: tuple[str, ...] = ("key", "value"),
    excluded_layers: tuple[int, ...] = (),
) -> bool:
    layer_match = RWKV7_LAYER_PATTERN.match(name)
    if layer_match and int(layer_match.group(1)) in excluded_layers:
        return False
    ffn_match = RWKV7_FFN_PATTERN.fullmatch(name)
    return (
        bool(ffn_match and ffn_match.group("projection") in ffn_projections)
        or (
            quantize_attention_projections
            and bool(RWKV7_ATTENTION_PATTERN.fullmatch(name))
        )
        or (quantize_lm_head and name == "lm_head.weight")
    )


def quantize_weight(
    weight: torch.Tensor, scheme: QuantizationScheme
) -> dict[str, torch.Tensor]:
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError(
            "RWKV7 compressed-tensors conversion expects a floating-point "
            f"matrix, got shape={tuple(weight.shape)} dtype={weight.dtype}"
        )
    args = scheme.weights
    assert args is not None

    observed = weight.float()
    if args.strategy == QuantizationStrategy.GROUP:
        assert args.group_size is not None
        if weight.shape[-1] % args.group_size != 0:
            raise ValueError(
                f"Input size {weight.shape[-1]} is not divisible by group size "
                f"{args.group_size}"
            )
        observed = observed.unflatten(
            -1, (weight.shape[-1] // args.group_size, args.group_size)
        )
        min_vals, max_vals = torch.aminmax(observed, dim=-1)
    elif args.strategy == QuantizationStrategy.CHANNEL:
        min_vals, max_vals = torch.aminmax(observed, dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unsupported weight strategy: {args.strategy}")

    scale, zero_point = calculate_qparams(min_vals, max_vals, args)
    compressed = PackedQuantizationCompressor.compress(
        {
            "weight": weight,
            "weight_scale": scale.to(weight.dtype),
            "weight_zero_point": zero_point,
        },
        scheme,
    )
    return {
        name: tensor.contiguous()
        for name, tensor in compressed.items()
        if tensor is not None
    }


def discover_weight_files(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        filenames = list(dict.fromkeys(index["weight_map"].values()))
        files = [model_dir / filename for filename in filenames]
    else:
        single_file = model_dir / "model.safetensors"
        files = (
            [single_file]
            if single_file.exists()
            else sorted(model_dir.glob("*.safetensors"))
        )
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError(f"No complete safetensors checkpoint in {model_dir}")
    return files


def iter_tensors(weight_files: list[Path]) -> Iterator[tuple[str, torch.Tensor]]:
    seen: set[str] = set()
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as source:
            for name in list(source.keys()):
                if name in seen:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                seen.add(name)
                yield name, source.get_tensor(name)


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def prepare_output(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}; pass --overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_model_assets(model_dir: Path, output_dir: Path) -> None:
    for source in model_dir.iterdir():
        if source.name in WEIGHT_FILES or source.suffix in {".safetensors", ".bin"}:
            continue
        destination = output_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        elif source.name != "config.json":
            shutil.copy2(source, destination)


def write_sharded_checkpoint(
    tensors: Iterator[tuple[str, torch.Tensor]],
    output_dir: Path,
    scheme: QuantizationScheme,
    *,
    quantize_lm_head: bool,
    quantize_attention_projections: bool,
    ffn_projections: tuple[str, ...],
    excluded_layers: tuple[int, ...],
    max_shard_size: int,
) -> dict[str, Any]:
    if max_shard_size <= 0:
        raise ValueError("Maximum shard size must be positive")

    pending: dict[str, torch.Tensor] = {}
    pending_bytes = 0
    temporary_shards: list[Path] = []
    weight_map: dict[str, str] = {}
    total_size = 0
    quantized_modules = 0

    def flush() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return
        shard_path = output_dir / f"model-{len(temporary_shards) + 1:05d}.safetensors"
        save_file(pending, shard_path, metadata={"format": "pt"})
        for name in pending:
            weight_map[name] = shard_path.name
        temporary_shards.append(shard_path)
        pending = {}
        pending_bytes = 0

    for name, tensor in tensors:
        if should_quantize(
            name,
            quantize_lm_head=quantize_lm_head,
            quantize_attention_projections=quantize_attention_projections,
            ffn_projections=ffn_projections,
            excluded_layers=excluded_layers,
        ):
            module_name = name.removesuffix(".weight")
            output_tensors = {
                f"{module_name}.{suffix}": value
                for suffix, value in quantize_weight(tensor, scheme).items()
            }
            quantized_modules += 1
        else:
            output_tensors = {name: tensor.contiguous()}

        output_bytes = sum(tensor_bytes(value) for value in output_tensors.values())
        if pending and pending_bytes + output_bytes > max_shard_size:
            flush()
        pending.update(output_tensors)
        pending_bytes += output_bytes
        total_size += output_bytes
    flush()

    shard_count = len(temporary_shards)
    renamed_weight_map = dict(weight_map)
    for index, temporary_path in enumerate(temporary_shards, start=1):
        final_name = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        temporary_path.rename(output_dir / final_name)
        for name, shard_name in weight_map.items():
            if shard_name == temporary_path.name:
                renamed_weight_map[name] = final_name

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": renamed_weight_map,
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "quantized_modules": quantized_modules,
        "tensor_count": len(weight_map),
        "shard_count": shard_count,
        "total_size": total_size,
    }


def convert_model(
    model_dir: Path,
    output_dir: Path,
    *,
    bits: int,
    group_size: int = 128,
    quantize_lm_head: bool = False,
    quantize_attention_projections: bool = True,
    ffn_projections: tuple[str, ...] = ("key", "value"),
    excluded_layers: tuple[int, ...] = (),
    max_shard_size: int = 512 * 1024 * 1024,
    overwrite: bool = False,
) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    output_dir = output_dir.resolve()
    if model_dir == output_dir:
        raise ValueError("Input and output directories must be different")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model config: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures", [])
    if "RWKV7ForCausalLM" not in architectures:
        raise ValueError(f"Expected RWKV7ForCausalLM, got {architectures}")
    excluded_layers = tuple(sorted(set(excluded_layers)))
    if any(layer < 0 for layer in excluded_layers):
        raise ValueError("Excluded layer indices must be non-negative")
    num_hidden_layers = config.get("num_hidden_layers")
    if isinstance(num_hidden_layers, int) and any(
        layer >= num_hidden_layers for layer in excluded_layers
    ):
        raise ValueError(f"Excluded layer indices must be below {num_hidden_layers}")
    ffn_projections = tuple(dict.fromkeys(ffn_projections))
    if not ffn_projections or any(
        projection not in {"key", "value"} for projection in ffn_projections
    ):
        raise ValueError("FFN projections must contain key and/or value")

    ffn_selector = "|".join(ffn_projections)
    ffn_target = rf"re:^model\.layers\.\d+\.ffn\.(?:{ffn_selector})$"
    targets = [ffn_target]
    if quantize_attention_projections:
        targets.append(RWKV7_ATTENTION_TARGET)
    if quantize_lm_head:
        targets.append("lm_head")
    scheme = make_scheme(bits, group_size, targets=targets)
    excluded_projection_names = [f"ffn.{projection}" for projection in ffn_projections]
    if quantize_attention_projections:
        excluded_projection_names.extend(["attn.r_proj", "attn.o_proj"])
    ignored_projections = [
        f"model.layers.{layer}.{projection}"
        for layer in excluded_layers
        for projection in excluded_projection_names
    ]
    quantization_config = make_quantization_config(
        scheme,
        quantize_lm_head=quantize_lm_head,
        ignore=ignored_projections,
    )
    prepare_output(output_dir, overwrite=overwrite)
    copy_model_assets(model_dir, output_dir)
    summary = write_sharded_checkpoint(
        iter_tensors(discover_weight_files(model_dir)),
        output_dir,
        scheme,
        quantize_lm_head=quantize_lm_head,
        quantize_attention_projections=quantize_attention_projections,
        ffn_projections=ffn_projections,
        excluded_layers=excluded_layers,
        max_shard_size=max_shard_size,
    )
    if summary["quantized_modules"] == 0:
        raise ValueError("No RWKV7 projection weights matched the quantization policy")

    config["quantization_config"] = quantization_config.model_dump(mode="json")
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "source": str(model_dir),
        "bits": bits,
        "group_size": group_size if bits == 4 else None,
        "quantize_lm_head": quantize_lm_head,
        "quantize_attention_projections": quantize_attention_projections,
        "ffn_projections": list(ffn_projections),
        "excluded_layers": list(excluded_layers),
        **summary,
    }
    (output_dir / "rwkv7_quantization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = convert_model(
        args.model,
        args.output,
        bits=args.bits,
        group_size=args.group_size,
        quantize_lm_head=args.quantize_lm_head,
        quantize_attention_projections=args.quantize_attention_projections,
        ffn_projections=tuple(args.ffn_projections),
        excluded_layers=tuple(args.exclude_layer),
        max_shard_size=args.max_shard_size_mib * 1024 * 1024,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
