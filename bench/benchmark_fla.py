#!/usr/bin/env python3
"""Record reproducible FLA correctness and operator timing on one GPU."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from vllm_rwkv7.backends import rwkv7_scan_backend
from vllm_rwkv7.components import RWKV7ReferenceLayer
from vllm_rwkv7.config import RWKV7ModelConfig


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "--id=0",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip().splitlines()[0]


def _cuda_time_ms(operation: Callable[[], Any], *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iterations)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    cosine = functional.cosine_similarity(actual_float.flatten(), expected_float.flatten(), dim=0)
    difference = (actual_float - expected_float).abs()
    return {
        "min_cosine": float(cosine.item()),
        "max_abs_diff": float(difference.max().item()),
        "mean_abs_diff": float(difference.mean().item()),
    }


def _make_recurrence_inputs(sequence_length: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(700 + sequence_length)
    shape = (1, sequence_length, 16, 64)
    return {
        "r": torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16),
        "decay_logits": torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16),
        "k": torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16),
        "v": torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16),
        "kk": torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16),
        "a": torch.sigmoid(
            torch.randn(shape, generator=generator, device="cuda", dtype=torch.float16)
        ),
        "initial_state": torch.randn(
            1, 16, 64, 64, generator=generator, device="cuda", dtype=torch.float32
        ),
    }


def _recurrence_row(sequence_length: int, *, warmup: int, iterations: int) -> dict[str, Any]:
    inputs = _make_recurrence_inputs(sequence_length)

    def run(backend: str):
        return rwkv7_scan_backend(
            **inputs,
            backend=backend,
            fla_prefill_min_tokens=64,
            fla_chunk_size=32,
        )

    reference_output, reference_state = run("reference")
    fla_output, fla_state = run("fla")
    output_metrics = _metrics(fla_output, reference_output)
    state_metrics = _metrics(fla_state, reference_state)
    reference_ms = _cuda_time_ms(lambda: run("reference"), warmup=warmup, iterations=iterations)
    fla_ms = _cuda_time_ms(lambda: run("fla"), warmup=warmup, iterations=iterations)
    return {
        "axis": "rwkv7_recurrence_operator",
        "batch_size": 1,
        "sequence_length": sequence_length,
        "num_attention_heads": 16,
        "head_dim": 64,
        "dtype": "float16",
        "state_dtype": "float32",
        "fla_path": "chunk" if sequence_length >= 64 else "fused_mul_recurrent",
        "fla_prefill_min_tokens": 64,
        "fla_chunk_size": 32,
        "output": output_metrics,
        "state": state_metrics,
        "reference_ms": reference_ms,
        "fla_ms": fla_ms,
        "operator_speedup": reference_ms / fla_ms,
        "fla_tokens_per_second": 1000.0 * sequence_length / fla_ms,
        "passed": (
            output_metrics["min_cosine"] >= 0.9999 and state_metrics["max_abs_diff"] <= 0.02
        ),
    }


def _layer_row(sequence_length: int) -> dict[str, Any]:
    config = RWKV7ModelConfig.from_hf_config(
        {
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "head_dim": 4,
            "attention_hidden_size": 8,
            "intermediate_size": 24,
            "decay_low_rank_dim": 4,
            "gate_low_rank_dim": 4,
            "a_low_rank_dim": 4,
            "v_low_rank_dim": 4,
        }
    )
    reference_layer = RWKV7ReferenceLayer(config, 1, backend="reference").cuda().half().eval()
    fla_layer = (
        RWKV7ReferenceLayer(
            config,
            1,
            backend="fla",
            fla_prefill_min_tokens=64,
            fla_chunk_size=32,
        )
        .cuda()
        .half()
        .eval()
    )
    fla_layer.load_state_dict(reference_layer.state_dict())
    generator = torch.Generator(device="cuda").manual_seed(900 + sequence_length)
    hidden_states = torch.randn(
        1,
        sequence_length,
        config.hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    previous_attention = torch.randn(
        1,
        config.hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    previous_ffn = torch.randn(
        1,
        config.hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    value_first = torch.randn(
        1,
        sequence_length,
        config.attention_hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    matrix_state = torch.randn(
        1,
        config.num_attention_heads,
        config.head_dim,
        config.head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    arguments = (
        hidden_states,
        previous_attention,
        previous_ffn,
        value_first,
        matrix_state,
    )
    with torch.inference_mode():
        reference_result = reference_layer.forward_sequence(*arguments)
        fla_result = fla_layer.forward_sequence(*arguments)
    output_metrics = _metrics(fla_result[0], reference_result[0])
    state_metrics = _metrics(fla_result[4], reference_result[4])
    return {
        "axis": "rwkv7_complete_layer_correctness",
        "batch_size": 1,
        "sequence_length": sequence_length,
        "hidden_size": config.hidden_size,
        "attention_hidden_size": config.attention_hidden_size,
        "dtype": "float16",
        "state_dtype": "float32",
        "fla_path": "chunk" if sequence_length >= 64 else "fused_mul_recurrent",
        "output": output_metrics,
        "state": state_metrics,
        "passed": (
            output_metrics["min_cosine"] >= 0.9999 and state_metrics["max_abs_diff"] <= 0.02
        ),
    }


def _environment() -> dict[str, Any]:
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_memory_mib": round(properties.total_memory / 1024**2),
        "driver": _driver_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": _version("triton"),
        "fla": _version("fla"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA or ROCm is required")
    if args.warmup < 1 or args.iterations < 1:
        parser.error("--warmup and --iterations must be positive")

    result = {
        "schema_version": 1,
        "scope": "operator_correctness_and_microbenchmark_only",
        "end_to_end_vllm_claim": False,
        "vllm_interface_baseline": "837eae64580c885101ee95b073aafb27a485e7ce",
        "correctness_thresholds": {
            "output_min_cosine": 0.9999,
            "state_max_abs_diff": 0.02,
        },
        "environment": _environment(),
        "rows": [
            _recurrence_row(1, warmup=args.warmup, iterations=args.iterations),
            _recurrence_row(64, warmup=args.warmup, iterations=args.iterations),
            _recurrence_row(512, warmup=args.warmup, iterations=args.iterations),
            _layer_row(1),
            _layer_row(64),
        ],
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8", newline="\n")
    print(serialized)
    return 0 if all(row["passed"] for row in result["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
