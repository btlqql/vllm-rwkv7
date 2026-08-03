# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from benchmarks.benchmark_rwkv7_quantization import (
    compact_result,
    select_unsupported_reason,
)
from benchmarks.kernels.benchmark_rwkv7 import compact_backend_result
from benchmarks.kernels.benchmark_rwkv7_lora import generation_signature


def test_select_unsupported_reason_prefers_root_cause():
    reason = select_unsupported_reason(
        [
            "RuntimeError: the requested ROCm kernel is unsupported",
            "RuntimeError: Engine core initialization failed. See root cause above.",
        ]
    )

    assert reason == "RuntimeError: the requested ROCm kernel is unsupported"


def test_compact_result_keeps_backend_and_unsupported_reason():
    result = compact_result(
        {
            "supported": False,
            "unsupported_reason": "RuntimeError: unsupported on gfx1100",
            "rwkv7_backend": "triton",
            "error": "verbose worker output",
        }
    )

    assert result == {
        "supported": False,
        "unsupported_reason": "RuntimeError: unsupported on gfx1100",
        "rwkv7_backend": "triton",
    }


def test_lora_signature_includes_logprobs_when_tokens_match():
    generation = SimpleNamespace(
        token_ids=[7, 8],
        logprobs=[
            {7: SimpleNamespace(logprob=-0.25)},
            {8: SimpleNamespace(logprob=-0.5)},
        ],
    )
    outputs = [SimpleNamespace(outputs=[generation])]

    assert generation_signature(outputs) == {
        "token_ids": (7, 8),
        "token_logprobs": (-0.25, -0.5),
    }


def test_kernel_result_omits_full_request_payloads():
    result = compact_backend_result(
        {
            "backend": "triton",
            "output_tok_s": 100.0,
            "requests": [{"token_ids": [1], "logprobs": [[-0.1]]}],
        }
    )

    assert result == {"backend": "triton", "output_tok_s": 100.0}
