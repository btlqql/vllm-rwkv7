# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.benchmark_rwkv7_quantization import (
    _setting_model,
    compact_result,
    compare_with_reference,
    load_prompts,
    repeat_mismatch_is_tied,
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


def test_compressed_tensors_settings_require_explicit_models():
    args = SimpleNamespace(
        model="reference",
        ct_w8_model="rwkv-w8",
        ct_w4_model="rwkv-w4",
    )

    assert _setting_model(args, "ct-w8") == ("rwkv-w8", False)
    assert _setting_model(args, "ct-w4") == ("rwkv-w4", False)


def test_quant_comparison_scores_fixed_prompt_when_greedy_tokens_diverge():
    reference = {
        "requests": [[1, 2]],
        "generated_token_logprobs": [[-0.1, -0.2]],
        "prompt_token_logprobs": [[None, -1.0, -2.0]],
        "model_memory_gib": 2.0,
        "output_tok_s": 100.0,
        "dtype": "half",
    }
    candidate = {
        "setting": "ct-w8",
        "requests": [[8, 9]],
        "generated_token_logprobs": [[-0.3, -0.4]],
        "prompt_token_logprobs": [[None, -1.1, -1.8]],
        "model_memory_gib": 1.0,
        "output_tok_s": 120.0,
    }

    comparison = compare_with_reference(reference, candidate)

    assert comparison["token_agreement"] == 0.0
    assert comparison["prompt_logprob_mean_abs_error"] == pytest.approx(0.15)
    assert comparison["prompt_perplexity_ratio"] == pytest.approx(math.exp(-0.05))


def test_load_prompts_accepts_token_id_corpus(tmp_path):
    corpus = tmp_path / "prompts.json"
    corpus.write_text("[[1, 2, 3], [4, 5]]", encoding="utf-8")
    args = SimpleNamespace(
        warmup_runs=0,
        repeats=1,
        prompt_token_length=[],
        prompt_token_ids_file=corpus,
        prompt=[],
        prompts_file=None,
    )

    assert load_prompts(args) == [
        {"prompt_token_ids": [1, 2, 3]},
        {"prompt_token_ids": [4, 5]},
    ]


def test_rwkv7_quantization_corpus_has_fixed_b8_p128_shape():
    corpus = (
        Path(__file__).parents[2]
        / "benchmarks/data/rwkv7_quantization_prompts_v1.tokens"
    )
    prompts = json.loads(corpus.read_text(encoding="utf-8"))

    assert len(prompts) == 8
    assert all(len(prompt) == 128 for prompt in prompts)
    assert all(0 <= token_id < 65536 for prompt in prompts for token_id in prompt)


def test_repeat_mismatch_tolerance_only_accepts_two_small_margins():
    tied = {"reference_margin": 0.0008, "candidate_margin": 0.009}
    missing = {"reference_margin": 0.0008, "candidate_margin": None}

    assert repeat_mismatch_is_tied(tied, 0.01)
    assert not repeat_mismatch_is_tied(tied, 0.0)
    assert not repeat_mismatch_is_tied(tied, 0.005)
    assert not repeat_mismatch_is_tied(missing, 0.01)


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
