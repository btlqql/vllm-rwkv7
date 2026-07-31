from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_real_vllm_eager_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    model = os.environ.get("RWKV7_VLLM_TEST_MODEL")
    if not model:
        pytest.skip("set RWKV7_VLLM_TEST_MODEL to a local or Hugging Face checkpoint")
    revision = os.environ.get("RWKV7_VLLM_TEST_REVISION")
    gpu_memory_utilization = float(os.environ.get("RWKV7_VLLM_GPU_MEMORY_UTILIZATION", "0.75"))
    if not 0 < gpu_memory_utilization <= 1:
        pytest.fail("RWKV7_VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1]")
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    pytest.importorskip("vllm")
    from vllm import LLM, SamplingParams

    model_kwargs = {"revision": revision} if revision else {}
    llm = LLM(
        model=model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=128,
        max_num_batched_tokens=32,
        max_num_seqs=4,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        skip_tokenizer_init=True,
        **model_kwargs,
    )
    shared_prefix = list(range(1, 97))
    outputs = llm.generate(
        [
            {"prompt_token_ids": shared_prefix + [101]},
            {"prompt_token_ids": shared_prefix + [102]},
        ],
        SamplingParams(temperature=0.0, max_tokens=2),
    )
    reused = llm.generate(
        [{"prompt_token_ids": shared_prefix + [103]}],
        SamplingParams(temperature=0.0, max_tokens=2),
    )
    cached_token_ids = reused[0].outputs[0].token_ids
    assert reused[0].num_cached_tokens is not None
    assert reused[0].num_cached_tokens >= 16

    assert llm.reset_prefix_cache()
    cold = llm.generate(
        [{"prompt_token_ids": shared_prefix + [103]}],
        SamplingParams(temperature=0.0, max_tokens=2),
    )

    assert len(outputs) == 2
    assert len(reused) == 1
    assert len(cold) == 1
    assert cold[0].num_cached_tokens == 0
    assert cold[0].outputs[0].token_ids == cached_token_ids
    for request_output in [*outputs, *reused, *cold]:
        assert len(request_output.outputs) == 1
        assert len(request_output.outputs[0].token_ids) == 2
