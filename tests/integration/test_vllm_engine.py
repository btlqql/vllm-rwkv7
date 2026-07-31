from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


def test_real_vllm_eager_generation() -> None:
    model = os.environ.get("RWKV7_VLLM_TEST_MODEL")
    if not model:
        pytest.skip("set RWKV7_VLLM_TEST_MODEL to a local or Hugging Face checkpoint")

    pytest.importorskip("vllm")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=32,
    )
    outputs = llm.generate(
        ["Hello"],
        SamplingParams(temperature=0.0, max_tokens=2),
    )

    assert len(outputs) == 1
    assert len(outputs[0].outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) == 2
