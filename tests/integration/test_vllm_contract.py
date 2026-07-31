from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.integration

vllm = pytest.importorskip("vllm")


def test_rwkv7_uses_the_pinned_vllm_state_contract() -> None:
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.model_executor.models.interfaces import (
        has_inner_state,
        is_attention_free,
        supports_mamba_prefix_caching,
    )
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    from vllm_rwkv7.model import RWKV7ForCausalLM, RWKV7StatefulBlock

    assert issubclass(RWKV7StatefulBlock, MambaBase)
    assert RWKV7StatefulBlock.mamba_type.fget is not None
    assert RWKV7StatefulBlock.mamba_type.fget(object()) is MambaAttentionBackendEnum.LINEAR
    assert has_inner_state(RWKV7ForCausalLM)
    assert is_attention_free(RWKV7ForCausalLM)
    assert supports_mamba_prefix_caching(RWKV7ForCausalLM)


def test_state_dtype_contract_keeps_matrix_state_in_fp32() -> None:
    from types import SimpleNamespace

    from vllm_rwkv7.model import RWKV7ForCausalLM

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float16),
    )

    assert RWKV7ForCausalLM.get_mamba_state_dtype_from_config(vllm_config) == (
        torch.float16,
        torch.float32,
    )
