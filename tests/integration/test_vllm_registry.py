from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

vllm = pytest.importorskip("vllm")


def test_installed_plugin_registers_rwkv7_architectures() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models.interfaces import supports_mamba_prefix_caching

    from vllm_rwkv7 import SUPPORTED_ARCHITECTURES
    from vllm_rwkv7.model import RWKV7ForCausalLM
    from vllm_rwkv7.plugin import register

    register()

    registered = set(ModelRegistry.get_supported_archs())
    assert set(SUPPORTED_ARCHITECTURES) <= registered
    assert supports_mamba_prefix_caching(RWKV7ForCausalLM)
