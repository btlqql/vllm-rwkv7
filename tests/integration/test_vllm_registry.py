from __future__ import annotations

import pytest

vllm = pytest.importorskip("vllm")


def test_installed_plugin_registers_rwkv7_architectures() -> None:
    from vllm import ModelRegistry

    from vllm_rwkv7 import SUPPORTED_ARCHITECTURES
    from vllm_rwkv7.plugin import register

    register()

    registered = set(ModelRegistry.get_supported_archs())
    assert set(SUPPORTED_ARCHITECTURES) <= registered
