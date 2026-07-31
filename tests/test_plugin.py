from __future__ import annotations

import sys
from types import ModuleType

import pytest

from vllm_rwkv7 import SUPPORTED_ARCHITECTURES
from vllm_rwkv7.plugin import register


class FakeRegistry:
    def __init__(self) -> None:
        self.models: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []

    def get_supported_archs(self):
        return self.models.keys()

    def register_model(self, architecture: str, target: str) -> None:
        self.calls.append((architecture, target))
        self.models[architecture] = target


class FakeMambaModelConfig:
    pass


def install_fake_vllm(monkeypatch, registry: FakeRegistry) -> dict[str, type]:
    module = ModuleType("vllm")
    module.ModelRegistry = registry
    model_executor = ModuleType("vllm.model_executor")
    models = ModuleType("vllm.model_executor.models")
    config = ModuleType("vllm.model_executor.models.config")
    config.MODELS_CONFIG_MAP = {}
    config.MambaModelConfig = FakeMambaModelConfig
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", models)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.config", config)
    return config.MODELS_CONFIG_MAP


def test_register_uses_lazy_model_paths(monkeypatch) -> None:
    registry = FakeRegistry()
    config_map = install_fake_vllm(monkeypatch, registry)

    register()

    assert tuple(registry.models) == SUPPORTED_ARCHITECTURES
    assert set(registry.models.values()) == {"vllm_rwkv7.model:RWKV7ForCausalLM"}
    assert config_map == {
        architecture: FakeMambaModelConfig for architecture in SUPPORTED_ARCHITECTURES
    }


def test_register_is_reentrant_and_does_not_replace_existing_models(monkeypatch) -> None:
    registry = FakeRegistry()
    registry.models["RWKV7ForCausalLM"] = "existing.module:ExistingModel"
    config_map = install_fake_vllm(monkeypatch, registry)

    register()
    register()

    assert registry.models["RWKV7ForCausalLM"] == "existing.module:ExistingModel"
    assert registry.calls == [("NativeRWKV7ForCausalLM", "vllm_rwkv7.model:RWKV7ForCausalLM")]
    assert config_map == {"NativeRWKV7ForCausalLM": FakeMambaModelConfig}


def test_register_rejects_a_foreign_config_hook(monkeypatch) -> None:
    registry = FakeRegistry()
    config_map = install_fake_vllm(monkeypatch, registry)
    config_map["RWKV7ForCausalLM"] = object

    with pytest.raises(RuntimeError, match="foreign configuration hook"):
        register()

    assert registry.calls == []
