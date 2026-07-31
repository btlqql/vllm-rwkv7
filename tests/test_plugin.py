from __future__ import annotations

import sys
from types import ModuleType

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


def install_fake_vllm(monkeypatch, registry: FakeRegistry) -> None:
    module = ModuleType("vllm")
    module.ModelRegistry = registry
    monkeypatch.setitem(sys.modules, "vllm", module)


def test_register_uses_lazy_model_paths(monkeypatch) -> None:
    registry = FakeRegistry()
    install_fake_vllm(monkeypatch, registry)

    register()

    assert tuple(registry.models) == SUPPORTED_ARCHITECTURES
    assert set(registry.models.values()) == {"vllm_rwkv7.model:RWKV7ForCausalLM"}


def test_register_is_reentrant_and_does_not_replace_existing_models(monkeypatch) -> None:
    registry = FakeRegistry()
    registry.models["RWKV7ForCausalLM"] = "existing.module:ExistingModel"
    install_fake_vllm(monkeypatch, registry)

    register()
    register()

    assert registry.models["RWKV7ForCausalLM"] == "existing.module:ExistingModel"
    assert registry.calls == [("NativeRWKV7ForCausalLM", "vllm_rwkv7.model:RWKV7ForCausalLM")]
