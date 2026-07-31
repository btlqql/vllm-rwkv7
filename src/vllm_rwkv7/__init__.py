"""Public package metadata for the RWKV-7 vLLM plugin."""

from __future__ import annotations

__version__ = "0.1.0a0"

SUPPORTED_ARCHITECTURES = (
    "RWKV7ForCausalLM",
    "NativeRWKV7ForCausalLM",
)

__all__ = ["SUPPORTED_ARCHITECTURES", "__version__"]
