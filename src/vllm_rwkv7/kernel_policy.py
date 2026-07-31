"""Conservative, explicit backend routing by GPU architecture."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

BackendName = str
Capability = tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class KernelPolicy:
    family: str
    backend: BackendName
    allowed_backends: tuple[BackendName, ...]
    fla_prefill_min_tokens: int
    fla_chunk_size: int | None
    reason: str


def detect_gpu_family(*, name: str | None, capability: Capability) -> str:
    normalized_name = (name or "").strip().lower()
    if "amd" in normalized_name or "radeon" in normalized_name or "instinct" in normalized_name:
        return "amd"
    if normalized_name == "cpu":
        return "cpu"
    if capability is None:
        return "unknown"

    major, minor = capability
    if major >= 12:
        return "blackwell"
    if major == 9:
        return "hopper"
    if (major, minor) == (8, 9):
        return "ada"
    if major == 8:
        return "ampere"
    if (major, minor) == (7, 5):
        return "turing"
    if major == 7:
        return "volta"
    if major == 6:
        return "pascal"
    return "legacy_cuda"


_ALLOWED_BY_FAMILY: dict[str, tuple[BackendName, ...]] = {
    "cpu": ("reference",),
    "amd": ("reference",),
    "unknown": ("reference",),
    "legacy_cuda": ("reference",),
    "pascal": ("reference",),
    "volta": ("reference", "fla"),
    "turing": ("reference", "fla"),
    "ampere": ("reference", "fla", "triton"),
    "ada": ("reference", "fla", "triton"),
    "hopper": ("reference", "fla", "triton"),
    "blackwell": ("reference", "fla", "triton"),
}


def _positive_override(environment: Mapping[str, str], name: str) -> int | None:
    raw_value = environment.get(name)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def select_kernel_policy(
    *,
    name: str | None,
    capability: Capability,
    environ: Mapping[str, str] | None = None,
) -> KernelPolicy:
    """Resolve the safe backend, honoring an explicit validated override."""

    family = detect_gpu_family(name=name, capability=capability)
    allowed = _ALLOWED_BY_FAMILY[family]
    environment = os.environ if environ is None else environ
    requested = environment.get("RWKV7_VLLM_BACKEND", "reference").strip().lower()
    if requested not in allowed:
        choices = ", ".join(allowed)
        raise ValueError(
            "RWKV7_VLLM_BACKEND is not supported for "
            f"{family}: {requested!r}; choose one of {choices}"
        )

    prefill_min_tokens = _positive_override(environment, "RWKV7_FLA_PREFILL_MIN_TOKENS") or 64
    chunk_size = _positive_override(environment, "RWKV7_FLA_CHUNK_SIZE")

    reason = (
        "explicit RWKV7_VLLM_BACKEND override"
        if "RWKV7_VLLM_BACKEND" in environment
        else "reference remains default until exact-card end-to-end validation"
    )
    return KernelPolicy(
        family=family,
        backend=requested,
        allowed_backends=allowed,
        fla_prefill_min_tokens=prefill_min_tokens,
        fla_chunk_size=chunk_size,
        reason=reason,
    )
