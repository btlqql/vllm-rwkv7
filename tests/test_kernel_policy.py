from __future__ import annotations

import pytest

from vllm_rwkv7.kernel_policy import detect_gpu_family, select_kernel_policy


@pytest.mark.parametrize(
    ("name", "capability", "expected"),
    [
        ("Tesla V100-PCIE-32GB", (7, 0), "volta"),
        ("NVIDIA T4", (7, 5), "turing"),
        ("NVIDIA A100-SXM4-80GB", (8, 0), "ampere"),
        ("NVIDIA RTX 4090", (8, 9), "ada"),
        ("NVIDIA H100", (9, 0), "hopper"),
        ("NVIDIA RTX 5090", (12, 0), "blackwell"),
        ("AMD Radeon PRO W7900", None, "amd"),
        ("cpu", None, "cpu"),
    ],
)
def test_gpu_family_detection(name, capability, expected) -> None:
    assert detect_gpu_family(name=name, capability=capability) == expected


def test_unvalidated_cards_default_to_reference_backend() -> None:
    policy = select_kernel_policy(name="NVIDIA RTX 4090", capability=(8, 9))

    assert policy.family == "ada"
    assert policy.backend == "reference"
    assert "triton" in policy.allowed_backends


def test_environment_override_is_explicit_and_validated() -> None:
    policy = select_kernel_policy(
        name="Tesla V100-PCIE-32GB",
        capability=(7, 0),
        environ={"RWKV7_VLLM_BACKEND": "fla"},
    )
    assert policy.backend == "fla"
    assert policy.fla_prefill_min_tokens == 64
    assert policy.fla_chunk_size is None

    with pytest.raises(ValueError, match="RWKV7_VLLM_BACKEND"):
        select_kernel_policy(
            name="Tesla V100-PCIE-32GB",
            capability=(7, 0),
            environ={"RWKV7_VLLM_BACKEND": "triton"},
        )


def test_fla_shape_tuning_is_an_exact_card_override() -> None:
    policy = select_kernel_policy(
        name="NVIDIA RTX 4090",
        capability=(8, 9),
        environ={
            "RWKV7_VLLM_BACKEND": "fla",
            "RWKV7_FLA_PREFILL_MIN_TOKENS": "48",
            "RWKV7_FLA_CHUNK_SIZE": "32",
        },
    )

    assert policy.fla_prefill_min_tokens == 48
    assert policy.fla_chunk_size == 32

    with pytest.raises(ValueError, match="RWKV7_FLA_CHUNK_SIZE"):
        select_kernel_policy(
            name="NVIDIA RTX 4090",
            capability=(8, 9),
            environ={"RWKV7_FLA_CHUNK_SIZE": "0"},
        )
