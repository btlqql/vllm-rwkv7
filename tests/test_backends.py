from __future__ import annotations

import torch
import torch.nn.functional as functional

from vllm_rwkv7.backends import FLAOperations, rwkv7_scan_backend
from vllm_rwkv7.reference import EXP_NEGATIVE_HALF


def make_inputs(time: int):
    shape = (1, time, 2, 4)
    generator = torch.Generator().manual_seed(time)
    return {
        "r": torch.randn(shape, generator=generator),
        "decay_logits": torch.randn(shape, generator=generator),
        "k": torch.randn(shape, generator=generator),
        "v": torch.randn(shape, generator=generator),
        "kk": torch.randn(shape, generator=generator),
        "a": torch.sigmoid(torch.randn(shape, generator=generator)),
        "initial_state": torch.randn(1, 2, 4, 4, generator=generator),
    }


def test_fla_decode_adapter_uses_log_decay_and_transposed_state() -> None:
    inputs = make_inputs(1)
    captured = {}

    def fake_fused(**kwargs):
        captured.update(kwargs)
        return kwargs["r"].clone(), kwargs["initial_state"] + 1

    operations = FLAOperations(fused_mul_recurrent=fake_fused, chunk=None)
    output, final_state = rwkv7_scan_backend(
        **inputs,
        backend="fla",
        fla_operations=operations,
    )

    expected_log_decay = -EXP_NEGATIVE_HALF * torch.sigmoid(inputs["decay_logits"].float())
    torch.testing.assert_close(captured["w"], expected_log_decay)
    torch.testing.assert_close(captured["initial_state"], inputs["initial_state"].transpose(-1, -2))
    torch.testing.assert_close(output, inputs["r"])
    torch.testing.assert_close(
        final_state,
        (inputs["initial_state"].transpose(-1, -2) + 1).transpose(-1, -2),
    )
    assert captured["scale"] == 1.0
    assert captured["output_final_state"] is True


def test_fla_prefill_adapter_uses_public_dplr_factors() -> None:
    inputs = make_inputs(64)
    captured = {}

    def fake_chunk(**kwargs):
        captured.update(kwargs)
        return kwargs["r"].clone(), kwargs["initial_state"]

    operations = FLAOperations(fused_mul_recurrent=None, chunk=fake_chunk)
    output, final_state = rwkv7_scan_backend(
        **inputs,
        backend="fla",
        fla_operations=operations,
        fla_prefill_min_tokens=64,
        fla_chunk_size=32,
    )

    normalized = functional.normalize(inputs["kk"].float(), dim=-1, p=2, eps=1e-12)
    torch.testing.assert_close(captured["a"], -normalized)
    torch.testing.assert_close(captured["b"], normalized * inputs["a"].float())
    torch.testing.assert_close(output, inputs["r"])
    torch.testing.assert_close(final_state, inputs["initial_state"])
    assert captured["safe_gate"] is True
    assert captured["chunk_size"] == 32


def test_reference_backend_matches_reference_scan() -> None:
    inputs = make_inputs(5)
    output, final_state = rwkv7_scan_backend(**inputs, backend="reference")

    assert output.shape == inputs["r"].shape
    assert final_state.shape == inputs["initial_state"].shape
    assert final_state.dtype == torch.float32
