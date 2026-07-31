from __future__ import annotations

import math

import pytest
import torch

from vllm_rwkv7.reference import rwkv7_recurrence, rwkv7_scan


def test_scalar_recurrence_matches_public_equation() -> None:
    r = torch.tensor([[[2.0]]])
    decay_logits = torch.tensor([[[0.0]]])
    k = torch.tensor([[[3.0]]])
    v = torch.tensor([[[5.0]]])
    kk = torch.tensor([[[4.0]]])
    a = torch.tensor([[[0.25]]])
    state = torch.tensor([[[[7.0]]]])

    output, next_state = rwkv7_recurrence(
        r=r,
        decay_logits=decay_logits,
        k=k,
        v=v,
        kk=kk,
        a=a,
        state=state,
    )

    decay = math.exp(-math.exp(-0.5) * 0.5)
    expected_state = 7.0 * decay - 7.0 * 0.25 + 15.0
    assert next_state.item() == pytest.approx(expected_state, rel=1e-6)
    assert output.item() == pytest.approx(expected_state * 2.0, rel=1e-6)


def test_scan_equals_repeated_token_updates() -> None:
    generator = torch.Generator().manual_seed(7)
    shape = (2, 5, 3, 4)
    r = torch.randn(shape, generator=generator)
    decay_logits = torch.randn(shape, generator=generator)
    k = torch.randn(shape, generator=generator)
    v = torch.randn(shape, generator=generator)
    kk = torch.randn(shape, generator=generator)
    a = torch.sigmoid(torch.randn(shape, generator=generator))
    initial = torch.randn(2, 3, 4, 4, generator=generator)

    outputs, final_state = rwkv7_scan(
        r=r,
        decay_logits=decay_logits,
        k=k,
        v=v,
        kk=kk,
        a=a,
        initial_state=initial,
    )

    expected_outputs = []
    expected_state = initial
    for token_index in range(shape[1]):
        token_output, expected_state = rwkv7_recurrence(
            r=r[:, token_index],
            decay_logits=decay_logits[:, token_index],
            k=k[:, token_index],
            v=v[:, token_index],
            kk=kk[:, token_index],
            a=a[:, token_index],
            state=expected_state,
        )
        expected_outputs.append(token_output)

    torch.testing.assert_close(outputs, torch.stack(expected_outputs, dim=1))
    torch.testing.assert_close(final_state, expected_state)


def test_fp16_inputs_keep_fp32_state_accumulation() -> None:
    inputs = torch.ones(1, 2, 4, dtype=torch.float16)
    state = torch.zeros(1, 2, 4, 4, dtype=torch.float32)

    output, next_state = rwkv7_recurrence(
        r=inputs,
        decay_logits=inputs,
        k=inputs,
        v=inputs,
        kk=inputs,
        a=inputs,
        state=state,
    )

    assert output.dtype == torch.float16
    assert next_state.dtype == torch.float32


def test_recurrence_rejects_invalid_shapes() -> None:
    vector = torch.ones(1, 2, 4)
    with pytest.raises(ValueError, match="state"):
        rwkv7_recurrence(
            r=vector,
            decay_logits=vector,
            k=vector,
            v=vector,
            kk=vector,
            a=vector,
            state=torch.zeros(1, 2, 4, 3),
        )
