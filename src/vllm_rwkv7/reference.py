"""Correctness-first PyTorch implementation of the RWKV-7 state equation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

EXP_NEGATIVE_HALF = math.exp(-0.5)


def _validate_recurrence_shapes(
    *,
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
) -> None:
    vector_shape = r.shape
    if r.ndim != 3:
        raise ValueError(f"r must have shape [batch, heads, head_dim], got {vector_shape}")
    for name, tensor in (
        ("decay_logits", decay_logits),
        ("k", k),
        ("v", v),
        ("kk", kk),
        ("a", a),
    ):
        if tensor.shape != vector_shape:
            raise ValueError(f"{name} must have shape {vector_shape}, got {tensor.shape}")
    expected_state_shape = (*vector_shape, vector_shape[-1])
    if state.shape != expected_state_shape:
        raise ValueError(f"state must have shape {expected_state_shape}, got {tuple(state.shape)}")


def rwkv7_recurrence(
    *,
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one RWKV-7 token update with fp32 state accumulation.

    Inputs are shaped `[batch, heads, head_dim]`; state is
    `[batch, heads, head_dim, head_dim]`. `kk` is normalized here because that
    normalization is part of the public RWKV-7 recurrence contract.
    """

    _validate_recurrence_shapes(
        r=r,
        decay_logits=decay_logits,
        k=k,
        v=v,
        kk=kk,
        a=a,
        state=state,
    )

    output_dtype = r.dtype
    r_fp32 = r.float()
    k_fp32 = k.float()
    v_fp32 = v.float()
    a_fp32 = a.float()
    kk_fp32 = functional.normalize(kk.float(), dim=-1, p=2, eps=1e-12)
    state_fp32 = state.float()

    decay = torch.exp(-EXP_NEGATIVE_HALF * torch.sigmoid(decay_logits.float()))
    removal = torch.matmul(state_fp32, kk_fp32.unsqueeze(-1))
    removal = torch.matmul(removal, (kk_fp32 * a_fp32).unsqueeze(-2))
    addition = torch.matmul(v_fp32.unsqueeze(-1), k_fp32.unsqueeze(-2))
    next_state = state_fp32 * decay.unsqueeze(-2) - removal + addition

    output = torch.matmul(next_state, r_fp32.unsqueeze(-1)).squeeze(-1)
    return output.to(dtype=output_dtype), next_state


def rwkv7_scan(
    *,
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the reference recurrence across `[batch, time, heads, head_dim]`."""

    shape = r.shape
    if r.ndim != 4:
        raise ValueError(f"r must have shape [batch, time, heads, head_dim], got {shape}")
    for name, tensor in (
        ("decay_logits", decay_logits),
        ("k", k),
        ("v", v),
        ("kk", kk),
        ("a", a),
    ):
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tensor.shape}")

    state = initial_state
    outputs = []
    for token_index in range(shape[1]):
        output, state = rwkv7_recurrence(
            r=r[:, token_index],
            decay_logits=decay_logits[:, token_index],
            k=k[:, token_index],
            v=v[:, token_index],
            kk=kk[:, token_index],
            a=a[:, token_index],
            state=state,
        )
        outputs.append(output)

    if not outputs:
        empty = r.new_empty((shape[0], 0, shape[2], shape[3]))
        return empty, state.float()
    return torch.stack(outputs, dim=1), state
