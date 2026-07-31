"""Backend adapters around the independently implemented RWKV-7 equation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from vllm_rwkv7.reference import EXP_NEGATIVE_HALF, rwkv7_scan

FLAOperation = Callable[..., tuple[torch.Tensor, torch.Tensor | None]]


@dataclass(frozen=True, slots=True)
class FLAOperations:
    fused_mul_recurrent: FLAOperation | None
    chunk: FLAOperation | None


def load_fla_operations() -> FLAOperations:
    try:
        from fla.ops.rwkv7 import chunk_rwkv7, fused_mul_recurrent_rwkv7
    except ImportError as error:
        raise RuntimeError(
            "RWKV7_VLLM_BACKEND=fla requires flash-linear-attention with RWKV-7 ops"
        ) from error
    return FLAOperations(
        fused_mul_recurrent=fused_mul_recurrent_rwkv7,
        chunk=chunk_rwkv7,
    )


def _fla_scan(
    *,
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor,
    operations: FLAOperations | None,
    prefill_min_tokens: int,
    chunk_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    injected_operations = operations is not None
    operations = load_fla_operations() if operations is None else operations
    if not injected_operations and not r.is_cuda:
        raise RuntimeError("the FLA RWKV-7 backend requires CUDA or ROCm tensors")

    log_decay = -EXP_NEGATIVE_HALF * torch.sigmoid(decay_logits.float())
    normalized_kk = functional.normalize(kk.float(), dim=-1, p=2, eps=1e-12).to(kk.dtype)
    fla_initial_state = initial_state.float().transpose(-1, -2).contiguous()

    if r.shape[1] >= prefill_min_tokens and operations.chunk is not None:
        output, fla_final_state = operations.chunk(
            r=r,
            w=log_decay,
            k=k,
            v=v,
            a=-normalized_kk,
            b=normalized_kk * a,
            scale=1.0,
            initial_state=fla_initial_state,
            output_final_state=True,
            safe_gate=True,
            chunk_size=chunk_size,
        )
    else:
        if operations.fused_mul_recurrent is None:
            raise RuntimeError("the installed FLA package lacks fused_mul_recurrent_rwkv7")
        output, fla_final_state = operations.fused_mul_recurrent(
            r=r,
            w=log_decay,
            k=k,
            v=v,
            kk=normalized_kk,
            a=a,
            scale=1.0,
            initial_state=fla_initial_state,
            output_final_state=True,
        )

    if fla_final_state is None:
        raise RuntimeError("FLA RWKV-7 backend did not return the requested final state")
    return output.to(r.dtype), fla_final_state.float().transpose(-1, -2).contiguous()


def rwkv7_scan_backend(
    *,
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    initial_state: torch.Tensor,
    backend: str,
    fla_operations: FLAOperations | None = None,
    fla_prefill_min_tokens: int = 64,
    fla_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a full sequence through the selected recurrence backend."""

    if backend == "reference":
        return rwkv7_scan(
            r=r,
            decay_logits=decay_logits,
            k=k,
            v=v,
            kk=kk,
            a=a,
            initial_state=initial_state,
        )
    if backend == "fla":
        return _fla_scan(
            r=r,
            decay_logits=decay_logits,
            k=k,
            v=v,
            kk=kk,
            a=a,
            initial_state=initial_state,
            operations=fla_operations,
            prefill_min_tokens=fla_prefill_min_tokens,
            chunk_size=fla_chunk_size,
        )
    if backend == "triton":
        raise NotImplementedError("the native Triton RWKV-7 backend is not implemented yet")
    raise ValueError(f"unknown RWKV-7 backend: {backend!r}")
