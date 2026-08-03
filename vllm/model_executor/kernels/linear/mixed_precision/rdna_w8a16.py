# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton W8A16 weight-only GEMM for AMD RDNA GPUs."""

import torch

from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


@triton.jit
def _rdna_w8a16_kernel(
    a_ptr,
    b_ptr,
    scales_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    scales = tl.load(scales_ptr + offs_n, mask=offs_n < N, other=1.0)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    shifts_row = tl.arange(0, 4) * 8
    shifts = tl.reshape(
        tl.broadcast_to(shifts_row[None, :], (BLOCK_K // 4, 4)),
        (BLOCK_K,),
    )
    shifts = tl.broadcast_to(shifts[None, :], (BLOCK_N, BLOCK_K))

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        offs_k4 = k_start * (BLOCK_K // 4) + tl.arange(0, BLOCK_K // 4)
        packed = tl.load(
            b_ptr + offs_n[:, None] * stride_bn + offs_k4[None, :] * stride_bk,
            mask=(offs_n[:, None] < N) & (offs_k4[None, :] < K // 4),
            other=0,
        )
        unpacked = tl.interleave(packed, packed)
        unpacked = tl.interleave(unpacked, unpacked)
        unpacked = (unpacked >> shifts) & 0xFF
        weights = (unpacked - 128).to(a.dtype) * scales[:, None]
        accumulator += tl.dot(a, tl.trans(weights), out_dtype=tl.float32)

    output = accumulator.to(c_ptr.type.element_ty)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        output,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def rdna_w8a16_gemm(
    activations: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Run channelwise W8A16 GEMM from a K-packed int32 weight matrix."""
    if activations.ndim != 2:
        raise ValueError(f"Expected 2D activations, got {activations.shape}")
    if not activations.is_contiguous():
        activations = activations.contiguous()
    if not packed_weights.is_contiguous() or not scales.is_contiguous():
        raise ValueError("Packed weights and scales must be contiguous")

    M, K = activations.shape
    N = packed_weights.shape[1]
    if K % 4 != 0 or packed_weights.shape[0] != K // 4:
        raise ValueError(
            f"Packed weight shape must be ({K // 4}, N), got "
            f"{tuple(packed_weights.shape)}"
        )
    if scales.shape != (1, N):
        raise ValueError(f"Scale shape must be (1, {N}), got {tuple(scales.shape)}")

    num_stages = 2
    if M <= 8:
        if K >= 2 * N:
            block_m, block_n, block_k, num_warps = 16, 32, 128, 2
        elif N >= 2 * K:
            block_m, block_n, block_k, num_warps = 16, 64, 64, 4
        else:
            block_m, block_n, block_k, num_warps = 16, 32, 128, 2
        num_stages = 1
    elif M <= 32:
        block_m, block_n, block_k, num_warps = 32, 64, 64, 4
    elif M <= 64:
        block_m, block_n, block_k, num_warps = 64, 64, 64, 8
    else:
        block_m, block_n, block_k, num_warps = 128, 64, 64, 8

    output = torch.empty((M, N), dtype=activations.dtype, device=activations.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _rdna_w8a16_kernel[grid](
        activations,
        packed_weights,
        scales,
        output,
        M,
        N,
        K,
        activations.stride(0),
        activations.stride(1),
        packed_weights.stride(0),
        packed_weights.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


class RDNAW8A16LinearKernel(MPLinearKernel):
    """RDNA fast path for symmetric channelwise packed W8A16 weights."""

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RDNA W8A16 kernel is ROCm-only"
        from vllm.platforms.rocm import on_gfx1x

        if not on_gfx1x():
            return False, "RDNA W8A16 kernel requires a gfx11/gfx12 GPU"
        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "RDNA W8A16 kernel only supports fp16 and bf16"
        if c.weight_type != scalar_types.uint8b128:
            return False, "RDNA W8A16 kernel requires symmetric uint8 weights"
        if c.group_size != -1:
            return False, "RDNA W8A16 kernel requires channelwise quantization"
        if c.zero_points or c.has_g_idx:
            return False, "RDNA W8A16 kernel does not support zero points or g_idx"
        if c.partition_weight_shape[0] % 4 != 0:
            return False, "Input features must be divisible by four"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        def transform_weight(parameter: BasevLLMParameter):
            permute_param_layout_(parameter, input_dim=0, output_dim=1, packed_dim=0)
            parameter.data = parameter.data.contiguous()
            return parameter

        def transform_scale(parameter: BasevLLMParameter):
            permute_param_layout_(parameter, input_dim=0, output_dim=1)
            parameter.data = parameter.data.contiguous()
            return parameter

        self._transform_param(layer, self.w_q_name, transform_weight)
        self._transform_param(layer, self.w_s_name, transform_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        packed_weights, scales, _, _ = self._get_weight_params(layer)
        x_2d = x.reshape(-1, x.shape[-1])
        output = rdna_w8a16_gemm(x_2d, packed_weights, scales)
        if bias is not None:
            output.add_(bias)
        return output.reshape(x.shape[:-1] + (self.config.partition_weight_shape[1],))
