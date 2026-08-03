# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.rdna_w8a16 import (
    RDNAW8A16LinearKernel,
    rdna_w8a16_gemm,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="RDNA W8A16 requires ROCm"
)


def pack_int8_weights(weight: torch.Tensor) -> torch.Tensor:
    unsigned = (weight.to(torch.int16) + 128).to(torch.int32)
    grouped = unsigned.reshape(weight.shape[0] // 4, 4, weight.shape[1])
    shifts = torch.arange(4, device=weight.device, dtype=torch.int32) * 8
    return torch.sum(grouped << shifts[None, :, None], dim=1, dtype=torch.int32)


@pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
@pytest.mark.parametrize("shape", [(256, 256), (256, 512), (512, 256)])
def test_rdna_w8a16_matches_dequantized_reference(batch_size, shape):
    from vllm.platforms.rocm import on_gfx1x

    if not on_gfx1x():
        pytest.skip("RDNA W8A16 requires gfx11/gfx12")
    K, N = shape
    generator = torch.Generator(device="cuda").manual_seed(batch_size + K + N)
    activations = torch.randn(
        batch_size,
        K,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    weight = torch.randint(
        -127,
        128,
        (K, N),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    scales = (
        torch.rand(1, N, device="cuda", dtype=torch.float16, generator=generator) / 64
    )
    packed = pack_int8_weights(weight).contiguous()

    output = rdna_w8a16_gemm(activations, packed, scales)
    reference = activations @ (weight.to(torch.float16) * scales).contiguous()

    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-1)


def test_rdna_w8a16_is_selected_for_channelwise_weights():
    from vllm.platforms.rocm import on_gfx1x

    if not on_gfx1x():
        pytest.skip("RDNA W8A16 requires gfx11/gfx12")
    config = MPLinearLayerConfig(
        full_weight_shape=(2048, 8192),
        partition_weight_shape=(2048, 8192),
        weight_type=scalar_types.uint8b128,
        act_type=torch.float16,
        group_size=-1,
        zero_points=False,
        has_g_idx=False,
    )

    assert choose_mp_linear_kernel(config) is RDNAW8A16LinearKernel
