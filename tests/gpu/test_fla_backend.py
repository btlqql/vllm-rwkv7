from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional

from vllm_rwkv7.backends import rwkv7_scan_backend
from vllm_rwkv7.components import RWKV7ReferenceLayer
from vllm_rwkv7.config import RWKV7ModelConfig

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("sequence_length", [1, 64])
def test_real_fla_backend_matches_reference(sequence_length: int) -> None:
    pytest.importorskip("fla")
    if not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm is required")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(31 + sequence_length)
    shape = (1, sequence_length, 2, 4)
    inputs = {
        "r": torch.randn(shape, generator=generator, device=device, dtype=torch.float16),
        "decay_logits": torch.randn(shape, generator=generator, device=device, dtype=torch.float16),
        "k": torch.randn(shape, generator=generator, device=device, dtype=torch.float16),
        "v": torch.randn(shape, generator=generator, device=device, dtype=torch.float16),
        "kk": torch.randn(shape, generator=generator, device=device, dtype=torch.float16),
        "a": torch.sigmoid(
            torch.randn(shape, generator=generator, device=device, dtype=torch.float16)
        ),
        "initial_state": torch.randn(
            1, 2, 4, 4, generator=generator, device=device, dtype=torch.float32
        ),
    }

    reference_output, reference_state = rwkv7_scan_backend(**inputs, backend="reference")
    fla_output, fla_state = rwkv7_scan_backend(
        **inputs,
        backend="fla",
        fla_prefill_min_tokens=64,
        fla_chunk_size=32,
    )

    cosine = functional.cosine_similarity(
        reference_output.float().flatten(),
        fla_output.float().flatten(),
        dim=0,
    )
    assert cosine.item() >= 0.9999
    torch.testing.assert_close(fla_output.float(), reference_output.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(fla_state, reference_state, rtol=1e-2, atol=2e-2)


@pytest.mark.parametrize("sequence_length", [1, 64])
def test_real_fla_full_layer_matches_reference(sequence_length: int) -> None:
    pytest.importorskip("fla")
    if not torch.cuda.is_available():
        pytest.skip("CUDA or ROCm is required")

    config = RWKV7ModelConfig.from_hf_config(
        {
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "head_dim": 4,
            "attention_hidden_size": 8,
            "intermediate_size": 24,
            "decay_low_rank_dim": 4,
            "gate_low_rank_dim": 4,
            "a_low_rank_dim": 4,
            "v_low_rank_dim": 4,
        }
    )
    reference_layer = RWKV7ReferenceLayer(config, 1, backend="reference").cuda().half().eval()
    fla_layer = (
        RWKV7ReferenceLayer(
            config,
            1,
            backend="fla",
            fla_prefill_min_tokens=64,
            fla_chunk_size=32,
        )
        .cuda()
        .half()
        .eval()
    )
    fla_layer.load_state_dict(reference_layer.state_dict())

    generator = torch.Generator(device="cuda").manual_seed(91 + sequence_length)
    hidden_states = torch.randn(
        1,
        sequence_length,
        config.hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    previous_attention = torch.randn(
        1,
        config.hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    previous_ffn = torch.randn(
        1,
        config.hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    value_first = torch.randn(
        1,
        sequence_length,
        config.attention_hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    matrix_state = torch.randn(
        1,
        config.num_attention_heads,
        config.head_dim,
        config.head_dim,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )

    with torch.inference_mode():
        reference_result = reference_layer.forward_sequence(
            hidden_states,
            previous_attention,
            previous_ffn,
            value_first,
            matrix_state,
        )
        fla_result = fla_layer.forward_sequence(
            hidden_states,
            previous_attention,
            previous_ffn,
            value_first,
            matrix_state,
        )

    cosine = functional.cosine_similarity(
        reference_result[0].float().flatten(),
        fla_result[0].float().flatten(),
        dim=0,
    )
    assert cosine.item() >= 0.9999
    torch.testing.assert_close(
        fla_result[0].float(), reference_result[0].float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(fla_result[4], reference_result[4], rtol=1e-2, atol=2e-2)
