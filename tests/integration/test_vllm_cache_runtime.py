from __future__ import annotations

from math import prod
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

pytest.importorskip("vllm")


def _model_config():
    from vllm_rwkv7.config import RWKV7ModelConfig

    return RWKV7ModelConfig.from_hf_config(
        {
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
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


def _vllm_config(dtype: torch.dtype):
    from vllm.config import CompilationConfig, ParallelConfig

    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=dtype),
        cache_config=SimpleNamespace(),
        compilation_config=CompilationConfig(),
        parallel_config=ParallelConfig(),
        speculative_config=None,
    )


def _metadata(
    *,
    query_starts: list[int],
    sequence_lengths: list[int],
    state_slots: list[int],
    device: torch.device,
):
    from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata

    return LinearAttentionMetadata(
        num_prefills=len(state_slots),
        num_prefill_tokens=query_starts[-1],
        num_decodes=0,
        num_decode_tokens=0,
        query_start_loc=torch.tensor(query_starts, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(sequence_lengths, dtype=torch.int32, device=device),
        state_indices_tensor=torch.tensor(state_slots, dtype=torch.int32, device=device),
    )


def _forward(block, hidden_states, metadata, vllm_config):
    from vllm.forward_context import set_forward_context

    value_first = hidden_states.new_zeros((hidden_states.shape[0], 8))
    with set_forward_context(metadata, vllm_config, num_tokens=hidden_states.shape[0]):
        return block(hidden_states, value_first)[0]


def _make_bound_block(device: torch.device):
    from vllm_rwkv7.model import RWKV7StatefulBlock

    dtype = torch.float16
    config = _model_config()
    vllm_config = _vllm_config(dtype)
    prefix = "model.layers.0"
    block = RWKV7StatefulBlock(
        config,
        layer_index=0,
        vllm_config=vllm_config,
        prefix=prefix,
        backend="reference",
        fla_prefill_min_tokens=64,
        fla_chunk_size=None,
    ).to(device=device, dtype=dtype)

    state_bytes = sum(
        prod(shape) * torch.empty((), dtype=state_dtype).element_size()
        for shape, state_dtype in zip(block.get_state_shape(), block.get_state_dtype(), strict=True)
    )
    raw_cache = torch.empty((3, 1, 1, state_bytes), dtype=torch.int8, device=device)
    block.bind_kv_cache(raw_cache)
    return block, vllm_config, prefix


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_vllm_cache_binding_chunk_reorder_release_and_prefix_copy() -> None:
    from vllm_rwkv7.model import RWKV7ForCausalLM

    torch.manual_seed(29)
    device = torch.device("cuda")
    block, vllm_config, prefix = _make_bound_block(device)

    chunked = torch.randn((4, 16), dtype=torch.float16, device=device)
    expected_chunked = _forward(block, chunked, None, vllm_config)
    for state in block.kv_cache:
        state.zero_()

    first = _forward(
        block,
        chunked[:2],
        {
            prefix: _metadata(
                query_starts=[0, 2],
                sequence_lengths=[2],
                state_slots=[1],
                device=device,
            )
        },
        vllm_config,
    )
    second = _forward(
        block,
        chunked[2:],
        {
            prefix: _metadata(
                query_starts=[0, 2],
                sequence_lengths=[4],
                state_slots=[1],
                device=device,
            )
        },
        vllm_config,
    )
    torch.testing.assert_close(torch.cat((first, second)), expected_chunked)

    request_a = torch.randn((3, 16), dtype=torch.float16, device=device)
    request_b = torch.randn((3, 16), dtype=torch.float16, device=device)
    expected_a = _forward(block, request_a, None, vllm_config)
    expected_b = _forward(block, request_b, None, vllm_config)
    for state in block.kv_cache:
        state.zero_()

    _forward(
        block,
        torch.cat((request_a[:2], request_b[:1])),
        {
            prefix: _metadata(
                query_starts=[0, 2, 3],
                sequence_lengths=[2, 1],
                state_slots=[2, 0],
                device=device,
            )
        },
        vllm_config,
    )
    reordered = _forward(
        block,
        torch.cat((request_b[1:], request_a[2:])),
        {
            prefix: _metadata(
                query_starts=[0, 2, 3],
                sequence_lengths=[3, 3],
                state_slots=[0, 2],
                device=device,
            )
        },
        vllm_config,
    )
    torch.testing.assert_close(reordered[:2], expected_b[1:])
    torch.testing.assert_close(reordered[2:], expected_a[2:])

    fresh = torch.randn((2, 16), dtype=torch.float16, device=device)
    expected_fresh = _forward(block, fresh, None, vllm_config)
    for state in block.kv_cache:
        state[2].fill_(1)
    reused = _forward(
        block,
        fresh,
        {
            prefix: _metadata(
                query_starts=[0, 2],
                sequence_lengths=[2],
                state_slots=[2],
                device=device,
            )
        },
        vllm_config,
    )
    torch.testing.assert_close(reused, expected_fresh)

    copy_functions = RWKV7ForCausalLM.get_mamba_state_copy_func()
    assert len(copy_functions) == len(block.kv_cache) == 2
    for state, copy_function in zip(block.kv_cache, copy_functions, strict=True):
        copy_spec = copy_function(state, [0, 1], 0, 1)
        assert copy_spec.start_addr == state[0].data_ptr()
        assert copy_spec.num_elements == state[0].numel()
