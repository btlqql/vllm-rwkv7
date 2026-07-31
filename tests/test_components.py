from __future__ import annotations

import torch

from vllm_rwkv7.components import RWKV7ReferenceLayer
from vllm_rwkv7.config import RWKV7ModelConfig


def make_config() -> RWKV7ModelConfig:
    return RWKV7ModelConfig.from_hf_config(
        {
            "vocab_size": 32,
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


def test_layer_parameter_names_match_hf_checkpoint_contract() -> None:
    layer = RWKV7ReferenceLayer(make_config(), layer_index=1)
    names = set(layer.state_dict())

    assert "attn.x_r" in names
    assert "attn.r_proj.weight" in names
    assert "attn.w_lora.lora.0.weight" in names
    assert "attn.w_lora.lora.2.bias" in names
    assert "attn.v_lora.lora.2.bias" in names
    assert "attn.g_norm.weight" in names
    assert "ffn.x_k" in names
    assert "ffn.key.weight" in names
    assert "attn_norm.weight" in names
    assert "ffn_norm.weight" in names


def test_attention_projection_may_differ_from_hidden_size() -> None:
    config = make_config()
    layer = RWKV7ReferenceLayer(config, layer_index=0)
    batch = 3
    hidden = torch.randn(batch, config.hidden_size)
    previous_attention = torch.zeros_like(hidden)
    previous_ffn = torch.zeros_like(hidden)
    value_first = torch.zeros(batch, config.attention_hidden_size)
    matrix_state = torch.zeros(
        batch,
        config.num_attention_heads,
        config.head_dim,
        config.head_dim,
        dtype=torch.float32,
    )

    output, next_attention, next_ffn, next_value_first, next_matrix = layer.forward_token(
        hidden,
        previous_attention,
        previous_ffn,
        value_first,
        matrix_state,
    )

    assert output.shape == hidden.shape
    assert next_attention.shape == hidden.shape
    assert next_ffn.shape == hidden.shape
    assert next_value_first.shape == (batch, config.attention_hidden_size)
    assert next_matrix.shape == matrix_state.shape
    assert next_matrix.dtype == torch.float32


def test_state_handoff_matches_contiguous_execution() -> None:
    torch.manual_seed(11)
    config = make_config()
    layer = RWKV7ReferenceLayer(config, layer_index=0).eval()
    tokens = torch.randn(1, 2, config.hidden_size)

    def initial_state():
        return (
            torch.zeros(1, config.hidden_size),
            torch.zeros(1, config.hidden_size),
            torch.zeros(1, config.attention_hidden_size),
            torch.zeros(
                1,
                config.num_attention_heads,
                config.head_dim,
                config.head_dim,
                dtype=torch.float32,
            ),
        )

    attention, ffn, value_first, matrix = initial_state()
    contiguous_outputs = []
    for token_index in range(2):
        output, attention, ffn, value_first, matrix = layer.forward_token(
            tokens[:, token_index], attention, ffn, value_first, matrix
        )
        contiguous_outputs.append(output)

    attention, ffn, value_first, matrix = initial_state()
    first_output, attention, ffn, value_first, matrix = layer.forward_token(
        tokens[:, 0], attention, ffn, value_first, matrix
    )
    second_output, attention, ffn, value_first, matrix = layer.forward_token(
        tokens[:, 1], attention, ffn, value_first, matrix
    )

    torch.testing.assert_close(first_output, contiguous_outputs[0])
    torch.testing.assert_close(second_output, contiguous_outputs[1])


def test_sequence_path_matches_token_loop() -> None:
    torch.manual_seed(17)
    config = make_config()
    layer = RWKV7ReferenceLayer(config, layer_index=0).eval()
    tokens = torch.randn(2, 5, config.hidden_size)
    previous_attention = torch.zeros(2, config.hidden_size)
    previous_ffn = torch.zeros(2, config.hidden_size)
    value_first = torch.zeros(2, 5, config.attention_hidden_size)
    matrix_state = torch.zeros(
        2,
        config.num_attention_heads,
        config.head_dim,
        config.head_dim,
        dtype=torch.float32,
    )

    sequence_result = layer.forward_sequence(
        tokens,
        previous_attention,
        previous_ffn,
        value_first,
        matrix_state,
    )

    token_outputs = []
    token_values = []
    attention = previous_attention
    ffn = previous_ffn
    state = matrix_state
    for token_index in range(tokens.shape[1]):
        output, attention, ffn, token_value, state = layer.forward_token(
            tokens[:, token_index],
            attention,
            ffn,
            value_first[:, token_index],
            state,
        )
        token_outputs.append(output)
        token_values.append(token_value)

    torch.testing.assert_close(sequence_result[0], torch.stack(token_outputs, dim=1))
    torch.testing.assert_close(sequence_result[1], attention)
    torch.testing.assert_close(sequence_result[2], ffn)
    torch.testing.assert_close(sequence_result[3], torch.stack(token_values, dim=1))
    torch.testing.assert_close(sequence_result[4], state)
