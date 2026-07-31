"""Pure PyTorch RWKV-7 components with Hugging Face-compatible weight names."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn

from vllm_rwkv7.backends import FLAOperations, rwkv7_scan_backend
from vllm_rwkv7.config import RWKV7ModelConfig


class RWKV7LowRank(nn.Module):
    """Two projections stored under the checkpoint-compatible `lora` path."""

    def __init__(
        self,
        input_size: int,
        rank: int,
        output_size: int,
        *,
        bias: bool,
    ) -> None:
        super().__init__()
        self.lora = nn.Sequential(
            nn.Linear(input_size, rank, bias=False),
            nn.Identity(),
            nn.Linear(rank, output_size, bias=bias),
        )

    def project(self, inputs: torch.Tensor, *, first_activation: str | None = None) -> torch.Tensor:
        hidden = self.lora[0](inputs)
        if first_activation == "tanh":
            hidden = torch.tanh(hidden)
        elif first_activation == "sigmoid":
            hidden = torch.sigmoid(hidden)
        elif first_activation is not None:
            raise ValueError(f"unsupported low-rank activation: {first_activation}")
        return self.lora[2](hidden)


class RWKV7TimeMix(nn.Module):
    def __init__(
        self,
        config: RWKV7ModelConfig,
        layer_index: int,
        *,
        backend: str = "reference",
        fla_operations: FLAOperations | None = None,
        fla_prefill_min_tokens: int = 64,
        fla_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.backend = backend
        self.fla_operations = fla_operations
        self.fla_prefill_min_tokens = fla_prefill_min_tokens
        self.fla_chunk_size = fla_chunk_size
        self.hidden_size = config.hidden_size
        self.attention_hidden_size = config.attention_hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, name, nn.Parameter(torch.zeros(1, 1, config.hidden_size)))

        self.k_k = nn.Parameter(torch.zeros(config.attention_hidden_size))
        self.k_a = nn.Parameter(torch.zeros(config.attention_hidden_size))
        self.r_k = nn.Parameter(torch.zeros(config.num_attention_heads, config.head_dim))

        self.r_proj = nn.Linear(config.hidden_size, config.attention_hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.attention_hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.attention_hidden_size, bias=False)
        self.o_proj = nn.Linear(config.attention_hidden_size, config.hidden_size, bias=False)

        self.w_lora = RWKV7LowRank(
            config.hidden_size,
            config.decay_low_rank_dim,
            config.attention_hidden_size,
            bias=True,
        )
        self.a_lora = RWKV7LowRank(
            config.hidden_size,
            config.a_low_rank_dim,
            config.attention_hidden_size,
            bias=True,
        )
        self.g_lora = RWKV7LowRank(
            config.hidden_size,
            config.gate_low_rank_dim,
            config.attention_hidden_size,
            bias=False,
        )
        self.v_lora = (
            None
            if layer_index == 0
            else RWKV7LowRank(
                config.hidden_size,
                config.v_low_rank_dim,
                config.attention_hidden_size,
                bias=True,
            )
        )
        self.g_norm = nn.GroupNorm(
            config.num_attention_heads,
            config.attention_hidden_size,
            eps=config.head_dim * 1e-5,
        )
        if not config.norm_bias:
            self.g_norm.register_parameter("bias", None)

    def forward_token(
        self,
        hidden_states: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        value_first: torch.Tensor,
        matrix_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output, next_hidden, next_value_first, next_state = self.forward_sequence(
            hidden_states.unsqueeze(1),
            previous_hidden_states,
            value_first.unsqueeze(1),
            matrix_state,
        )
        return output[:, 0], next_hidden, next_value_first[:, 0], next_state

    def forward_sequence(
        self,
        hidden_states: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        value_first: torch.Tensor,
        matrix_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("RWKV7TimeMix expects [batch, time, hidden_size]")
        batch_size, sequence_length, _ = hidden_states.shape
        shifted = torch.cat(
            (previous_hidden_states.unsqueeze(1), hidden_states[:, :-1]),
            dim=1,
        )
        delta = shifted - hidden_states

        mixed = {
            name: hidden_states + delta * getattr(self, name).reshape(1, 1, self.hidden_size)
            for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g")
        }

        receptance = self.r_proj(mixed["x_r"])
        decay_logits = self.w_lora.project(mixed["x_w"], first_activation="tanh")
        key = self.k_proj(mixed["x_k"])
        value = self.v_proj(mixed["x_v"])
        in_context_rate = torch.sigmoid(self.a_lora.project(mixed["x_a"]))
        gate = self.g_lora.project(mixed["x_g"], first_activation="sigmoid")

        if self.layer_index == 0:
            next_value_first = value
        else:
            assert self.v_lora is not None
            blend = torch.sigmoid(self.v_lora.project(mixed["x_v"]))
            value = value + (value_first - value) * blend
            next_value_first = value_first

        heads = self.num_attention_heads
        head_dim = self.head_dim
        head_shape = (batch_size, sequence_length, heads, head_dim)
        normalized_key = key * self.k_k.reshape(1, 1, self.attention_hidden_size)
        adjusted_key = key * (
            1 + (in_context_rate - 1) * self.k_a.reshape(1, 1, self.attention_hidden_size)
        )

        recurrent_output, next_matrix_state = rwkv7_scan_backend(
            r=receptance.reshape(head_shape),
            decay_logits=decay_logits.reshape(head_shape),
            k=adjusted_key.reshape(head_shape),
            v=value.reshape(head_shape),
            kk=normalized_key.reshape(head_shape),
            a=in_context_rate.reshape(head_shape),
            initial_state=matrix_state,
            backend=self.backend,
            fla_operations=self.fla_operations,
            fla_prefill_min_tokens=self.fla_prefill_min_tokens,
            fla_chunk_size=self.fla_chunk_size,
        )
        recurrent_output = recurrent_output.reshape(
            batch_size * sequence_length,
            self.attention_hidden_size,
        )
        recurrent_output = self.g_norm(recurrent_output).reshape(
            batch_size,
            sequence_length,
            self.attention_hidden_size,
        )

        bonus_scale = (
            receptance.reshape(head_shape)
            * adjusted_key.reshape(head_shape)
            * self.r_k.reshape(1, 1, heads, head_dim)
        ).sum(dim=-1, keepdim=True)
        bonus = (bonus_scale * value.reshape(head_shape)).reshape(
            batch_size, sequence_length, self.attention_hidden_size
        )
        output = self.o_proj((recurrent_output + bonus) * gate)
        return output, hidden_states[:, -1], next_value_first, next_matrix_state


class RWKV7ChannelMix(nn.Module):
    def __init__(self, config: RWKV7ModelConfig) -> None:
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(config.hidden_size))
        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward_token(
        self,
        hidden_states: torch.Tensor,
        previous_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, next_hidden = self.forward_sequence(
            hidden_states.unsqueeze(1), previous_hidden_states
        )
        return output[:, 0], next_hidden

    def forward_sequence(
        self,
        hidden_states: torch.Tensor,
        previous_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shifted = torch.cat(
            (previous_hidden_states.unsqueeze(1), hidden_states[:, :-1]),
            dim=1,
        )
        mixed = hidden_states + (shifted - hidden_states) * self.x_k
        activated = torch.relu(self.key(mixed)).square()
        return self.value(activated), hidden_states[:, -1]


class RWKV7LayerLike(Protocol):
    layer_index: int
    attn: RWKV7TimeMix
    ffn: RWKV7ChannelMix
    attn_norm: nn.LayerNorm
    ffn_norm: nn.LayerNorm
    pre_norm: nn.LayerNorm | None


def rwkv7_layer_token(
    layer: RWKV7LayerLike,
    hidden_states: torch.Tensor,
    previous_attention: torch.Tensor,
    previous_ffn: torch.Tensor,
    value_first: torch.Tensor,
    matrix_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output, next_attention, next_ffn, next_value_first, next_matrix_state = rwkv7_layer_sequence(
        layer,
        hidden_states.unsqueeze(1),
        previous_attention,
        previous_ffn,
        value_first.unsqueeze(1),
        matrix_state,
    )
    return (
        output[:, 0],
        next_attention,
        next_ffn,
        next_value_first[:, 0],
        next_matrix_state,
    )


def rwkv7_layer_sequence(
    layer: RWKV7LayerLike,
    hidden_states: torch.Tensor,
    previous_attention: torch.Tensor,
    previous_ffn: torch.Tensor,
    value_first: torch.Tensor,
    matrix_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    residual = layer.pre_norm(hidden_states) if layer.pre_norm is not None else hidden_states
    attention_input = layer.attn_norm(residual)
    attention_output, next_attention, next_value_first, next_matrix_state = (
        layer.attn.forward_sequence(
            attention_input,
            previous_attention,
            value_first,
            matrix_state,
        )
    )
    after_attention = residual + attention_output
    ffn_input = layer.ffn_norm(after_attention)
    ffn_output, next_ffn = layer.ffn.forward_sequence(ffn_input, previous_ffn)
    return (
        after_attention + ffn_output,
        next_attention,
        next_ffn,
        next_value_first,
        next_matrix_state,
    )


class RWKV7ReferenceLayer(nn.Module):
    """A cache-agnostic layer used by tests and the vLLM state wrapper."""

    def __init__(
        self,
        config: RWKV7ModelConfig,
        layer_index: int,
        *,
        backend: str = "reference",
        fla_operations: FLAOperations | None = None,
        fla_prefill_min_tokens: int = 64,
        fla_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.attn = RWKV7TimeMix(
            config,
            layer_index,
            backend=backend,
            fla_operations=fla_operations,
            fla_prefill_min_tokens=fla_prefill_min_tokens,
            fla_chunk_size=fla_chunk_size,
        )
        self.ffn = RWKV7ChannelMix(config)
        self.attn_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            bias=config.norm_bias,
        )
        self.ffn_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            bias=config.norm_bias,
        )
        self.pre_norm = (
            nn.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_epsilon,
                bias=config.norm_bias,
            )
            if layer_index == 0
            else None
        )

    def forward_token(
        self,
        hidden_states: torch.Tensor,
        previous_attention: torch.Tensor,
        previous_ffn: torch.Tensor,
        value_first: torch.Tensor,
        matrix_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return rwkv7_layer_token(
            self,
            hidden_states,
            previous_attention,
            previous_ffn,
            value_first,
            matrix_state,
        )

    def forward_sequence(
        self,
        hidden_states: torch.Tensor,
        previous_attention: torch.Tensor,
        previous_ffn: torch.Tensor,
        value_first: torch.Tensor,
        matrix_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return rwkv7_layer_sequence(
            self,
            hidden_states,
            previous_attention,
            previous_ffn,
            value_first,
            matrix_state,
        )
