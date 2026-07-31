"""Stateful vLLM model using the clean-room PyTorch RWKV-7 implementation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import get_temporal_copy_spec
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsAttentionFree,
    SupportsMambaPrefixCaching,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_rwkv7.cache import plan_packed_state_spans
from vllm_rwkv7.components import RWKV7ReferenceLayer
from vllm_rwkv7.config import RWKV7ModelConfig
from vllm_rwkv7.kernel_policy import KernelPolicy, select_kernel_policy


def _join_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _require_tensor_parallel_one(vllm_config: VllmConfig) -> None:
    tensor_parallel_size = vllm_config.parallel_config.tensor_parallel_size
    if tensor_parallel_size != 1:
        raise NotImplementedError(
            f"RWKV-7 clean-room P0 supports tensor_parallel_size=1; got {tensor_parallel_size}"
        )


def _require_p0_execution(vllm_config: VllmConfig) -> None:
    _require_tensor_parallel_one(vllm_config)
    if not vllm_config.model_config.enforce_eager:
        raise NotImplementedError(
            "RWKV-7 clean-room P0 requires enforce_eager=True; "
            "compiled execution is not implemented yet"
        )
    pipeline_parallel_size = getattr(vllm_config.parallel_config, "pipeline_parallel_size", 1)
    if pipeline_parallel_size != 1:
        raise NotImplementedError(
            f"RWKV-7 clean-room P0 supports pipeline_parallel_size=1; got {pipeline_parallel_size}"
        )
    if getattr(vllm_config, "speculative_config", None) is not None:
        raise NotImplementedError("RWKV-7 clean-room P0 does not support speculative decoding")


def _runtime_kernel_policy() -> KernelPolicy:
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        return select_kernel_policy(
            name=torch.cuda.get_device_name(device),
            capability=torch.cuda.get_device_capability(device),
        )
    return select_kernel_policy(name="cpu", capability=None)


class RWKV7StatefulBlock(RWKV7ReferenceLayer, MambaBase):
    """One RWKV block plus vLLM-managed recurrent cache slots."""

    supports_dcp = False

    def __init__(
        self,
        config: RWKV7ModelConfig,
        layer_index: int,
        *,
        vllm_config: VllmConfig,
        prefix: str,
        backend: str,
        fla_prefill_min_tokens: int,
        fla_chunk_size: int | None,
    ) -> None:
        super().__init__(
            config,
            layer_index,
            backend=backend,
            fla_prefill_min_tokens=fla_prefill_min_tokens,
            fla_chunk_size=fla_chunk_size,
        )
        self.rwkv_config = config
        self.model_dtype = vllm_config.model_config.dtype
        self.cache_config = vllm_config.cache_config
        self.prefix = prefix

        static_context = vllm_config.compilation_config.static_forward_context
        if prefix in static_context:
            raise ValueError(f"duplicate RWKV-7 stateful layer name: {prefix}")
        static_context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.LINEAR

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        config = self.rwkv_config
        return (
            (2, config.hidden_size),
            (config.num_attention_heads, config.head_dim, config.head_dim),
        )

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return self.model_dtype, torch.float32

    def _run_sequence(
        self,
        hidden_states: torch.Tensor,
        value_first: torch.Tensor,
        start: int,
        end: int,
        shift_state: torch.Tensor,
        matrix_state: torch.Tensor,
        output: torch.Tensor,
        next_value_first: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            sequence_output,
            next_attention,
            next_ffn,
            sequence_value_first,
            next_matrix,
        ) = self.forward_sequence(
            hidden_states[start:end].unsqueeze(0),
            shift_state[0].reshape(1, -1),
            shift_state[1].reshape(1, -1),
            value_first[start:end].unsqueeze(0),
            matrix_state.unsqueeze(0).float(),
        )
        output[start:end] = sequence_output[0]
        next_value_first[start:end] = sequence_value_first[0]
        next_shift = torch.stack((next_attention[0], next_ffn[0]), dim=0)
        return next_shift, next_matrix[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        value_first: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.zeros_like(hidden_states)
        next_value_first = torch.zeros_like(value_first)
        metadata_raw = get_forward_context().attn_metadata

        if metadata_raw is None:
            shift_state = hidden_states.new_zeros((2, self.rwkv_config.hidden_size))
            matrix_state = torch.zeros(
                self.rwkv_config.num_attention_heads,
                self.rwkv_config.head_dim,
                self.rwkv_config.head_dim,
                dtype=torch.float32,
                device=hidden_states.device,
            )
            self._run_sequence(
                hidden_states,
                value_first,
                0,
                hidden_states.shape[0],
                shift_state,
                matrix_state,
                output,
                next_value_first,
            )
            return output, next_value_first

        metadata = metadata_raw[self.prefix] if isinstance(metadata_raw, dict) else metadata_raw
        if metadata.num_decode_tokens != metadata.num_decodes:
            raise NotImplementedError("RWKV-7 P0 does not support speculative multi-token decode")

        shift_cache, matrix_cache = self.kv_cache
        expected_shift_shape = (2, self.rwkv_config.hidden_size)
        expected_matrix_shape = (
            self.rwkv_config.num_attention_heads,
            self.rwkv_config.head_dim,
            self.rwkv_config.head_dim,
        )
        if tuple(shift_cache.shape[1:]) != expected_shift_shape:
            raise ValueError(
                "RWKV-7 shift cache has incompatible shape: "
                f"expected [slots, {expected_shift_shape}], got {tuple(shift_cache.shape)}"
            )
        if tuple(matrix_cache.shape[1:]) != expected_matrix_shape:
            raise ValueError(
                "RWKV-7 matrix cache has incompatible shape: "
                f"expected [slots, {expected_matrix_shape}], got {tuple(matrix_cache.shape)}"
            )
        if shift_cache.shape[0] != matrix_cache.shape[0]:
            raise ValueError("RWKV-7 shift and matrix caches must have the same slot count")

        spans = plan_packed_state_spans(
            query_start_loc=metadata.query_start_loc.tolist(),
            state_slots=metadata.state_indices_tensor.tolist(),
            sequence_lengths=metadata.seq_lens.tolist(),
            total_tokens=hidden_states.shape[0],
            num_cache_slots=shift_cache.shape[0],
        )
        for span in spans:
            if not span.active:
                continue
            if span.has_cached_prefix:
                shift_state = shift_cache[span.state_slot].clone()
                matrix_state = matrix_cache[span.state_slot].clone()
            else:
                shift_state = torch.zeros_like(shift_cache[span.state_slot])
                matrix_state = torch.zeros_like(matrix_cache[span.state_slot])
            next_shift, next_matrix = self._run_sequence(
                hidden_states,
                value_first,
                span.start,
                span.end,
                shift_state,
                matrix_state,
                output,
                next_value_first,
            )
            shift_cache[span.state_slot].copy_(next_shift)
            matrix_cache[span.state_slot].copy_(next_matrix)

        return output, next_value_first


class RWKV7Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        _require_p0_execution(vllm_config)
        self.hf_config = vllm_config.model_config.hf_config
        self.config = RWKV7ModelConfig.from_hf_config(self.hf_config)
        self.kernel_policy = _runtime_kernel_policy()
        model_prefix = _join_prefix(prefix, "model")
        self.embeddings = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=_join_prefix(model_prefix, "embeddings"),
        )
        self.layers = nn.ModuleList(
            [
                RWKV7StatefulBlock(
                    self.config,
                    layer_index,
                    vllm_config=vllm_config,
                    prefix=_join_prefix(model_prefix, f"layers.{layer_index}"),
                    backend=self.kernel_policy.backend,
                    fla_prefill_min_tokens=self.kernel_policy.fla_prefill_min_tokens,
                    fla_chunk_size=self.kernel_policy.fla_chunk_size,
                )
                for layer_index in range(self.config.num_hidden_layers)
            ]
        )
        self.norm = nn.LayerNorm(
            self.config.hidden_size,
            eps=self.config.layer_norm_epsilon,
            bias=self.config.norm_bias,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        if intermediate_tensors is not None:
            raise NotImplementedError("RWKV-7 clean-room P0 does not support pipeline parallelism")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("RWKV7Model requires input_ids or inputs_embeds")
            hidden_states = self.embed_input_ids(input_ids)
        else:
            hidden_states = inputs_embeds

        value_first = hidden_states.new_zeros(
            (hidden_states.shape[0], self.config.attention_hidden_size)
        )
        for layer in self.layers:
            hidden_states, value_first = layer(hidden_states, value_first)
        return self.norm(hidden_states)


class RWKV7ForCausalLM(
    nn.Module,
    HasInnerState,
    IsAttentionFree,
    SupportsMambaPrefixCaching,
):
    """vLLM CausalLM entry point for canonical HF RWKV-7 checkpoints."""

    hf_to_vllm_mapper = WeightsMapper()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        _require_p0_execution(vllm_config)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.config = vllm_config.model_config.hf_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = RWKV7Model(vllm_config=vllm_config, prefix=prefix)
        config = self.model.config
        if config.tie_word_embeddings:
            self.lm_head = self.model.embeddings
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=_join_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        _require_tensor_parallel_one(vllm_config)
        config = RWKV7ModelConfig.from_hf_config(vllm_config.model_config.hf_config)
        return (
            (2, config.hidden_size),
            (config.num_attention_heads, config.head_dim, config.head_dim),
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return vllm_config.model_config.dtype, torch.float32

    @classmethod
    def get_mamba_state_copy_func(cls):
        return get_temporal_copy_spec, get_temporal_copy_spec

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)
