# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from transformers.configuration_utils import PretrainedConfig


def get_rwkv7_hybrid_attention_spec(
    attn: dict[str, Any] | list[dict[str, Any]] | None,
    layer_idx: int,
) -> dict[str, Any] | None:
    if attn is None:
        return None
    specs = [attn] if isinstance(attn, dict) else attn
    return next((spec for spec in specs if layer_idx in spec["layers"]), None)


class RWKV7Config(PretrainedConfig):
    model_type = "rwkv7"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        num_hidden_layers: int = 24,
        head_dim: int | None = 64,
        num_heads: int | None = None,
        decay_low_rank_dim: int = 64,
        gate_low_rank_dim: int = 128,
        a_low_rank_dim: int = 64,
        v_low_rank_dim: int = 16,
        hidden_act: str = "sqrelu",
        max_position_embeddings: int = 2048,
        norm_first: bool = True,
        norm_bias: bool = True,
        norm_eps: float = 1e-5,
        attn: dict[str, Any] | list[dict[str, Any]] | None = None,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = True,
        vocab_size: int = 32000,
        value_dim: int | list[int] | None = None,
        **kwargs,
    ):
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.norm_first = norm_first
        self.num_hidden_layers = num_hidden_layers

        if head_dim is None and num_heads is not None:
            head_dim = int(hidden_size // num_heads)
        elif head_dim is not None and num_heads is None:
            num_heads = int(hidden_size // head_dim)
        elif head_dim is None and num_heads is None:
            raise ValueError("Either `head_dim` or `num_heads` must be specified.")

        if value_dim is None:
            value_dim = [hidden_size] * num_hidden_layers
        elif isinstance(value_dim, int):
            if value_dim < hidden_size or value_dim % hidden_size != 0:
                raise ValueError(
                    "`value_dim` must be >= hidden_size and divisible by hidden_size."
                )
            value_dim = [value_dim] * num_hidden_layers
        else:
            if len(value_dim) != num_hidden_layers:
                raise ValueError(
                    "`value_dim` must have the same length as num_hidden_layers."
                )
            for dim in value_dim:
                if dim < hidden_size or dim % hidden_size != 0:
                    raise ValueError(
                        "`value_dim` must be >= hidden_size and divisible "
                        "by hidden_size."
                    )

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.decay_low_rank_dim = decay_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.a_low_rank_dim = a_low_rank_dim
        self.v_low_rank_dim = v_low_rank_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.norm_bias = norm_bias
        self.norm_eps = norm_eps
        self.attn = self._normalize_hybrid_attention(attn)
        self.layers_block_type = [
            (
                "attention"
                if get_rwkv7_hybrid_attention_spec(self.attn, layer_idx) is not None
                else "mamba"
            )
            for layer_idx in range(num_hidden_layers)
        ]
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` "
                "cannot both be enabled."
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        current_architectures = getattr(self, "architectures", None)
        if self.attn is not None and current_architectures in (
            None,
            ["RWKV7ForCausalLM"],
        ):
            self.architectures = ["RWKV7HybridForCausalLM"]

    def _normalize_hybrid_attention(
        self,
        attn: dict[str, Any] | list[dict[str, Any]] | None,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        if attn is None:
            return None
        if not isinstance(attn, (dict, list)):
            raise ValueError("`attn` must be a dictionary or list of dictionaries.")
        specs = [attn] if isinstance(attn, dict) else attn
        normalized_specs = []
        assigned_layers: set[int] = set()
        for spec in specs:
            if not isinstance(spec, dict):
                raise ValueError(
                    "Each hybrid `attn` specification must be a dictionary."
                )
            normalized = dict(spec)
            layers = normalized.get("layers")
            if not isinstance(layers, list) or not layers:
                raise ValueError("`attn.layers` must be a non-empty list.")
            if any(
                isinstance(layer, bool)
                or not isinstance(layer, int)
                or layer < 0
                or layer >= self.num_hidden_layers
                for layer in layers
            ):
                raise ValueError(
                    "Every `attn.layers` entry must be a valid layer index."
                )
            duplicate_layers = assigned_layers.intersection(layers)
            if duplicate_layers:
                raise ValueError(
                    f"Hybrid attention layers are assigned more than once: "
                    f"{sorted(duplicate_layers)}"
                )
            assigned_layers.update(layers)
            num_heads = normalized.get("num_heads")
            if (
                isinstance(num_heads, bool)
                or not isinstance(num_heads, int)
                or num_heads <= 0
                or self.hidden_size % num_heads != 0
            ):
                raise ValueError(
                    "`attn.num_heads` must be positive and divide hidden_size."
                )
            num_kv_heads = normalized.get("num_kv_heads", num_heads)
            if (
                isinstance(num_kv_heads, bool)
                or not isinstance(num_kv_heads, int)
                or num_kv_heads <= 0
                or num_heads % num_kv_heads != 0
            ):
                raise ValueError(
                    "`attn.num_kv_heads` must be positive and divide num_heads."
                )
            normalized["layers"] = list(layers)
            normalized["num_heads"] = num_heads
            normalized["num_kv_heads"] = num_kv_heads
            normalized["qkv_bias"] = bool(normalized.get("qkv_bias", False))
            normalized["window_size"] = normalized.get("window_size")
            normalized["rope_theta"] = float(normalized.get("rope_theta", 10000.0))
            if normalized["rope_theta"] <= 0:
                raise ValueError("`attn.rope_theta` must be positive.")
            normalized_specs.append(normalized)
        return normalized_specs[0] if isinstance(attn, dict) else normalized_specs
