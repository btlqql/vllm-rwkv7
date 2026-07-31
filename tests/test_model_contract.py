from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as functional
from torch import nn


def test_model_module_is_present_for_lazy_registry_target() -> None:
    assert importlib.util.find_spec("vllm_rwkv7.model") is not None


def test_vllm_model_interfaces_when_vllm_is_importable() -> None:
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("vLLM is not installed in this CPU unit-test environment")

    from vllm.model_executor.models.interfaces import has_inner_state, is_attention_free

    from vllm_rwkv7.model import RWKV7ForCausalLM

    assert has_inner_state(RWKV7ForCausalLM)
    assert is_attention_free(RWKV7ForCausalLM)


def _module(monkeypatch, name: str, **attributes):
    module = ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_model_constructs_and_runs_against_documented_vllm_contract(monkeypatch) -> None:
    class FakeMambaBase:
        pass

    class FakeHasInnerState:
        has_inner_state = True

    class FakeIsAttentionFree:
        is_attention_free = True

    class FakeSupportsMambaPrefixCaching:
        supports_mamba_prefix_caching = True

    class FakeEmbedding(nn.Embedding):
        def __init__(self, num_embeddings, embedding_dim, **_):
            super().__init__(num_embeddings, embedding_dim)

    class FakeLMHead(FakeEmbedding):
        pass

    class FakeLogitsProcessor:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size

        def __call__(self, lm_head, hidden_states):
            return functional.linear(hidden_states, lm_head.weight)

    class FakeWeightsLoader:
        last_mapper = None

        def __init__(self, model):
            self.model = model

        def load_weights(self, weights, mapper=None):
            type(self).last_mapper = mapper
            return {name for name, _ in weights}

    class FakeWeightsMapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeMambaAttentionBackendEnum:
        LINEAR = "linear"

    _module(monkeypatch, "vllm")
    _module(monkeypatch, "vllm.config", VllmConfig=object)
    forward_context = SimpleNamespace(attn_metadata=None)
    _module(
        monkeypatch,
        "vllm.forward_context",
        get_forward_context=lambda: forward_context,
    )
    _module(monkeypatch, "vllm.model_executor")
    _module(monkeypatch, "vllm.model_executor.layers")
    _module(
        monkeypatch,
        "vllm.model_executor.layers.logits_processor",
        LogitsProcessor=FakeLogitsProcessor,
    )
    _module(monkeypatch, "vllm.model_executor.layers.mamba")
    _module(
        monkeypatch,
        "vllm.model_executor.layers.mamba.abstract",
        MambaBase=FakeMambaBase,
    )
    _module(
        monkeypatch,
        "vllm.model_executor.layers.mamba.mamba_utils",
        get_temporal_copy_spec=lambda *_: None,
    )
    _module(
        monkeypatch,
        "vllm.model_executor.layers.vocab_parallel_embedding",
        ParallelLMHead=FakeLMHead,
        VocabParallelEmbedding=FakeEmbedding,
    )
    _module(monkeypatch, "vllm.model_executor.models")
    _module(
        monkeypatch,
        "vllm.model_executor.models.interfaces",
        HasInnerState=FakeHasInnerState,
        IsAttentionFree=FakeIsAttentionFree,
        SupportsMambaPrefixCaching=FakeSupportsMambaPrefixCaching,
    )
    _module(
        monkeypatch,
        "vllm.model_executor.models.utils",
        AutoWeightsLoader=FakeWeightsLoader,
        WeightsMapper=FakeWeightsMapper,
    )
    _module(monkeypatch, "vllm.v1")
    _module(monkeypatch, "vllm.v1.attention")
    _module(monkeypatch, "vllm.v1.attention.backends")
    _module(
        monkeypatch,
        "vllm.v1.attention.backends.registry",
        MambaAttentionBackendEnum=FakeMambaAttentionBackendEnum,
    )

    monkeypatch.delitem(sys.modules, "vllm_rwkv7.model", raising=False)
    from vllm_rwkv7.model import RWKV7ForCausalLM

    hf_config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=4,
        attention_hidden_size=8,
        intermediate_size=24,
        decay_low_rank_dim=4,
        gate_low_rank_dim=4,
        a_low_rank_dim=4,
        v_low_rank_dim=4,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=False,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_config,
            dtype=torch.float32,
            enforce_eager=True,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        cache_config=SimpleNamespace(),
        compilation_config=SimpleNamespace(static_forward_context={}),
        scheduler_config=SimpleNamespace(),
    )
    model = RWKV7ForCausalLM(vllm_config=vllm_config).eval()
    token_ids = torch.tensor([1, 2, 3])
    hidden_states = model(token_ids, positions=torch.arange(3))
    logits = model.compute_logits(hidden_states)

    assert hidden_states.shape == (3, 16)
    assert logits.shape == (3, 32)
    assert model.config is hf_config
    assert model.model_config is vllm_config.model_config
    assert model.supports_mamba_prefix_caching is True
    assert model.hf_to_vllm_mapper.kwargs == {}
    public_keys = {
        "lm_head.weight",
        "model.embeddings.weight",
        "model.layers.0.attn.x_r",
        "model.layers.0.attn.w_lora.lora.0.weight",
        "model.layers.0.attn.w_lora.lora.2.bias",
        "model.layers.0.attn.g_norm.bias",
        "model.layers.0.ffn.key.weight",
        "model.layers.0.pre_norm.bias",
        "model.norm.bias",
    }
    assert public_keys <= set(model.state_dict())
    assert tuple(vllm_config.compilation_config.static_forward_context) == (
        "model.layers.0",
        "model.layers.1",
    )
    assert model.get_mamba_state_shape_from_config(vllm_config) == (
        (2, 16),
        (2, 4, 4),
    )
    loaded = model.load_weights([("model.embeddings.weight", torch.empty(0))])
    assert loaded == {"model.embeddings.weight"}
    assert FakeWeightsLoader.last_mapper is model.hf_to_vllm_mapper

    metadata = SimpleNamespace(
        num_decode_tokens=0,
        num_decodes=0,
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([2, 1], dtype=torch.int32),
        state_indices_tensor=torch.tensor([1, 0], dtype=torch.int32),
    )
    forward_context.attn_metadata = {
        layer_name: metadata for layer_name in vllm_config.compilation_config.static_forward_context
    }
    for layer in model.model.layers:
        layer.kv_cache = (
            torch.zeros(3, 2, 16),
            torch.zeros(3, 2, 4, 4, dtype=torch.float32),
        )

    packed_hidden = model(torch.tensor([4, 5, 6]), positions=torch.arange(3))

    assert packed_hidden.shape == (3, 16)
    for layer in model.model.layers:
        shift_cache, matrix_cache = layer.kv_cache
        assert torch.count_nonzero(shift_cache[0]) > 0
        assert torch.count_nonzero(shift_cache[1]) > 0
        assert torch.count_nonzero(matrix_cache[0]) > 0
        assert torch.count_nonzero(matrix_cache[1]) > 0
        assert torch.count_nonzero(shift_cache[2]) == 0
        assert torch.count_nonzero(matrix_cache[2]) == 0

    new_prompt = torch.tensor([7, 8, 9])
    forward_context.attn_metadata = None
    expected = model(new_prompt, positions=torch.arange(3))
    new_sequence_metadata = SimpleNamespace(
        num_decode_tokens=0,
        num_decodes=0,
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        state_indices_tensor=torch.tensor([2], dtype=torch.int32),
    )
    forward_context.attn_metadata = {
        layer_name: new_sequence_metadata
        for layer_name in vllm_config.compilation_config.static_forward_context
    }
    for layer in model.model.layers:
        layer.kv_cache[0][2].fill_(7.0)
        layer.kv_cache[1][2].fill_(5.0)

    actual = model(new_prompt, positions=torch.arange(3))

    torch.testing.assert_close(actual, expected)

    chunked_tokens = torch.tensor([10, 11, 12, 13])
    forward_context.attn_metadata = None
    expected_chunked = model(chunked_tokens, positions=torch.arange(4))
    for layer in model.model.layers:
        layer.kv_cache[0].zero_()
        layer.kv_cache[1].zero_()

    first_chunk_metadata = SimpleNamespace(
        num_decode_tokens=0,
        num_decodes=0,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        state_indices_tensor=torch.tensor([1], dtype=torch.int32),
    )
    forward_context.attn_metadata = {
        layer_name: first_chunk_metadata
        for layer_name in vllm_config.compilation_config.static_forward_context
    }
    first_chunk = model(chunked_tokens[:2], positions=torch.arange(2))
    second_chunk_metadata = SimpleNamespace(
        num_decode_tokens=0,
        num_decodes=0,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        state_indices_tensor=torch.tensor([1], dtype=torch.int32),
    )
    forward_context.attn_metadata = {
        layer_name: second_chunk_metadata
        for layer_name in vllm_config.compilation_config.static_forward_context
    }
    second_chunk = model(chunked_tokens[2:], positions=torch.arange(2, 4))
    torch.testing.assert_close(torch.cat((first_chunk, second_chunk)), expected_chunked)

    request_a = torch.tensor([14, 15, 16])
    request_b = torch.tensor([17, 18, 19])
    forward_context.attn_metadata = None
    expected_a = model(request_a, positions=torch.arange(3))
    expected_b = model(request_b, positions=torch.arange(3))
    for layer in model.model.layers:
        layer.kv_cache[0].zero_()
        layer.kv_cache[1].zero_()

    first_reordered_metadata = SimpleNamespace(
        num_decode_tokens=0,
        num_decodes=0,
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([2, 1], dtype=torch.int32),
        state_indices_tensor=torch.tensor([2, 0], dtype=torch.int32),
    )
    forward_context.attn_metadata = {
        layer_name: first_reordered_metadata
        for layer_name in vllm_config.compilation_config.static_forward_context
    }
    model(torch.cat((request_a[:2], request_b[:1])), positions=torch.arange(3))

    second_reordered_metadata = SimpleNamespace(
        num_decode_tokens=0,
        num_decodes=0,
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([3, 3], dtype=torch.int32),
        state_indices_tensor=torch.tensor([0, 2], dtype=torch.int32),
    )
    forward_context.attn_metadata = {
        layer_name: second_reordered_metadata
        for layer_name in vllm_config.compilation_config.static_forward_context
    }
    reordered = model(
        torch.cat((request_b[1:], request_a[2:])),
        positions=torch.arange(3),
    )
    torch.testing.assert_close(reordered[:2], expected_b[1:])
    torch.testing.assert_close(reordered[2:], expected_a[2:])

    vllm_config.model_config.enforce_eager = False
    with pytest.raises(NotImplementedError, match="enforce_eager=True"):
        RWKV7ForCausalLM(vllm_config=vllm_config)

    vllm_config.model_config.enforce_eager = True
    vllm_config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(NotImplementedError, match="pipeline_parallel_size=1"):
        RWKV7ForCausalLM(vllm_config=vllm_config)

    vllm_config.parallel_config.pipeline_parallel_size = 1
    vllm_config.speculative_config = SimpleNamespace()
    with pytest.raises(NotImplementedError, match="speculative decoding"):
        RWKV7ForCausalLM(vllm_config=vllm_config)
