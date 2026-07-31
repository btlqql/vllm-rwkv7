# Clean-room RWKV-7 vLLM Plugin Implementation Plan

**Goal:** Build an independently authored, Hugging Face-compatible RWKV-7 model plugin for current vLLM without importing previous RWKV-7 pull-request code.

**Architecture:** Publish a small `src/` Python package with a re-entrant vLLM entry point, a strict HF configuration normalizer, an executable PyTorch recurrence oracle, and a vLLM stateful model. Add fused backends only behind per-GPU policy after the reference path passes.

**Tech Stack:** Python 3.10+, PyTorch, Transformers configuration objects, vLLM general plugins, pytest, Ruff.

---

### Task 1: Repository and package contract

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `AUTHORS.md`
- Create: `src/vllm_rwkv7/__init__.py`
- Test: `tests/test_package_metadata.py`

1. Write tests for the distribution name, entry point, architecture names, and sole-author metadata.
2. Run the metadata test and confirm it fails while the files are absent.
3. Add the minimum package metadata and public constants.
4. Run the test and confirm it passes.

### Task 2: Hugging Face configuration normalization

**Files:**
- Create: `src/vllm_rwkv7/config.py`
- Test: `tests/test_config.py`

1. Write failing tests for canonical `num_attention_heads`, legacy `num_heads`, distinct `attention_hidden_size`, defaults, and invalid shapes.
2. Implement an immutable normalized configuration with actionable errors.
3. Verify JSON-like mappings and attribute-based Transformers configs behave identically.

### Task 3: RWKV-7 recurrence oracle

**Files:**
- Create: `src/vllm_rwkv7/reference.py`
- Test: `tests/test_reference.py`

1. Write failing shape/dtype tests and a direct scalar recurrence example.
2. Implement the fp32 state update and model-dtype output calculation from the public RWKV-7 equation.
3. Test token-by-token and chunked execution equality, including nonzero state.

### Task 4: Re-entrant vLLM registration

**Files:**
- Create: `src/vllm_rwkv7/plugin.py`
- Test: `tests/test_plugin.py`

1. Stub `vllm.ModelRegistry` and verify lazy registration.
2. Verify repeated plugin loading does not duplicate or replace registration.
3. Register canonical and explicitly documented legacy architecture aliases.

### Task 5: Stateful vLLM model

**Files:**
- Create: `src/vllm_rwkv7/model.py`
- Test: `tests/test_model_contract.py`
- Test: `tests/integration/test_vllm_registry.py`

1. Implement HF-compatible module and parameter names.
2. Expose shift and fp32 matrix state through vLLM's Mamba-like cache contract.
3. Implement packed prefill/decode sequencing using cache-slot metadata.
4. Add strict P0 guards for TP=1 and unsupported speculative modes.
5. Verify registry inspection on Linux; keep tiny-checkpoint loading as the
   next end-to-end integration gate.

### Task 6: Per-GPU backend policy

**Files:**
- Create: `src/vllm_rwkv7/kernel_policy.py`
- Test: `tests/test_kernel_policy.py`
- Create: `docs/kernel-policy.md`

1. Add explicit Volta, Turing, Ampere, Ada, Hopper, Blackwell, and AMD classes.
2. Default to the reference backend unless exact-card evidence promotes a fused backend.
3. Add environment overrides and invalid-value tests.

### Task 7: Validation and publication

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `scripts/check_provenance.py`
- Create: `docs/testing.md`

1. Run Ruff and the CPU test suite.
2. Test against the pinned vLLM main revision on Linux.
3. Run V100 correctness before performance claims.
4. Audit Git history and staged diff for unexpected authors, co-author trailers, copied patches, secrets, and generated artifacts.
5. Publish only after all available gates pass.

## Current progress

Tasks 1 through 6 are implemented. The optional FLA adapter has exact-card
operator and tiny-layer correctness evidence on an RTX 5070 Laptop GPU. Task 7
still requires Linux vLLM model loading and a converted-checkpoint end-to-end
run before any production backend promotion.
