# RWKV-7 vLLM Contract Completion Implementation Plan

**Goal:** Complete all locally verifiable vLLM plugin, configuration, weight-loading, and recurrent-cache contracts while preparing honest external integration gates.

**Architecture:** Keep formal vLLM integration thin and move packed-state planning into a pure, validated module. Preserve exact public Hugging Face weight names and use vLLM's identity `WeightsMapper` for loading.

**Tech Stack:** Python 3.10+, PyTorch, vLLM pinned API contracts, pytest, Ruff, GitHub Actions.

---

### Task 1: PR provenance checkout

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_documentation.py`

1. Add a failing test that requires pull-request CI to check out the head SHA.
2. Run `python -m pytest tests/test_documentation.py -q` and confirm failure.
3. Configure checkout `ref` to use the PR head SHA with `github.sha` fallback.
4. Rerun the focused test.

### Task 2: Public Hugging Face configuration contract

**Files:**
- Modify: `src/vllm_rwkv7/config.py`
- Modify: `src/vllm_rwkv7/components.py`
- Modify: `tests/test_config.py`

1. Add failing tests for `norm_eps`, nullable head count inference, non-integral dimensions, norm bias, and unsupported activation/norm ordering.
2. Implement strict normalization and early compatibility errors.
3. Apply normalized norm bias to every layer norm.
4. Run `python -m pytest tests/test_config.py tests/test_components.py -q`.

### Task 3: Explicit weight and model interface contract

**Files:**
- Modify: `src/vllm_rwkv7/model.py`
- Modify: `tests/test_model_contract.py`
- Modify: `tests/integration/test_vllm_registry.py`

1. Extend the vLLM stub with `WeightsMapper` and the prefix-caching protocol.
2. Add failing assertions for the identity mapper, public checkpoint keys, model config attributes, and prefix-cache capability.
3. Pass the identity mapper to `AutoWeightsLoader` and expose the formal marker.
4. Run the focused model and plugin contract tests.

### Task 4: Packed recurrent-state planner

**Files:**
- Create: `src/vllm_rwkv7/cache.py`
- Modify: `src/vllm_rwkv7/model.py`
- Create: `tests/test_cache.py`
- Modify: `tests/test_model_contract.py`

1. Write failing tests for valid packed spans, reordered slots, padding, duplicate slots, out-of-range slots, and malformed boundaries.
2. Implement immutable span planning with complete validation.
3. Replace ad-hoc metadata parsing in the model with the planner.
4. Add model-stub tests for two-stage chunked prefill and released-slot reuse.
5. Run the cache and model contract tests.

### Task 5: Honest external integration entry points and docs

**Files:**
- Create: `tests/integration/test_vllm_contract.py`
- Create: `tests/integration/test_vllm_engine.py`
- Modify: `docs/testing.md`
- Modify: `README.md`
- Modify: `pyproject.toml`

1. Add a Linux vLLM contract test for registered capabilities and layer cache interfaces.
2. Add an environment-gated real-engine smoke test using a user-provided local tiny checkpoint.
3. Document which checks are local and which require Linux vLLM or exact GPUs.
4. Run the complete local suite and confirm external tests skip with explicit reasons.

### Task 6: Final verification and publication

**Files:**
- Modify only files already in scope if verification finds defects.

1. Run `python -m ruff check .`.
2. Run `python -m ruff format --check .`.
3. Run `python -m pytest -q`.
4. Run `python -m pip wheel --no-deps . -w dist`.
5. Run `python scripts/check_provenance.py` and audit Git identities.
6. Commit as `btlqql`, push `btlqql/cleanroom-rwkv7`, and comment on PR #1 with the patchset and exact results.
