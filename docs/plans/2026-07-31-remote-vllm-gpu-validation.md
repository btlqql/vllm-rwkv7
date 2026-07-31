# Remote vLLM GPU Validation Implementation Plan

> **For btlqql:** Execute this plan task-by-task and record only observed results.

**Goal:** Validate and repair the clean-room RWKV-7 adapter against the fixed real vLLM Python APIs on Ada and V100 GPU servers.

**Architecture:** Use repository-external remote worktrees and virtual environments. Run fast API/cache contracts first, a constrained existing-local-checkpoint engine gate second, and a sanitized cross-GPU evidence report last.

**Tech Stack:** Linux x86_64, Python 3.10/3.12, uv, PyTorch 2.13 with cu132 on RTX 4080 and cu126 on V100, fixed vLLM, pytest.

---

### Task 1: Primary isolated runtime

**Files:**
- External: `${HOME}/.venvs/vllm-rwkv7-837eae`
- External: `${HOME}/.cache/vllm-rwkv7/upstream/837eae645`
- External: `${HOME}/.cache/vllm-rwkv7/worktree`

1. Create the external virtual environment on `gpu4080` and install current `uv`.
2. Clone only official `vllm-project/vllm` source and detach at `837eae64580c885101ee95b073aafb27a485e7ce`.
3. Install with `VLLM_USE_PRECOMPILED=1` and `VLLM_PRECOMPILED_WHEEL_COMMIT=553fcb82d5602c75fb6ab41b6dc3c46f480c1785` using the official automatic torch backend.
4. Clone the existing PR branch into the external worktree and install it editable with `--no-deps` plus test dependencies.
5. Verify Python, torch, CUDA, compiled vLLM import, source SHA, and GPU capability without printing private paths.

### Task 2: Real fixed-API contracts

**Files:**
- Modify: `tests/integration/test_vllm_contract.py`
- Modify: `tests/integration/test_vllm_registry.py`
- Modify as required: `src/vllm_rwkv7/model.py`

1. Run registry and contract tests on `gpu4080` and capture the first incompatibility.
2. Add a dependency-light regression assertion before each adapter fix.
3. Implement only the minimum independently derived compatibility change.
4. Rerun focused Windows and remote contract tests after every change.

### Task 3: Real cache lifecycle

**Files:**
- Create: `tests/integration/test_vllm_cache_runtime.py`
- Modify as required: `src/vllm_rwkv7/cache.py`
- Modify as required: `src/vllm_rwkv7/model.py`

1. Bind actual vLLM recurrent cache tensors to a tiny model configuration.
2. Verify one-shot versus chunked prefill state and output equality.
3. Verify packed dynamic request reordering selects state solely by slot.
4. Verify a released and reused slot is cleared for a fresh request.
5. Exercise the prefix-cache capability query and temporal copy specifications through the fixed API, without claiming unsupported `all` mode.

### Task 4: Existing local checkpoint and Ada engine

**Files:**
- Modify: `tests/integration/test_vllm_engine.py`
- Create after successful run: `bench/results/remote_gpu_vllm_20260731.json`

1. Add environment-gated checkpoint loading and deterministic generation assertions.
2. Add short-context chunked-prefill, two-request scheduling, release/reuse, and prefix-cache cases exposed by the public API.
3. Verify the existing local 0.1B HF checkpoint's config, keys, size, and SHA; transfer only config, generation config, and safetensors into the external cache.
4. Run on `gpu4080` with `trust_remote_code=False`, eager TP=1, conservative memory utilization, short context, and at most four sequences.
5. Emit sanitized exact environment, revisions, memory, outcomes, and skipped gates.

### Task 5: V100 compatibility regression

**Files:**
- Modify: `bench/results/remote_gpu_vllm_20260731.json`

1. Create a separate isolated fixed environment on `WZU_Server`.
2. Run real plugin, API, cache, PyTorch CUDA, and backend-policy tests.
3. Run the constrained engine gate with a safe test-only memory reservation that does not interfere with other GPU workloads.
4. Record exact V100 results and explicit blockers without exposing connection details.

### Task 6: Documentation and publication

**Files:**
- Modify: `README.md`
- Modify: `docs/testing.md`
- Modify: `tests/test_documentation.py`

1. Document fixed-source/precompiled-binary distinction and sanitized commands.
2. Add tests for evidence schema and honest pass/skip semantics.
3. Run Ruff, format, full pytest, wheel build, and provenance on Windows.
4. Run every available integration and GPU gate on both remote hosts.
5. Audit Git identities, source references, staged changes, and checkpoint exclusion.
6. Commit as btlqql, push the existing branch, and comment on PR #1 with exact observed results.
