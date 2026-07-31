# RWKV-7 Upstream Sync Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert `btlqql/vllm-rwkv7` into an official-vLLM-based downstream and integrate the existing native RWKV-7 implementation without duplicating or reattributing it.

**Architecture:** Keep `origin/main` synchronized to official `upstream/main`, preserve the retired standalone plugin on an archive branch, and carry RWKV-7 on a feature branch. Use upstream PR #50077 as the implementation reference, publish the resulting downstream delta only as btlqql-authored/committed branch history, resolve current-main registry conflicts, and validate through existing vLLM model, recurrent-state, compilation, and quantization tests.

**Tech Stack:** Git, Python 3.12 through `uv`, PyTorch, CUDA, vLLM V1, pytest, pre-commit, GitHub Actions.

---

## Task 1: Preserve and synchronize repository history

### Files

- No source-file changes.

### Step 1: Record exact refs

Run:

```bash
git ls-remote origin refs/heads/main \
  refs/heads/wangyue/cleanroom-prototype-archive
git rev-parse upstream/main
```

Expected: the archive points to `ca0b7ad034e0b35fb1aa9a23c60bfcbda7c92b51`
and both main refs point to the selected official baseline.

### Step 2: Verify remote topology

Run `git remote -v`.

Expected: `origin` is `btlqql/vllm-rwkv7`; `upstream` is
`vllm-project/vllm`.

## Task 2: Integrate existing upstream RWKV-7 work

### Files

- Integrate into the btlqql-owned downstream patch: `vllm/model_executor/models/rwkv7.py`
- Integrate into the btlqql-owned downstream patch: `vllm/model_executor/layers/rwkv7.py`
- Integrate into the btlqql-owned downstream patch: `vllm/transformers_utils/configs/rwkv7.py`
- Integrate into the btlqql-owned downstream patch: `tests/model_executor/test_rwkv7.py`
- Integrate into the btlqql-owned downstream patch: the RWKV-7 benchmark and quantization files referenced by PR #50077

### Step 1: Fetch the reviewed upstream head

Run:

```bash
git fetch upstream pull/50077/head:refs/remotes/upstream/pr-50077
git rev-parse upstream/pr-50077
```

Expected head: `d43cdf2a2ea3361c00119c61f30ec60cb9e7ab62` unless the
upstream PR has advanced, in which case record and inspect the new head first.

### Step 2: Reconcile the reviewed source tree without importing its commits

Run:

```bash
git diff --stat upstream/main...upstream/pr-50077
git diff upstream/main...upstream/pr-50077 -- <audited-rwkv7-paths>
```

Expected: the reviewed implementation is used as an explicit source reference,
then reconciled into the downstream working tree. Its feature-branch commits
are not merged or cherry-picked; `origin/main..HEAD` remains btlqql-only.

## Task 3: Resolve current-main registry conflicts

### Files

- Modify: `vllm/config/compilation.py`
- Modify: `vllm/model_executor/models/registry.py`
- Modify: `vllm/transformers_utils/config.py`

### Step 1: Preserve current splitting ops and add RWKV-7

Keep every current upstream entry in `CompilationConfig._attention_ops` and
insert:

```python
"vllm::rwkv7_block_forward",
```

### Step 2: Preserve current model registrations and add RWKV-7

Add the text-generation registration next to the existing `RWForCausalLM`
entry:

```python
"RWKV7ForCausalLM": ("rwkv7", "RWKV7ForCausalLM"),
```

### Step 3: Preserve current config registrations and add RWKV-7

Add to `_CONFIG_REGISTRY`:

```python
rwkv7="RWKV7Config",
```

### Step 4: Verify all conflicts are resolved

Run `git diff --check` and `git status --short`.

Expected: no conflict markers and only intended merged/plan files are changed.

## Task 4: Run the cheapest behavioral gates first

### Files

- Test: `tests/model_executor/test_rwkv7.py`
- Test: `tests/models/test_registry.py`
- Test: `tests/models/test_initialization.py`
- Test: `tests/quantization/test_online.py`
- Test: `tests/quantization/test_torchao.py`

### Step 1: Prepare a supported environment

On Linux, run:

```bash
uv venv --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install -r requirements/test/cuda.in
uv pip install -r requirements/lint.txt
```

### Step 2: Run registry and initialization tests

Run:

```bash
.venv/bin/python -m pytest -v tests/models/test_registry.py \
  -k RWKV7ForCausalLM
.venv/bin/python -m pytest -v tests/models/test_initialization.py \
  -k RWKV7ForCausalLM
```

Expected: RWKV-7 imports, registers, and initializes with dummy weights.

### Step 3: Run the focused RWKV-7 and quantization suite

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -v \
  tests/model_executor/test_rwkv7.py \
  tests/quantization/test_online.py \
  tests/quantization/test_torchao.py
```

Expected: CPU tests pass; GPU- or external-reference-only tests either run on
the selected server or skip for an explicit environment reason.

## Task 5: Validate real serving and exact-card behavior

### Files

- Modify only if a current-main compatibility defect is reproduced by a test.
- Record downstream evidence in the PR body or a dedicated benchmark artifact.

### Step 1: Run a real eager engine smoke

Use an external HF-format RWKV-7 checkpoint and deterministic generation.
Verify model resolution, health endpoint, completion, chunked prefill, and
recurrent-state continuity.

### Step 2: Run exact-card rows independently

Run RTX 4080 and V100 correctness before performance. Record GPU, driver,
CUDA, PyTorch, vLLM commit, checkpoint hash, dtype, prompt/decode dimensions,
greedy-token agreement, peak memory, prefill throughput, and decode throughput.

### Step 3: Fail closed on unproven kernels

Keep automatic dispatch on the reference implementation unless the fused path
passes long-horizon state/output parity on that exact GPU family.

## Task 6: Lint, review, and publish the downstream patchset

### Files

- All files changed relative to `origin/main`.

### Step 1: Run changed-file checks

Run `pre-commit run --files <changed-files>` and
`pre-commit run mypy-3.12 --files <changed-python-files> --hook-stage manual`.

Expected: all applicable hooks pass.

### Step 2: Audit scope and attribution

Run:

```bash
git diff --check origin/main...HEAD
git log --format='%h %an <%ae> %s' origin/main..HEAD
git diff --stat origin/main...HEAD
```

Expected: every commit in `origin/main..HEAD` uses
`btlqql <2977859784@qq.com>` as both author and committer.

### Step 3: Commit downstream-only integration work

Create a human-reviewed compatibility commit signed off by btlqql and include
the required AI-assistance attribution. Squash any source-reference ancestry so
the downstream branch range remains btlqql-only.

### Step 4: Push and open a downstream pull request

Push `wangyue/rwkv7-upstream-adapter` to `btlqql/vllm-rwkv7` and open a PR
against that repository's `main`. State explicitly that the change references
upstream PR #50077, is not a competing official-vLLM submission, keeps the
downstream commit range btlqql-only, and lists exact test/evaluation results.
