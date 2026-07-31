# RWKV-7 Downstream Completion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring downstream PR #11 onto the latest official vLLM base, make its
history and documentation btlqql-only, obtain green CI, and close the remaining
RWKV-7 correctness, distributed, quantization, benchmark, and checkpoint-shape
gaps with reproducible RTX 4080 and V100 evidence.

**Architecture:** Keep official vLLM `main` as the immutable base and publish
the complete RWKV-7 diff as one squashed commit authored and committed by
`btlqql <2977859784@qq.com>`. Reuse vLLM's Mamba state manager, distributed
linear layers, attention implementations, and quantization interfaces. Add
compatibility only behind tests, retain fail-closed kernel dispatch, and record
exact-card evidence in the existing benchmark/reporting surfaces and PR body.

**Tech Stack:** Python 3.10/3.12, PyTorch, Triton, vLLM V1, pytest, Ruff,
pre-commit, GitHub Actions, RTX 4080, and 2x Tesla V100-PCIE-32GB.

---

## Task 1: Synchronize the official base and correct repository history docs

### Files

- Modify: `docs/plans/2026-07-31-rwkv7-upstream-sync-design.md`
- Modify: `docs/plans/2026-07-31-rwkv7-upstream-sync.md`
- Modify: `docs/plans/2026-07-31-rwkv7-completion.md`

### Steps

1. Fast-forward `origin/main` to the latest fetched `upstream/main`.
2. Rebuild the feature commit directly on that base without changing its tree.
3. Replace every claim about preserved third-party ancestry with the current
   btlqql-only branch-history contract and upstream-reference disclosure.
4. Run `git diff --check` and audit author plus committer for `main..HEAD`.
5. Keep the final published branch as one btlqql commit.

## Task 2: Unblock and run GitHub pre-commit CI

### Files

- No source modification unless a real hook failure identifies one.

### Steps

1. Create/apply the downstream `ready` label required by the inherited workflow.
2. Re-run the PR workflow and inspect its logs with `gh`.
3. Reproduce each real hook failure locally or on the server.
4. Apply the smallest fix and rerun until all checks are green.

## Task 3: Add long-horizon recurrent and cache parity gates

### Files

- Modify: `tests/model_executor/test_rwkv7.py`
- Modify only if a failing test proves a defect:
  `vllm/model_executor/layers/rwkv7.py`

### Steps

1. Add deterministic long-sequence packed-scan tests against the torch oracle.
2. Add multi-chunk prefill-to-decode state/output continuity coverage.
3. Run CPU tests, then CUDA tests on RTX 4080 and V100.
4. Keep automatic Triton dispatch fail-closed unless the exact-card tolerance
   and greedy-token gates pass.

## Task 4: Complete exact-card benchmark evidence

### Files

- Modify: `benchmarks/kernels/benchmark_rwkv7.py`
- Create: `benchmarks/results/rwkv7_20260731.jsonl`

### Steps

1. Make the benchmark emit machine-readable GPU, software, shape, latency,
   throughput, peak-memory, state-diff, and output-similarity fields.
2. Run fp16 prompt/decode and dynamic-batch rows on RTX 4080 and V100.
3. Record deterministic rows in JSONL and summarize them in the PR body.

## Task 5: Validate tensor and pipeline parallel execution

### Files

- Modify: `tests/model_executor/test_rwkv7.py` only for reusable unit coverage.
- Record: `benchmarks/results/rwkv7_20260731.jsonl`

### Steps

1. Run local shape/weight-loading tests for TP and PP partitioning.
2. Run real 2-GPU `tensor_parallel_size=2` generation on the V100 server.
3. Run real 2-GPU `pipeline_parallel_size=2` generation on the V100 server.
4. Compare greedy tokens with the single-GPU reference.

## Task 6: Complete INT8 and INT4 end-to-end evidence

### Files

- Modify: `benchmarks/benchmark_rwkv7_quantization.py`
- Record: `benchmarks/results/rwkv7_20260731.jsonl`
- Modify quantization implementation only for reproduced correctness defects.

### Steps

1. Add JSONL output for fp16, online INT8, and TorchAO INT4 rows.
2. Record model footprint, peak VRAM, prefill/decode throughput, logits error,
   and greedy-token agreement.
3. Run supported rows on RTX 4080 and V100 and label unsupported kernels
   explicitly rather than silently falling back.

## Task 7: Support heterogeneous value dimensions and hybrid attention configs

### Files

- Modify: `vllm/transformers_utils/configs/rwkv7.py`
- Modify: `vllm/model_executor/models/rwkv7.py`
- Modify: `tests/model_executor/test_rwkv7.py`

### Steps

1. Add failing config/model tests for per-layer `value_dim` and `attn` layouts.
2. Represent heterogeneous recurrent states with a padded/cache-safe maximum
   dimension while slicing each layer to its declared logical dimension.
3. Route configured hybrid-attention layers through existing vLLM attention
   components without changing pure-RWKV checkpoint behavior.
4. Add loading, forward, cache, TP, and PP tests for both configurations.
5. Run the focused suite on CPU and CUDA.

## Task 8: Final audit and publish one btlqql patchset

### Files

- All files changed relative to `origin/main`.

### Steps

1. Run Ruff, formatting, focused pytest, and GitHub pre-commit.
2. Verify the worktree is clean and `git diff --check` passes.
3. Squash the entire tree to one commit whose author and committer are
   `btlqql <2977859784@qq.com>` and include AI-assistance disclosure.
4. Force-push with an explicit lease, update PR #11, and verify its commit list.
5. Mark the PR ready only after required checks are green.

## Completion Evidence

- Latest official base synchronized at
  `10e6b400150c8d2cbedad54260def4871d464667` before final publication.
- RWKV7 focused suite: 30 passed and 2 skipped on Tesla V100-PCIE-32GB;
  the deterministic 512-token Triton long-horizon test also passed on RTX
  4080.
- Native and hybrid model registry imports passed. The generic HF initialization
  parametrization could not access Hugging Face from the V100 host, while the
  equivalent local native and synthetic-hybrid checkpoints initialized and
  generated successfully through the real vLLM engine.
- Native, hybrid-attention, and heterogeneous-`value_dim` checkpoints passed
  cached greedy generation with both TP=2 and PP=2 on two V100 GPUs, matching
  their single-GPU reference tokens.
- Exact-card torch-versus-Triton, quantization, compatibility, and distributed
  rows are recorded in `benchmarks/results/rwkv7_20260731.jsonl`.
- Online INT8, bitsandbytes NF4, and TorchAO INT4 reduced the measured 0.1B
  model footprint on RTX 4080, but none beat the W16 throughput reference;
  NF4 and INT4 also missed the greedy-token quality gate. V100 online INT8 is
  explicitly recorded as unsupported because its compute capability is 7.0.
