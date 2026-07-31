# RWKV-7 Production Gap Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add verified RWKV-7 prefix caching and runtime LoRA support, pursue a
gated quantized-speed path, and expand reproducible exact-card evidence.

**Architecture:** Reuse vLLM's Mamba state manager, LoRA manager, quantization
configuration, and V1 engine. Keep all promotions fail-closed and card-local;
extend the existing RWKV-7 tests and benchmarks rather than forking framework
subsystems.

**Tech Stack:** Python 3.10/3.12, PyTorch, Triton, vLLM V1, PEFT LoRA, pytest,
Ruff, pre-commit, RTX 4080, and dual Tesla V100-PCIE-32GB.

---

## Task 1: Establish the follow-up baseline

### Files

- Create: `docs/plans/2026-07-31-rwkv7-production-gaps-design.md`
- Create: `docs/plans/2026-07-31-rwkv7-production-gaps.md`

### Steps

1. Create `wangyue/rwkv7-production-gaps` from synchronized `origin/main`.
2. Verify `origin/main` equals the fetched official `upstream/main`.
3. Record the four acceptance lanes and their fail-closed gates.
4. Run markdownlint and `git diff --check` on the plan documents.

## Task 2: Enable aligned recurrent-state prefix caching

### Files

- Modify: `vllm/model_executor/models/rwkv7.py`
- Modify: `tests/model_executor/test_rwkv7.py`
- Modify: `benchmarks/kernels/benchmark_rwkv7.py`
- Modify: `docs/models/supported_models.md`

### Steps

1. Replace the existing negative capability test with a failing assertion that
   pure and hybrid RWKV-7 declare Mamba prefix-caching support.
2. Add a configuration test that requests prefix caching and verifies the
   selected Mamba cache mode and block alignment.
3. Run the focused tests and confirm they fail only because the interface is
   not declared.
4. Add `SupportsMambaPrefixCaching` to the pure and hybrid model interfaces.
5. Run the focused tests and registry tests.
6. Extend the existing engine benchmark with cold/prime/hit execution and
   machine-readable `num_cached_tokens`, latency, and token-parity fields.
7. Run a real RTX 4080 engine test with a shared prefix longer than one block;
   require `num_cached_tokens > 0` and exact continuation tokens.
8. Repeat on V100 and retain `align` mode if the card-local `all` mode is not
   correctness-passing.

## Task 3: Add standard vLLM runtime LoRA adapters

### Files

- Modify: `vllm/model_executor/models/rwkv7.py`
- Modify: `tests/model_executor/test_rwkv7.py`
- Modify: `docs/models/supported_models.md`
- Create: `benchmarks/benchmark_rwkv7_lora.py`

### Steps

1. Add a failing interface test using `supports_lora` for pure and hybrid
   RWKV-7 classes.
2. Assert the expected empty packed-module mapping and embedding/lm-head module
   mappings.
3. Add `SupportsLoRA` to both causal-LM classes and expose the standard module
   metadata without changing checkpoint-native RWKV low-rank weights.
4. Construct a minimal local PEFT-compatible adapter for one FFN projection.
5. Load the adapter through a real V1 engine, generate base/adapter/base rows,
   and assert activation changes output while removal restores the base output.
6. Exercise two adapters in one batch when supported by the available server.
7. Record latency, tokens, adapter target modules, and exact base restoration in
   JSONL; mark LoRA supported in the model table only after the engine gate.

## Task 4: Profile and gate selective weight-only quantization

### Files

- Modify: `benchmarks/benchmark_rwkv7_quantization.py`
- Modify only after profiling proves the need:
  `vllm/model_executor/models/rwkv7.py`
- Modify only for a new kernel:
  `vllm/model_executor/layers/rwkv7.py`
- Modify: `tests/model_executor/test_rwkv7.py`

### Steps

1. Add module-family timing and model-size metadata to the existing quantization
   report.
2. Add CLI policies `memory` and `speed`; make each policy emit its exact set of
   quantized module prefixes.
3. Run 16-bit, online INT8, TorchAO INT8/INT4, and BitsAndBytes rows on RTX 4080
   for the available real checkpoints.
4. Select the smallest profitable module set and rerun batches 1/2/4/8.
5. If no generic policy reaches speed ratio 1.0, add a correctness-first
   RWKV-shaped weight-only kernel for the profitable GEMV shapes and test it
   against `torch.nn.functional.linear`.
6. Require memory reduction, speed ratio at least 1.0, and greedy-token
   agreement 1.0 before labeling a row as the speed lane.
7. Record V100 settings as unsupported or memory-only unless an sm_70 kernel
   passes the same gates.

## Task 5: Expand checkpoint and workload evidence

### Files

- Modify: `benchmarks/kernels/benchmark_rwkv7.py`
- Modify: `benchmarks/results/rwkv7_20260731.jsonl`
- Modify: `docs/models/supported_models.md`

### Steps

1. Inventory locally available real RWKV-7 checkpoints on RTX 4080 and V100.
2. Run prompt 128/512 and decode 16/64 at batches 1/2/4/8 where memory permits.
3. Record throughput, TTFT, decode latency, peak memory, exact tokens, and
   recurrent-state similarity for torch and Triton backends.
4. Repeat prefix-cache and LoRA smoke rows on each checkpoint that fits.
5. Re-run native/hybrid/heterogeneous TP=2 and PP=2 regressions on dual V100.
6. Parse every JSONL line and summarize supported and unsupported axes without
   extrapolating to untested GPU families.

## Task 6: Final regression, publication, and upstream handoff

### Files

- All files changed from `origin/main`.

### Steps

1. Run focused RWKV-7 pytest, registry tests, Ruff, formatting, markdownlint,
   and `git diff --check`.
2. Run the deterministic 512-token recurrent test and real prefix/LoRA/quant
   gates on RTX 4080; run compatible regressions on V100.
3. Fetch official `upstream/main`, reconcile non-overlapping upstream changes,
   and rerun affected checks.
4. Rebuild the downstream range as commits authored and committed only by
   `btlqql <2977859784@qq.com>` with AI-assistance disclosure.
5. Push only after local and exact-card gates pass; do not open a duplicate PR
   against official vLLM while `vllm-project/vllm#50077` remains open.
6. Report the downstream evidence on the existing upstream review thread only
   when the human submitter has reviewed every changed line.
