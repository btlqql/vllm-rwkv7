# RWKV-7 Production Gap Closure Design

## Scope and acceptance

This follow-up closes the downstream gaps left after RWKV-7 native serving was
merged into `btlqql/vllm-rwkv7`. The work remains based directly on official
vLLM and reuses its model, cache, LoRA, quantization, and distributed runtime
interfaces. It does not create a competing official-vLLM pull request while
`vllm-project/vllm#50077` is open.

The deliverable has four independently gated lanes. Prefix caching must produce
a real cache hit and exactly the same greedy continuation as a cold run. Runtime
LoRA must load, activate, switch, and remove a standard adapter without
confusing RWKV-7's checkpoint-native low-rank modules with user adapters.
Quantization must report compatibility, footprint, throughput, and token
agreement separately; a speed claim requires all three production gates to
pass. The validation matrix must expand beyond the existing 0.1B short-prompt
rows using the RTX 4080 and dual-V100 servers that are currently available.

Official review is an external dependency, not a local completion gate. The
local completion gate is a clean downstream patch, reproducible evidence, and
no unsupported default promotion.

## Architecture and data flow

RWKV-7 already implements vLLM's `HasInnerState` contract and supplies state
shape, dtype, and copy functions. Prefix caching therefore uses the existing
Mamba block manager by declaring `SupportsMambaPrefixCaching`. An RWKV-specific
metadata builder preserves and restores the three recurrent state tensors at
every allocated block boundary. The same block-aware path handles pure RWKV
attention state and the recurrent FFN state in hybrid attention/RWKV models.
No scheduler or cache-engine fork is introduced, and both forms use Mamba cache
mode `all`.

Runtime LoRA uses `SupportsLoRA` on both pure and hybrid causal-LM classes. The
model exposes its un-packed linear topology and embedding/output module names
to the existing LoRA manager. Checkpoint-native modules such as `w_lora` remain
ordinary base-model submodules. User adapters target the actual leaf linears,
for example attention projections and FFN key/value projections, using their
normal PEFT names.

Quantization stays behind vLLM's quantization configuration. The benchmark can
full-match a requested module-name regex so memory and selective-speed rows
state exactly which modules were quantized. A selective speed row quantizes
only profitable modules; the memory row may quantize more modules but is
reported separately. If the available generic kernels cannot beat the 16-bit
row, the design permits an RWKV-shaped fused weight-only GEMV kernel, but only
after a correctness oracle and representative shape sweep exist.

## Failure handling and rollout

Every new capability is fail-closed. Prefix caching is not advertised until a
non-vacuous hit is observed and continuation tokens match. A missing or
incompatible LoRA target fails during adapter loading rather than silently
skipping tensors. Quantized settings record an explicit unsupported reason for
incompatible devices and never fall back to a different precision under the
same result label.

Automatic quantized-speed selection is card-local. The RTX 4080 result cannot
set a V100, Ampere, Hopper, Blackwell, or AMD default. V100 online INT8 remains
unsupported unless a kernel with compute-capability 7.0 support is implemented
and validated. BitsAndBytes and TorchAO remain memory/compatibility lanes when
their speed or quality gates fail.

Documentation reports negative results alongside positive ones. Existing
behavior remains the default during development; capability flags are promoted
only after unit, engine, exact-card, and regression gates pass. Upstream main is
fetched again before publication, and the final downstream range remains
authored and committed only by `btlqql <2977859784@qq.com>`.

## Verification strategy

The cheapest tests guard interfaces and state semantics in
`tests/model_executor/test_rwkv7.py`. GPU engine tests then prove non-vacuous
prefix hits, LoRA activation, recurrent continuity, and token parity. Existing
kernel and quantization benchmark entrypoints are extended instead of creating
test-only performance scripts.

The minimum exact-card matrix is RTX 4080 and V100, prompt lengths 128 and 512,
decode lengths 16 and 64, and batches 1, 2, 4, and 8 where memory permits. The
matrix includes the available real RWKV-7 checkpoints, not only synthetic
configs. Each row records GPU/software identity, model shape, latency,
throughput, peak memory, cache-hit tokens, and greedy-token agreement. TP=2 and
PP=2 regressions remain mandatory on dual V100 for pure, hybrid, and
heterogeneous-value-dimension checkpoints.

Static completion requires Ruff, format, mypy-compatible annotations,
markdownlint, `git diff --check`, focused pytest, valid JSONL, and the inherited
GitHub pre-commit workflow. Model-affecting changes additionally require the
existing reference-parity and deterministic long-horizon gates.

## Completion evidence and claim boundaries

The RTX 4080 engine gates pass for both pure and hybrid models. Pure RWKV-7
reuses 112 prompt tokens and produces an exact cold/hit continuation. The
hybrid model reuses 2048 prompt tokens, also exactly, and reduces the Torch
prefill-plus-decode row from 1.586 seconds to 0.051 seconds. Both paths use
block-aware `all` mode. Multi-request rows also pass exactly: pure RWKV-7 at
batch four and prompt 512 reuses 496 tokens per request, while the hybrid model
at batch two reuses 2048 tokens per request.

Runtime LoRA rank-8 adapters targeting
`model.layers.0.attn.r_proj` pass load, two-adapter switching, removal, and
exact base-output restoration for both pure and hybrid models. The model table
advertises LoRA only after these engine rows passed.

Quantization claims remain deliberately narrower. Full online INT8 and the W4
rows reduce model footprint but fail either throughput or token gates, so they
remain memory/compatibility lanes. On RTX 4080 with the real 1.5B checkpoint,
online INT8 restricted to `lm_head` passes the batch-one, prompt-128,
decode-64 gate: model memory falls from 2.85 GiB to 2.72 GiB, throughput rises
from 49.67 to 50.78 output tokens/s, and all tokens agree. The same setting at
batch four reaches only 0.9535x fp16 throughput and 0.8008 token agreement, so
it is recorded as a failed gate and is not a general default.

All exact rows, including negative W8/W4 results, are in
`benchmarks/results/rwkv7_20260731.jsonl`. The dual V100 functional,
tensor-parallel, and pipeline-parallel rows remain valid. A fresh V100
prefix/LoRA rerun was not forced while both shared cards were occupied; no
process was terminated and no new V100 capability claim is made from the RTX
4080 evidence.

### ROCm qualification

The same native implementation was qualified on a 48 GiB `gfx1100` device
with ROCm 7.2.1 and Triton 3.5.1. The current ROCm extension was built for
`gfx1100`; its wave32 skinny-GEMM path and the RWKV-7 Triton recurrent kernels
then passed the full focused matrix: 40 tests passed and two external-checkpoint
tests skipped. This includes the packed-scan oracle, 512-token state continuity,
custom-op parity, batched decode, and the ROCm skinny-GEMM tests. ROCm-hosted
CPU model tests now bypass the GPU-only custom GEMM and use the normal Torch
linear fallback.

The real 1.5B checkpoint passed an exact batch-eight, prompt-128, decode-64
Torch-versus-Triton engine comparison across three measured repeats. Triton
produced the same continuation for all eight requests and improved median
output throughput from 125.03 to 201.05 tokens/s, a 1.608x speedup. With Mamba
prefix-cache mode `all`, both backends reused 112 tokens per request and
preserved all eight continuations; the Triton row reached 105.01 tokens/s
versus 75.32 tokens/s for Torch. A separate batch-eight asynchronous-scheduler
smoke also preserved all eight Torch continuations while chunked prefill was
enabled.

Runtime LoRA rank-8 load, two-adapter switching, removal, and exact base-output
restoration also passed with the Triton recurrent backend. The two adapters
kept the same short greedy token sequence but produced distinct generated-token
log probabilities, which avoids treating an unchanged argmax as a false LoRA
failure.

The initial generic quantization result remained a compatibility result. Full
online INT8 reduced model memory from 2.85 to 1.78 GiB but reached only 0.856x
the 16-bit throughput. BitsAndBytes NF4 reduced model memory to 1.50 GiB but
reached only 0.713x. TorchAO INT4 is unsupported on this RDNA3 device because
its packing operator requires CDNA2 or newer.

A follow-up added native compressed-tensors W8 and qualified a deterministic
W4 policy. At batch eight, prompt 128, and decode 64, the selected W8 policy
reached 1.120x FP16 throughput with 24.21% lower model memory; the selected W4
policy reached 1.064x with 17.54% lower model memory. Both pass fixed-prompt
log-probability and perplexity gates. These are exact `gfx1100`/1.5B policy
claims rather than a global AMD default. Conversion policy, commands, rejected
experiments, and quality evidence are documented in
`2026-08-03-rwkv7-rocm-quantization.md`.

The benchmark entrypoints now record the HIP runtime and accept an explicit
RWKV-7 recurrent backend. This allows ROCm quantization and LoRA runs to test
the fused recurrent path without changing the conservative global `auto`
policy. The machine-readable ROCm rows are stored in
`benchmarks/results/rwkv7_amd_20260803.jsonl`.

The extension and runtime were subsequently aligned with the current source.
The non-eager Torch-compile/CUDAGraph row at batch eight, prompt 128, and decode
8 preserved all eight continuations and improved Triton throughput from 32.14
to 148.46 output tokens/s (4.619x). The qualified compressed-tensors W8/W4 rows
also use non-eager execution. Earlier eager evidence remains valid and is not
substituted for these compile-path runs.
