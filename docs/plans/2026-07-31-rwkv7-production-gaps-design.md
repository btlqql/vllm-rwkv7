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
Mamba block manager by declaring `SupportsMambaPrefixCaching`; the scheduler
stores and restores the three RWKV state tensors at aligned block boundaries.
No scheduler or cache-engine fork is introduced. Pure RWKV uses Mamba cache
mode `all`; hybrid attention/RWKV uses the framework's hybrid alignment rules.

Runtime LoRA uses `SupportsLoRA` on both pure and hybrid causal-LM classes. The
model exposes its un-packed linear topology and embedding/output module names
to the existing LoRA manager. Checkpoint-native modules such as `w_lora` remain
ordinary base-model submodules. User adapters target the actual leaf linears,
for example attention projections and FFN key/value projections, using their
normal PEFT names.

Quantization stays behind vLLM's quantization configuration. Profiling first
separates recurrent-scan time from linear time and identifies which matrix
families are large enough to amortize dequantization. A selective speed policy
quantizes only profitable modules; the memory policy may quantize more modules
but is reported separately. If the available generic kernels cannot beat the
16-bit row, the design permits an RWKV-shaped fused weight-only GEMV kernel,
but only after a correctness oracle and representative shape sweep exist.

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
