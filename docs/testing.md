# Testing

## CPU correctness

Run the dependency-light suite on every change:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python scripts/check_provenance.py
python -m pip wheel --no-deps . -w dist
```

The suite covers strict HF configuration normalization, fp32 recurrence state,
chunked state handoff, distinct attention/hidden sizes, checkpoint-compatible
parameter names, plugin idempotency, packed cache-slot routing with a vLLM API
stub, request reordering, released-slot reuse, and repository author/committer
identity.

## Current vLLM integration

The initial interface baseline is official vLLM `main` commit
`837eae64580c885101ee95b073aafb27a485e7ce`. On Linux, install that revision
and this repository, then run the integration tests:

```bash
python -m pip install "git+https://github.com/vllm-project/vllm.git@837eae64580c885101ee95b073aafb27a485e7ce"
python -m pip install -e ".[test]"
python -m pytest tests/integration -q
```

The contract and registry tests require only the pinned vLLM installation. A
real eager engine load additionally requires a checkpoint and GPU:

```bash
RWKV7_VLLM_TEST_MODEL=fla-hub/rwkv7-1.5B-world \
  python -m pytest tests/integration/test_vllm_engine.py -q
```

The engine test is deliberately skipped unless `RWKV7_VLLM_TEST_MODEL` is
set. It performs a two-token greedy generation with TP=1 and eager execution;
a skip is not a successful real-vLLM or GPU validation.

Use `enforce_eager=True` for model loading in P0. TP, PP, compiled execution,
and speculative decoding beyond the documented single-device path are guarded
until their state-cache behavior has dedicated tests.

Windows unit tests use a small API stub because vLLM is a Linux runtime. The
stub is not a substitute for the Linux integration gate.

## Acceptance matrix

| Contract | Dependency-light verification | External verification |
| --- | --- | --- |
| Plugin and architecture registration | entry point metadata and re-entrant stub test | pinned Linux vLLM registry test |
| HF configuration and checkpoint names | strict config and exact `state_dict` key tests | public checkpoint load |
| Recurrent state and chunked prefill | PyTorch oracle, handoff, and packed-span tests | checkpoint logits and final state |
| Dynamic batching | request reorder and released-slot reuse tests | scheduler allocation/release soak |
| Prefix caching | marker and temporal-copy contract tests | vLLM prefix-cache run |
| GPU execution | environment-gated test and benchmark entry points | each exact GPU model |

vLLM owns cache allocation and release. The model selects state by scheduler
slot, clears a reused slot when the query is the whole sequence, and overwrites
the selected slot after each chunk. The local tests prove those transitions;
only a real scheduler run can validate allocation and release integration.

## GPU matrix

Run correctness before performance on every exact card:

1. Load a tiny converted HF checkpoint.
2. Compare prompt logits and final recurrent state with the PyTorch reference.
3. Compare one-shot prefill with chunked prefill plus decode.
4. Exercise cache-slot selection and request reordering at batch sizes 1, 2,
   4, and 8.
5. Record peak memory, prefill throughput, decode throughput, and greedy-token
   agreement.

V100 (`sm_70`) is a separate row. FLA or Triton results from Ampere, Ada,
Hopper, or Blackwell must not be used to select V100 defaults.

## Recorded Blackwell operator row

The first exact-card result is
[`blackwell_5070_fla_20260730.json`](../bench/results/blackwell_5070_fla_20260730.json):

- NVIDIA GeForce RTX 5070 Laptop GPU, `sm_120`, 8151 MiB;
- driver 582.05, CUDA 12.8, PyTorch 2.11.0+cu128, Triton 3.7.1, FLA 0.5.1;
- fp16 inputs with fp32 recurrent state;
- `B=1,H=16,N=64,T=1/64/512` raw recurrence comparisons;
- complete tiny-layer comparisons at `T=1/64`;
- every row passes output cosine `>=0.9999` and state max-absolute error
  `<=0.02`.

For `T=512`, the recorded FLA chunk row is about 0.532 ms with output cosine
0.99999982. The 281x ratio in that row is against this repository's sequential
PyTorch recurrence oracle. It is an operator microbenchmark, not a complete
model or vLLM performance claim; the Blackwell default remains `reference`.
