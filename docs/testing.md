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

The interface baseline is official vLLM `main` commit
`837eae64580c885101ee95b073aafb27a485e7ce`. The recorded x86_64 environments
use vLLM's official Python-only source installation with precompiled binary
commit `553fcb82d5602c75fb6ab41b6dc3c46f480c1785`. The source and binary commits
are deliberately recorded separately. Install the fixed source in a
repository-external environment, install this project editable, and run:

```bash
python -m pip install -e ".[test]"
python -m pytest tests/integration -q
```

The contract and registry tests require only the pinned vLLM installation. A
real eager engine load additionally requires a checkpoint and GPU:

```bash
RWKV7_VLLM_TEST_MODEL=${HOME}/models/rwkv7-g1d-0.1b-hf \
  python -m pytest tests/integration/test_vllm_engine.py -q
```

The engine test is deliberately skipped unless `RWKV7_VLLM_TEST_MODEL` is
set. It performs TP=1 eager generation with two packed requests, a 96-token
shared prefix split by a 32-token chunked-prefill budget, a real prefix-cache
hit, cache reset, and cached-versus-cold token equality. A skip is not a
successful real-vLLM or GPU validation. On a shared card, set a smaller
test-only reservation such as
`RWKV7_VLLM_GPU_MEMORY_UTILIZATION=0.05`; the value must be in `(0, 1]`.

Use `enforce_eager=True` for model loading in P0. TP, PP, compiled execution,
and speculative decoding beyond the documented single-device path are guarded
until their state-cache behavior has dedicated tests.

Windows unit tests use a small API stub because vLLM is a Linux runtime. The
stub is not a substitute for the Linux integration gate.

## Acceptance matrix

| Contract | Dependency-light verification | RTX 4080 | Tesla V100 | Remaining |
| --- | --- | --- | --- | --- |
| Plugin, HF config, and model registration | entry point, re-entrant ownership, and no-remote-code config tests | passed against fixed vLLM | passed against fixed vLLM | none for P0 |
| Weight names and loading | exact `state_dict` names and identity mapper tests | existing local 0.1B checkpoint loaded | same checkpoint loaded | 0.4B/1.5B engine expansion |
| Recurrent state and chunked prefill | PyTorch oracle, handoff, and packed-span tests | real CUDA cache binding and engine passed | real CUDA cache binding and engine passed | optional long-context expansion |
| Dynamic batching and release | reorder and released-slot reuse tests | real two-request engine plus cache lifecycle passed | real two-request engine plus cache lifecycle passed | long scheduler soak |
| Prefix caching | temporal-copy contracts; no unsupported `all` marker | `align` hit/reset/cold equality passed | `align` hit/reset/cold equality passed | longer eviction soak |
| Reference GPU execution | environment-gated correctness tests | passed, `sm_89` | passed, `sm_70` | compiled and parallel modes |
| Optional FLA backend | runnable correctness gates | not installed; no claim | not installed; no claim | exact-card FLA validation |

vLLM owns cache allocation and release. The model selects state by scheduler
slot, clears a reused slot when the query is the whole sequence, and overwrites
the selected slot after each chunk. Tests now exercise those transitions with
actual vLLM cache tensors on CUDA and exercise bounded multi-request scheduling
through the public engine. A long allocation/eviction soak remains unclaimed.

The reference recurrence intentionally does not implement
`SupportsMambaPrefixCaching`. That marker selects vLLM's stronger `all` mode,
which requires state at every cache-block boundary. The correctness path uses
vLLM's official `align` fallback; real engine tests require an actual cache hit
and then prove reset/cold generation returns identical greedy tokens.

## Recorded remote vLLM rows

The sanitized
[`remote_gpu_vllm_20260731.json`](../bench/results/remote_gpu_vllm_20260731.json)
records these completed rows:

- `gpu4080`: Ubuntu 24.04.4, RTX 4080 (`sm_89`, 16376 MiB), driver
  595.71.05, Python 3.12.3, PyTorch 2.13.0+cu132, CUDA 13.2;
- `WZU_Server`: Ubuntu 22.04.5, 2x Tesla V100-PCIE-32GB (`sm_70`), driver
  580.173.02, Python 3.10.12, PyTorch 2.13.0+cu126, CUDA 12.6;
- fixed vLLM Python version
  `0.26.1rc1.dev146+g837eae645.d20260731` on both hosts;
- fixed source commit `837eae64580c885101ee95b073aafb27a485e7ce` and
  precompiled binary commit `553fcb82d5602c75fb6ab41b6dc3c46f480c1785`;
- final shared-GPU suite reservations of 0.15 on RTX 4080 and 0.05 on V100;
  the RTX 4080 engine also passed 0.75 earlier while the card was idle;
- full result on each host: 76 passed, 4 optional-FLA skips, 0 failed.

The engine checkpoint came from an existing local Hugging Face cache, not the
previously considered fixed public revision. Its `model.safetensors` is
382110672 bytes with SHA-256
`12d208adf2880927615656c2dc3f6fb6a3ea3120a9ed9fdfeeea55c841723d79`.
Its config declares `RWKV7ForCausalLM` and `rwkv7_hf_adapter`. Although the
config contains `auto_map`, the validation transferred no model Python files,
registered the local config class, and kept `trust_remote_code=False`.

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
