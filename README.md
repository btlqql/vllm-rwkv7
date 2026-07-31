# vllm-rwkv7

`vllm-rwkv7` is a clean-room, out-of-tree RWKV-7 model plugin for vLLM. It is
an independent repository, not a GitHub fork of vLLM.

The project follows the Hugging Face RWKV-7 configuration and checkpoint
contract while using vLLM's documented `vllm.general_plugins` registration
interface. The initial backend is a pure PyTorch correctness oracle. FLA and
native fused kernels will remain opt-in until they pass exact-card correctness
and end-to-end performance gates.

## Status

The current implementation contains:

- canonical and legacy HF configuration normalization;
- the fp32-state RWKV-7 recurrence oracle;
- re-entrant vLLM architecture registration;
- a correctness-first stateful vLLM model for TP=1 eager execution;
- strict public HF config and checkpoint-key normalization;
- validated recurrent-cache routing for chunked prefill, request reordering,
  and released-slot reuse;
- vLLM inner-state contracts and correctness-validated `align` prefix caching;
- optional FLA recurrent-decode and chunk-prefill dispatch;
- explicit per-GPU backend policy for V100 and newer GPU families.

The fixed vLLM runtime has loaded an existing local 0.1B Hugging Face
checkpoint and passed eager generation, chunked prefill, dynamic recurrent
cache routing, and prefix-cache hit/reset/cold-equality gates on an RTX 4080
and a Tesla V100. The exact sanitized evidence is
[`remote_gpu_vllm_20260731.json`](bench/results/remote_gpu_vllm_20260731.json).
Optional FLA on those two cards, compiled execution, parallel execution, and a
long scheduler soak remain explicit follow-up gates rather than claimed work.
See [the design](docs/plans/2026-07-30-cleanroom-rwkv7-vllm-design.md) and
[the current acceptance matrix](docs/testing.md#acceptance-matrix).

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
python -m ruff check .
```

The plugin targets the vLLM `main` Python interface at commit
`837eae64580c885101ee95b073aafb27a485e7ce`. The recorded x86_64 validation
uses vLLM's official Python-only installation over the precompiled binary
baseline `553fcb82d5602c75fb6ab41b6dc3c46f480c1785`; the evidence keeps those
revisions distinct.
P0 requests must use `enforce_eager=True` (CLI: `--enforce-eager`) with TP=1,
PP=1, and speculative decoding disabled. Unsupported combinations fail during
model construction.

Run the real engine gate with a repository-external local checkpoint:

```bash
RWKV7_VLLM_TEST_MODEL=${HOME}/models/rwkv7-g1d-0.1b-hf \
  python -m pytest tests/integration/test_vllm_engine.py -q
```

On a shared GPU, `RWKV7_VLLM_GPU_MEMORY_UTILIZATION` may lower the test-only
vLLM reservation. It does not change model behavior.

To test the optional FLA path on a supported GPU:

```bash
python -m pip install -e ".[fla,test]"
RWKV7_VLLM_BACKEND=fla python -m pytest -q
```

FLA decode uses its fused multiplicative RWKV-7 recurrence. Prefill switches
to FLA's chunk path at 64 tokens unless an exact-card sweep sets
`RWKV7_FLA_PREFILL_MIN_TOKENS` and `RWKV7_FLA_CHUNK_SIZE`.

The first exact-card FLA evidence is the
[RTX 5070 Laptop Blackwell result](bench/results/blackwell_5070_fla_20260730.json).
It validates recurrence and tiny-layer correctness and reports operator timing;
it is deliberately not labeled as end-to-end vLLM acceleration.

## Provenance

The RWKV-7 equations are implemented independently from the
[public algorithm description](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py).
The optional adapter calls FLA's published `fla.ops.rwkv7` API. Previous
third-party vLLM RWKV-7 pull-request implementations are not imported into this
repository.

The active [contribution policy](CONTRIBUTING.md) keeps both Git author and
committer identity restricted to `btlqql` for the clean-room phase.
