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

The suite covers HF configuration normalization, fp32 recurrence state,
chunked state handoff, distinct attention/hidden sizes, checkpoint-compatible
parameter names, plugin idempotency, packed cache-slot routing with a vLLM API
stub, new-request cache-slot clearing, and repository author/committer identity.

## Current vLLM integration

The initial interface baseline is official vLLM `main` commit
`837eae64580c885101ee95b073aafb27a485e7ce`. On Linux, install that revision
and this repository, then run the integration tests:

```bash
python -m pip install "git+https://github.com/vllm-project/vllm.git@837eae64580c885101ee95b073aafb27a485e7ce"
python -m pip install -e ".[test]"
python -m pytest tests/integration -q
```

Use `enforce_eager=True` for model loading in P0. TP, PP, compiled execution,
and speculative decoding beyond the documented single-device path are guarded
until their state-cache behavior has dedicated tests.

Windows unit tests use a small API stub because vLLM is a Linux runtime. The
stub is not a substitute for the Linux integration gate.

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
