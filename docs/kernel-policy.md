# Kernel policy

Backend selection is card-specific. The environment variable
`RWKV7_VLLM_BACKEND` may request `reference`, `fla`, or `triton`, but a backend
must be allowed for the detected family.

FLA's prefill crossover and chunk shape are deliberately not promoted as one
global GPU default. Use `RWKV7_FLA_PREFILL_MIN_TOKENS` and
`RWKV7_FLA_CHUNK_SIZE` for an exact-card sweep. Until evidence is recorded, the
adapter uses a 64-token crossover and lets FLA choose its internal default
chunk size.

| Family | Architectures | Allowed during P0 | Default |
| --- | --- | --- | --- |
| Pascal and older | `sm_60`, `sm_61`, older | reference | reference |
| Volta | `sm_70` / V100 | reference, FLA | reference |
| Turing | `sm_75` | reference, FLA | reference |
| Ampere | `sm_80`, `sm_86` | reference, FLA, Triton | reference |
| Ada | `sm_89` / RTX 4090 | reference, FLA, Triton | reference |
| Hopper | `sm_90` | reference, FLA, Triton | reference |
| Blackwell | `sm_120` | reference, FLA, Triton | reference |
| AMD/CPU | ROCm or CPU | reference | reference |

"Allowed" means the backend may be selected explicitly for validation. It does
not mean it is production-validated. Promotion requires exact-card logits,
state, greedy-token, prefill, decode, and peak-memory evidence.

## Exact-card evidence

RTX 5070 Laptop (`sm_120`) is the first Blackwell development row. On
2026-07-30, raw FLA recurrence and a complete tiny RWKV-7 layer passed fp16
reference comparisons for one-token decode and 64-token chunk prefill. The
machine-readable
[result](../bench/results/blackwell_5070_fla_20260730.json) records the exact
driver, CUDA, PyTorch, Triton, FLA, error, cosine, and operator timing values.

This is correctness and operator-level evidence only. It does not include a
converted checkpoint, vLLM scheduling, end-to-end throughput, greedy-token
agreement, or peak model memory, so Blackwell continues to default to the
reference backend.
