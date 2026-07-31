# RWKV-7 vLLM Contract Completion Design

## Scope and evidence boundary

This patchset completes everything that can be validated without a Linux vLLM
runtime or a specific GPU. The implementation is independent: it uses the
public RWKV-7 recurrence, the public Hugging Face configuration and checkpoint
key contract, and formal APIs from vLLM commit
`837eae64580c885101ee95b073aafb27a485e7ce`. It does not use another RWKV-7
vLLM implementation as a source.

The fixed first-party interface sources are vLLM's
[`MambaBase`](https://github.com/vllm-project/vllm/blob/837eae64580c885101ee95b073aafb27a485e7ce/vllm/model_executor/layers/mamba/abstract.py),
[`LinearAttentionMetadata`](https://github.com/vllm-project/vllm/blob/837eae64580c885101ee95b073aafb27a485e7ce/vllm/v1/attention/backends/linear_attn.py),
and [plugin contract](https://github.com/vllm-project/vllm/blob/837eae64580c885101ee95b073aafb27a485e7ce/docs/design/plugin_system.md),
plus the public checkpoint's pinned
[`config.json`](https://huggingface.co/fla-hub/rwkv7-1.5B-world/blob/004140baad7a62d49a26d97508ef19cf09672328/config.json).

The adapter remains deliberately narrow: TP=1, PP=1, eager execution, and no
speculative decoding. Unsupported combinations fail during construction.

## Architecture

Keep numerical code independent of vLLM. Configuration normalization, weight
name facts, recurrence, and packed-request cache planning remain importable and
testable with PyTorch alone. The vLLM model layer is a thin adapter that reads
`LinearAttentionMetadata`, binds the two recurrent states through `MambaBase`,
and invokes the independent sequence implementation.

Represent every packed request as a validated span containing token bounds,
state slot, total sequence length, and whether a cached prefix exists. Reordered
batches select state solely by slot. A newly admitted request always starts
from zeros even when its scheduler slot previously belonged to a released
request. vLLM owns allocation and release; the model owns safe selection and
overwrite of the selected slot.

The public checkpoint already uses the model's exact parameter paths. Weight
loading therefore uses an explicit identity `WeightsMapper` instead of
inventing aliases. Tests lock the top-level and per-layer key contract.

## Acceptance matrix

| Area | Implemented before this patch | Local completion target | External gate |
| --- | --- | --- | --- |
| Plugin discovery | entry point and re-entrant registration | fixed-API contract tests | installed Linux vLLM discovery |
| Configuration | canonical/legacy heads and dimensions | `norm_eps`, strict integers, norm/activation guards | real checkpoint config |
| Weight loading | matching module paths | explicit identity mapper and key tests | vLLM loader with safetensors |
| Reference math | fp32 recurrence and sequence scan | retain token/chunk equality and validation | checkpoint logits/state |
| Stateful cache | two state tensors and slot updates | validated spans, reorder, reuse clearing, chunk continuation | scheduler allocation/release soak |
| Prefix caching | temporal copy functions | formal capability marker | vLLM prefix-cache run |
| Dynamic batching | basic packed batch test | reorder and slot-reuse tests | scheduler-driven batches |
| CI provenance | complete-history audit | check out PR head rather than synthetic merge | GitHub Actions |
| GPU backends | optional FLA and Blackwell evidence | runnable environment-gated entry | V100 and each target GPU |

## Failure handling and verification

Malformed packed metadata raises `ValueError` before cache mutation. Unsupported
execution modes raise `NotImplementedError`. Unknown checkpoint keys remain the
responsibility of vLLM's strict loader. Each local increment starts with a
failing test, then runs its focused test set. Final gates are Ruff lint and
format, the complete pytest suite, wheel build, provenance audit, identity
audit, and a clean staged diff.
