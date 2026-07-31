# RWKV-7 Upstream-Based Downstream Design

## Purpose

`btlqql/vllm-rwkv7` is a downstream vLLM repository, not a GitHub fork and not
an out-of-tree reimplementation of vLLM integration surfaces. Its `main`
branch mirrors the official `vllm-project/vllm` `main` history. RWKV-7 work is
developed on feature branches and periodically synchronized with the official
upstream remote.

The previous standalone plugin remains available only as an archived recovery
point. It is not the implementation base for future work.

## Source Baseline

The initial synchronized baseline is official vLLM commit
`0f17394564fa2fccd332cf63321314884c15ee37`. Existing native RWKV-7 work is
available in upstream pull request #50077. That pull request is the explicit
continuation of #48686 and already provides the model, recurrent state-cache
integration, fused execution boundary, quantization paths, tests, and
benchmarks required for the next phase.

The downstream branch references #50077 as implementation provenance, but its
own `base..HEAD` history is intentionally squashed into commits authored and
committed only by `btlqql <2977859784@qq.com>`. It does not create a competing
pull request against `vllm-project/vllm` while the upstream work remains open.

## Repository Topology

- `origin`: `https://github.com/btlqql/vllm-rwkv7.git`
- `upstream`: `https://github.com/vllm-project/vllm.git`
- `origin/main`: exact synchronized official baseline
- `wangyue/cleanroom-prototype-archive`: immutable recovery point for the old
  standalone plugin at `ca0b7ad034e0b35fb1aa9a23c60bfcbda7c92b51`
- `wangyue/rwkv7-upstream-adapter`: downstream integration and validation
  branch based on official vLLM

Future upstream synchronization first fast-forwards `origin/main` to
`upstream/main`, then rebuilds the RWKV-7 patch directly on that base. This
keeps `origin/main..HEAD` free of third-party feature-branch commits.

## Integration Strategy

Use the #50077 implementation as the source reference, reconcile its resulting
tree with the current official baseline, and publish the downstream delta as a
btlqql-only patch. The initial integration had three content conflicts: the
compilation splitting-op registry, native model registry, and Transformers
config registry. Each resolution retained current upstream entries and added
the RWKV-7 entry in the appropriate collection.

No scheduler, cache engine, distributed runtime, compilation framework, or
quantization framework is reimplemented. RWKV-7 uses vLLM's existing Mamba
inner-state interfaces, model registry, custom-op boundary, tensor-parallel
linear layers, pipeline-parallel helpers, and quantization mechanisms.

## Validation Boundary

The first downstream patchset must prove that synchronization did not regress
the existing RWKV-7 implementation. Required checks are import/config and model
registry tests, dummy initialization, the RWKV-7 unit/CUDA suite, online INT8
and TorchAO INT4 focused tests, pre-commit on changed files, and at least one
real vLLM engine smoke using an external checkpoint. RTX 4080 and V100 remain
separate evidence rows; fused or quantized defaults are not generalized across
GPU families without exact-card results.

The downstream pull request targets `btlqql/vllm-rwkv7`, references the
relevant upstream work in prose, contains only btlqql-authored/committed branch
commits, lists exact test results, and discloses AI assistance. It is not
submitted to the official vLLM repository as duplicate work.
