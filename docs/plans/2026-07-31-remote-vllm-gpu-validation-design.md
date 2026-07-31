# Remote vLLM GPU Validation Design

## Scope and privacy boundary

This stage validates the clean-room adapter in real Linux vLLM runtimes on two
authorized SSH targets. `gpu4080` is the primary Ada checkpoint and scheduler
host; `WZU_Server` is the secondary V100/SM70 compatibility host. Reports may
name only those configured aliases. They must not contain remote usernames,
IP addresses, private key paths, credentials, or expanded home directories.

Every virtual environment, official vLLM checkout, repository checkout, Hugging
Face cache, and model weight lives in a dedicated repository-external directory
represented as `${HOME}`. No checkpoint or generated environment artifact is
committed. Git author, committer, and GitHub identity remain exclusively
`btlqql <2977859784@qq.com>`.

## Fixed upstream installation

The Python API baseline remains official vLLM commit
`837eae64580c885101ee95b073aafb27a485e7ce`. Its per-commit wheel index contains
only AArch64 artifacts, so x86_64 hosts use vLLM's official Python-only build
mechanism. The fixed source is overlaid on the pinned official precompiled
x86_64 wheel from `553fcb82d5602c75fb6ab41b6dc3c46f480c1785`. Both revisions
have identical CUDA runtime and build requirement blobs. Evidence records both
SHAs and never describes the binary revision as the fixed Python revision.

## Layered validation

The primary host runs compiled-extension import, CUDA discovery, plugin entry
point discovery, architecture/config registration, checkpoint-key loading,
real recurrent-cache binding, one-shot/chunked equality, packed request
reordering, released-slot reuse, prefix-cache copy contracts, and short eager
generation. An existing local 0.1B Hugging Face checkpoint is transferred
without model Python files and runs with `trust_remote_code=False`, TP=1, short
context, small batches, and conservative memory utilization. Its actual SHA,
size, config contract, and non-revision provenance are recorded rather than
equating it to the previously considered fixed public revision.

The V100 host repeats dependency-light contracts and all safe CUDA/cache/model
tests, then attempts the same constrained engine gate only if memory and kernel
compatibility permit. Failures are recorded by exact stage and never converted
to passes. The evidence report contains sanitized JSON with aliases, OS, Python,
torch, CUDA, vLLM source/binary revisions, GPU model/capability/memory, commands,
outcomes, and explicit skipped gates.

The observed engine path uses vLLM's `align` Mamba-cache fallback. The plugin
does not claim the stronger `all`-mode marker because the pure PyTorch
recurrence does not materialize state at every cache-block boundary.
