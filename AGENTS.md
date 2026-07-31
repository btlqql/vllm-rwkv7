# Repository Contract

This repository contains a clean-room, out-of-tree RWKV-7 integration for
vLLM. It is not a fork of vLLM.

## Scope

- Follow the public Hugging Face RWKV-7 checkpoint/configuration contract.
- Use the documented vLLM plugin interfaces from the pinned upstream revision.
- Keep a pure PyTorch correctness path before adding fused CUDA/Triton paths.
- Treat GPU policy as card-specific; do not reuse a V100 kernel policy on Ada,
  Hopper, Blackwell, or AMD without exact-card evidence.

## Provenance

- The sole implementation maintainer, Git author, and Git committer for this
  repository is `btlqql <2977859784@qq.com>`.
- Do not import, cherry-pick, vendor, or reattribute third-party RWKV-7 adapter
  or vLLM pull-request implementations.
- Public specifications and upstream APIs may be consulted. Record them in
  design documents and implement the code independently.
- Do not add `Co-authored-by` trailers for another identity.

## Verification

- Add CPU unit tests for configuration, recurrence math, registry behavior,
  and weight-name compatibility.
- Add GPU tests separately for V100 and each newer architecture.
- A fused path is opt-in until exact-card correctness and end-to-end speed rows
  show that it is safe to enable by default.
