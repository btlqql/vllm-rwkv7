# Clean-room RWKV-7 vLLM Plugin Design

## Decision

Build `vllm-rwkv7` as an independent Python distribution discovered through
the documented `vllm.general_plugins` entry-point group. The package registers
the canonical Hugging Face architecture name `RWKV7ForCausalLM` lazily with
vLLM's `ModelRegistry`. This keeps the GitHub repository non-forked, avoids
copying a vLLM source tree, and makes provenance easy to audit. The interface
baseline is vLLM `main` at `837eae64580c885101ee95b073aafb27a485e7ce`.

The model accepts the canonical Transformers fields `hidden_size`,
`num_hidden_layers`, `num_attention_heads`, `head_dim`, `intermediate_size`,
and `layer_norm_epsilon`. It accepts `num_heads` only as a compatibility alias
and normalizes it immediately. RWKV state is represented per block by a shift
state `[2, hidden_size]` and an fp32 matrix state
`[num_attention_heads, head_dim, head_dim]`. The first row stores time-mix
history and the second row channel-mix history. The per-token `v_first` value
flows between blocks inside a forward pass and is not persistent cache state.

## Execution and acceleration

The first executable backend is a correctness-first PyTorch recurrence. A
stateful block implements vLLM's documented Mamba-like cache interface and
uses linear-attention scheduling metadata to map packed prefill/decode tokens
to cache slots. P0 supports tensor parallel size one, eager execution, dynamic
batches, and chunked prefill. Unsupported modes fail clearly instead of
silently returning wrong logits.

The P0 model therefore requires `enforce_eager=True`, TP=1, PP=1, and no
speculative configuration at construction time. These guards are part of the
compatibility contract rather than performance policy.

Acceleration is a separate dispatch layer. `reference` remains the oracle;
`fla` and native Triton/CUDA backends are optional. Selection is made by exact
GPU family and capability, with environment overrides. V100 (`sm_70`) gets its
own conservative policy; Ampere, Ada, Hopper, Blackwell, and AMD do not inherit
its tile or launch choices. No fused backend becomes default until logits,
state, greedy-token, memory, prefill, and decode gates pass on the exact card.

## Correctness and release gates

CPU tests cover Hugging Face configuration aliases and validation, the RWKV-7
state equation, chunked-versus-token recurrence equality, plugin idempotency,
and package metadata. Linux integration tests install the package beside the
pinned vLLM revision and verify registry inspection with a tiny checkpoint.
GPU jobs then test prefill/decode state handoff and dynamic batching. Releases
record the vLLM revision they target and never claim FLA/native speedups from a
microbenchmark alone.
