# RWKV-7 ROCm W8/W4 Qualification

## Scope

This document qualifies two compressed-tensors weight-only policies for the
RWKV-7 1.5B checkpoint on one AMD RDNA3 (`gfx1100`) device. The result is an
exact-card result, not a default for CDNA, other RDNA generations, or other
model sizes.

The implementation keeps vLLM's mixed-precision linear dispatch intact:

- W8 uses a native Triton W8A16 kernel for symmetric, channelwise packed
  weights on `gfx11` and `gfx12` devices.
- W4 uses the existing RDNA kernels. Repeatability-sensitive runs disable the
  unordered FP16-atomic RDNA3 kernel through vLLM's standard
  `VLLM_DISABLED_KERNELS` mechanism and select the deterministic hybrid
  implementation instead.
- Unsupported devices, dtypes, group layouts, and shapes fall through to the
  normal mixed-precision kernel selection. No GPU-name checks are added to the
  RWKV model.

## Conversion policies

The converter streams safetensors shards, writes compressed-tensors
`pack-quantized` checkpoints, and records the selected policy in the output
manifest. The first and final recurrent layers remain in FP16 in both
qualified policies.

### W8

W8 applies symmetric channelwise quantization to the FFN key and value
projections. Attention projections and layers 0 and 23 remain FP16.

```bash
python scripts/quantize_rwkv7_compressed_tensors.py \
  --model /workspace/models/rwkv7-g1h-1.5b-vllm \
  --output /workspace/models/rwkv7-g1h-1.5b-w8a16-ffn-edgefp16-ct \
  --bits 8 \
  --no-quantize-attention-projections \
  --ffn-projections key value \
  --exclude-layer 0 \
  --exclude-layer 23
```

### W4

W4 applies symmetric group-32 quantization only to FFN key projections.
FFN value and attention projections, plus layers 0 and 23, remain FP16. This
policy was selected because quantizing additional projections reduced fixed
prompt likelihood quality beyond the qualification limit.

```bash
python scripts/quantize_rwkv7_compressed_tensors.py \
  --model /workspace/models/rwkv7-g1h-1.5b-vllm \
  --output /workspace/models/rwkv7-g1h-1.5b-w4a16-g32-key-edgefp16-ct \
  --bits 4 \
  --group-size 32 \
  --no-quantize-attention-projections \
  --ffn-projections key \
  --exclude-layer 0 \
  --exclude-layer 23
```

## Qualification method

The gate uses eight fixed, tokenizer-independent prompts of 128 tokens from
`benchmarks/data/rwkv7_quantization_prompts_v1.tokens`. Each setting runs in a
fresh engine process with batch 8, 64 generated tokens per request, chunked
prefill, Torch compilation/CUDAGraph enabled, three warmups, and three measured
repeats.

Generated-token equality is diagnostic only: one early quantized argmax change
causes the rest of an autoregressive continuation to follow a different path.
The quality gate therefore compares log probabilities on the same fixed
natural prompt tokens and reports both mean absolute log-probability error and
the prompt perplexity ratio.

W8 command:

```bash
python benchmarks/benchmark_rwkv7_quantization.py \
  --model /workspace/models/rwkv7-g1h-1.5b-vllm \
  --ct-w8-model /workspace/models/rwkv7-g1h-1.5b-w8a16-ffn-edgefp16-ct \
  --settings fp16 ct-w8 \
  --prompt-token-ids-file benchmarks/data/rwkv7_quantization_prompts_v1.tokens \
  --max-tokens 64 \
  --max-model-len 256 \
  --max-num-batched-tokens 2048 \
  --no-enforce-eager \
  --rwkv7-backend triton \
  --warmup-runs 3 \
  --repeats 3 \
  --min-speed-ratio 1.0 \
  --min-memory-reduction 0.01 \
  --min-token-agreement 0.0 \
  --max-prompt-logprob-mean-abs-error 0.05 \
  --max-prompt-perplexity-ratio 1.01 \
  --repeat-logprob-margin-tolerance 0.02
```

The W4 command substitutes `--ct-w4-model`, `--settings fp16 ct-w4`, and uses
`0.20` and `1.05` for the prompt-error and perplexity limits. It uses the same
explicit near-tie tolerance, although the final three-repeat W4 run did not
need it.

## Results

Environment: 48 GiB `gfx1100`, ROCm 7.2.1, PyTorch 2.11 development build,
Triton 3.6, and vLLM 0.26.1 development build.

| Policy | FP16 tok/s | Quant tok/s | Speed | Model GiB | Reduction | Prompt logprob MAE | Prompt PPL ratio | Repeatability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| W8 channel, FFN key/value | 321.11 | 359.73 | 1.120x | 2.16 | 24.21% | 0.03073 | 1.00046 | Passed; one explicit near-tie |
| W4 group-32, FFN key | 320.05 | 340.59 | 1.064x | 2.35 | 17.54% | 0.18453 | 1.03462 | Exact in three repeats |

The W8 near-tie occurred at request 1, generated token 43. The two competing
tokens differed by 0.00085 and 0.00903 log-probability units in the two runs,
both below the explicit 0.02 tolerance. The record retains this diagnostic and
does not label the row exact-repeatable.

Machine-readable records are tagged
`gfx1100-b8-p128-d64-natural-w8-v3` and
`gfx1100-b8-p128-d64-natural-w4-g32-key-v5` in
`benchmarks/results/rwkv7_amd_20260803.jsonl`.

## Negative results and claim boundaries

- Full online INT8 and BitsAndBytes NF4 reduced memory but did not beat FP16 on
  this device.
- W4 group-128 over both FFN projections reached about 1.20x speed and reduced
  model memory by about 35.8%, but its prompt logprob error was about 0.919 and
  its prompt perplexity ratio was about 1.706, so it was rejected.
- W4 group-32 over both FFN projections did not recover enough quality.
- The fast RDNA3 W4 kernel uses unordered FP16 atomic accumulation. It remains
  available for throughput-oriented use, while deterministic algorithms or the
  repeatability benchmark select the hybrid fallback.
- These gates establish W8/W4 production policies only for the stated model,
  batch, sequence lengths, software stack, and `gfx1100` device.

## Tests

The focused remote suite covers converter policy and sharding, W8 kernel
selection and numerical parity, W4 deterministic selection and fallback,
RDNA3 compile guards, the hybrid W4 kernel, prompt-corpus validation, quality
metrics, and repeatability diagnostics. On the qualification host it completed
with `174 passed, 5 skipped`.
