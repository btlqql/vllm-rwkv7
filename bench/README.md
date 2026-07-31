# Benchmarks

Benchmark artifacts are evidence, not automatic backend promotions. Every
result states whether it covers an isolated operator, a complete layer, or an
end-to-end vLLM request.

Run the current FLA correctness and microbenchmark probe after installing the
project with its optional FLA dependency:

```bash
PYTHONPATH=src python bench/benchmark_fla.py \
  --output bench/results/blackwell_5070_fla_20260730.json
```

The committed RTX 5070 Laptop result covers fp16 FLA recurrence at
`B=1,H=16,N=64,T=1/64/512` plus a tiny complete-layer comparison. Its timing
compares FLA with the deliberately simple PyTorch oracle. It is not a vLLM
throughput comparison and cannot by itself change the Blackwell default.

An exact-card production row must additionally record a converted checkpoint,
vLLM revision, prompt/decode matrix, greedy-token agreement, cache handoff,
dynamic batching, peak memory, prefill throughput, and decode throughput.
