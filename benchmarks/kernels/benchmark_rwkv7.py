# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare an RWKV7 candidate kernel policy with Torch through the vLLM engine.

Each backend runs in a fresh subprocess so environment selection, CUDA graphs,
and recurrent caches cannot leak between runs. The report identifies the first
greedy-token divergence and includes the top log-probabilities at that position.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import regex as re

DEFAULT_PROMPTS = [
    "The Eiffel Tower is located in",
    "A short proof that there are infinitely many primes begins",
    "Write a Python function that computes Fibonacci numbers:",
    "The most important property of recurrent neural networks is",
    "Once upon a time in a quiet village",
    "Explain why the sky appears blue during the day.",
    "In numerical analysis, floating point accumulation order",
    "Translate to Chinese: artificial intelligence inference engine",
]

ENGINE_MEMORY_PATTERN = re.compile(
    r"Actual usage is ([0-9.]+) GiB for consumed memory .*?"
    r"([0-9.]+) GiB for peak activation, and ([0-9.]+) GiB for CUDAGraph "
    r"memory\..*?Current kv cache memory in use is ([0-9.]+) GiB"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument(
        "--prompt-token-length",
        action="append",
        type=int,
        default=[],
        help="Use a deterministic token-ID prompt of this length; repeat for batches.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--async-scheduling", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--ignore-eos", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--require-exact", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--prefix-cache", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--require-prefix-hit", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--mamba-cache-mode", choices=("align", "all"), default="all")
    parser.add_argument(
        "--candidate-backend", choices=("auto", "triton"), default="triton"
    )
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        help="Append one machine-readable benchmark record to this JSONL file.",
    )
    parser.add_argument("--benchmark-tag", default="")
    parser.add_argument(
        "--worker",
        choices=("auto", "torch", "triton"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[Any]:
    if args.warmup_runs < 0 or args.repeats < 1:
        raise ValueError("--warmup-runs must be >= 0 and --repeats must be >= 1")
    if args.prompt_token_length:
        if args.prompt or args.prompts_file is not None:
            raise ValueError(
                "--prompt-token-length cannot be combined with text prompts"
            )
        if any(length < 1 for length in args.prompt_token_length):
            raise ValueError("--prompt-token-length values must be positive")
        return [
            {
                "prompt_token_ids": [
                    3 + ((request_index * 17 + token_index) % 124)
                    for token_index in range(length)
                ]
            }
            for request_index, length in enumerate(args.prompt_token_length)
        ]
    prompts = list(args.prompt)
    if args.prompts_file is not None:
        loaded = json.loads(args.prompts_file.read_text())
        if not isinstance(loaded, list) or not all(isinstance(x, str) for x in loaded):
            raise ValueError("--prompts-file must contain a JSON array of strings")
        prompts.extend(loaded)
    return prompts or DEFAULT_PROMPTS


def serialize_logprobs(logprobs: Any) -> list[list[dict[str, Any]]]:
    serialized = []
    for position in logprobs or []:
        candidates = [
            {
                "token_id": int(token_id),
                "logprob": float(value.logprob),
                "rank": value.rank,
            }
            for token_id, value in position.items()
        ]
        candidates.sort(key=lambda item: item["logprob"], reverse=True)
        serialized.append(candidates)
    return serialized


def run_worker(args: argparse.Namespace) -> None:
    assert args.worker is not None
    os.environ["VLLM_RWKV7_KERNEL"] = args.worker
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import torch

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.platforms import current_platform

    prompts = load_prompts(args)
    if torch.accelerator.is_available():
        torch.accelerator.reset_peak_memory_stats()
        device = torch.accelerator.current_device_index()
        capability = current_platform.get_device_capability(device)
        gpu = {
            "name": current_platform.get_device_name(device),
            "capability": None if capability is None else list(capability),
            "total_memory_bytes": current_platform.get_device_total_memory(device),
        }
    else:
        gpu = None
    llm_kwargs: dict[str, Any] = {}
    if args.prompt_token_length:
        llm_kwargs["skip_tokenizer_init"] = True
    else:
        llm_kwargs["tokenizer"] = args.tokenizer or args.model
    if args.prefix_cache:
        llm_kwargs.update(
            enable_prefix_caching=True,
            mamba_cache_mode=args.mamba_cache_mode,
        )
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=False,
        enforce_eager=args.enforce_eager,
        async_scheduling=args.async_scheduling,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=False,
        **llm_kwargs,
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
        ignore_eos=args.ignore_eos,
    )
    for _ in range(args.warmup_runs):
        llm.generate(prompts, sampling_params, use_tqdm=False)
    if args.prefix_cache and args.warmup_runs and not llm.reset_prefix_cache():
        raise RuntimeError("prefix cache could not be reset after warmup")

    prefix_cache_report = None
    cold_signature = None
    if args.prefix_cache:
        cold_started = time.perf_counter()
        cold_outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        cold_elapsed = time.perf_counter() - cold_started
        cold_signature = [tuple(item.outputs[0].token_ids) for item in cold_outputs]
    samples = []
    signatures = []
    outputs = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        samples.append(time.perf_counter() - started)
        signatures.append([tuple(item.outputs[0].token_ids) for item in outputs])
    assert outputs is not None
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError(
            f"{args.worker} output changed across repeated runs: {signatures}"
        )
    if cold_signature is not None:
        hit_cached_tokens = [int(item.num_cached_tokens or 0) for item in outputs]
        prefix_cache_report = {
            "mode": args.mamba_cache_mode,
            "cold_elapsed_s": cold_elapsed,
            "hit_elapsed_s": statistics.median(samples),
            "speedup": cold_elapsed / statistics.median(samples),
            "num_cached_tokens": hit_cached_tokens,
            "cold_hit_exact": cold_signature == signatures[0],
        }
        if not prefix_cache_report["cold_hit_exact"]:
            raise RuntimeError(
                f"{args.worker} prefix-cache output changed: "
                f"cold={cold_signature}, hit={signatures[0]}, "
                f"cached={hit_cached_tokens}"
            )
        if args.require_prefix_hit and not all(
            count > 0 for count in hit_cached_tokens
        ):
            raise RuntimeError(
                f"{args.worker} produced no prefix-cache hit: {hit_cached_tokens}"
            )
    elapsed = statistics.median(samples)
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()
        peak_allocated_bytes = torch.accelerator.max_memory_allocated()
        peak_reserved_bytes = torch.accelerator.max_memory_reserved()
    else:
        peak_allocated_bytes = None
        peak_reserved_bytes = None
    input_tokens = sum(len(item.prompt_token_ids) for item in outputs)
    output_tokens = sum(len(item.outputs[0].token_ids) for item in outputs)
    request_metrics = []
    for item in outputs:
        metrics = item.metrics
        arrival_time = getattr(metrics, "arrival_time", None)
        first_token_time = getattr(metrics, "first_token_time", None)
        finished_time = getattr(metrics, "finished_time", None)
        num_output_tokens = len(item.outputs[0].token_ids)
        ttft_s = (
            None
            if arrival_time is None or first_token_time is None
            else first_token_time - arrival_time
        )
        decode_s = (
            None
            if first_token_time is None or finished_time is None
            else finished_time - first_token_time
        )
        request_metrics.append(
            {
                "prompt_tokens": len(item.prompt_token_ids),
                "output_tokens": num_output_tokens,
                "ttft_s": ttft_s,
                "decode_s": decode_s,
                "decode_tok_s": (
                    None
                    if decode_s is None or decode_s <= 0 or num_output_tokens <= 1
                    else (num_output_tokens - 1) / decode_s
                ),
                "num_cached_tokens": int(item.num_cached_tokens or 0),
            }
        )
    result = {
        "backend": args.worker,
        "elapsed_s": elapsed,
        "samples_s": samples,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "gpu": gpu,
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hip": getattr(torch.version, "hip", None),
            "vllm": vllm.__version__,
        },
        "request_metrics": request_metrics,
        "prefix_cache": prefix_cache_report,
        "requests": [
            {
                "token_ids": list(item.outputs[0].token_ids),
                "logprobs": serialize_logprobs(item.outputs[0].logprobs),
            }
            for item in outputs
        ],
    }
    result["output_tok_s"] = result["output_tokens"] / elapsed
    result["total_tok_s"] = result["total_tokens"] / elapsed
    print("RESULT_JSON " + json.dumps(result), flush=True)


def run_backend(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    command = [sys.executable, __file__]
    for name in (
        "model",
        "max_tokens",
        "logprobs",
        "warmup_runs",
        "repeats",
        "dtype",
        "max_model_len",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
    ):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.tokenizer is not None:
        command.extend(["--tokenizer", args.tokenizer])
    for prompt in args.prompt:
        command.extend(["--prompt", prompt])
    for prompt_token_length in args.prompt_token_length:
        command.extend(["--prompt-token-length", str(prompt_token_length)])
    if args.prompts_file is not None:
        command.extend(["--prompts-file", str(args.prompts_file)])
    command.append("--enforce-eager" if args.enforce_eager else "--no-enforce-eager")
    command.append(
        "--async-scheduling" if args.async_scheduling else "--no-async-scheduling"
    )
    command.append("--ignore-eos" if args.ignore_eos else "--no-ignore-eos")
    command.append("--prefix-cache" if args.prefix_cache else "--no-prefix-cache")
    command.append(
        "--require-prefix-hit" if args.require_prefix_hit else "--no-require-prefix-hit"
    )
    command.extend(["--mamba-cache-mode", args.mamba_cache_mode])
    command.extend(["--worker", backend])

    result = None
    engine_memory = None
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        if match := ENGINE_MEMORY_PATTERN.search(line):
            consumed, activation, cudagraph, cache = map(float, match.groups())
            engine_memory = {
                "consumed_gib": consumed,
                "peak_activation_gib": activation,
                "cudagraph_gib": cudagraph,
                "cache_gib": cache,
                "estimated_peak_gib": consumed + activation + cudagraph + cache,
            }
        if line.startswith("RESULT_JSON "):
            result = json.loads(line.removeprefix("RESULT_JSON "))
        else:
            print(f"[{backend}] {line}", end="")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{backend} worker exited with status {return_code}")
    if result is None:
        raise RuntimeError(f"{backend} worker produced no result")
    result["engine_memory"] = engine_memory
    return result


def top_margin(candidates: list[dict[str, Any]]) -> float | None:
    if len(candidates) < 2:
        return None
    return candidates[0]["logprob"] - candidates[1]["logprob"]


def compare_results(
    torch_result: dict[str, Any], candidate_result: dict[str, Any]
) -> dict[str, Any]:
    if len(torch_result["requests"]) != len(candidate_result["requests"]):
        raise RuntimeError("Reference and candidate returned different batch sizes")
    request_reports = []
    exact_requests = 0
    for request_idx, (torch_request, candidate_request) in enumerate(
        zip(torch_result["requests"], candidate_result["requests"])
    ):
        torch_ids = torch_request["token_ids"]
        candidate_ids = candidate_request["token_ids"]
        first_difference = next(
            (
                index
                for index, (torch_id, candidate_id) in enumerate(
                    zip(torch_ids, candidate_ids)
                )
                if torch_id != candidate_id
            ),
            None,
        )
        if first_difference is None and len(torch_ids) != len(candidate_ids):
            first_difference = min(len(torch_ids), len(candidate_ids))
        exact = first_difference is None
        exact_requests += int(exact)
        report: dict[str, Any] = {
            "request": request_idx,
            "exact": exact,
            "torch_length": len(torch_ids),
            "candidate_length": len(candidate_ids),
            "first_difference": first_difference,
        }
        if first_difference is not None:
            position = first_difference
            torch_top = (
                torch_request["logprobs"][position]
                if position < len(torch_request["logprobs"])
                else []
            )
            candidate_top = (
                candidate_request["logprobs"][position]
                if position < len(candidate_request["logprobs"])
                else []
            )
            report.update(
                torch_token=torch_ids[position] if position < len(torch_ids) else None,
                candidate_token=(
                    candidate_ids[position] if position < len(candidate_ids) else None
                ),
                torch_top_logprobs=torch_top,
                candidate_top_logprobs=candidate_top,
                torch_top2_margin=top_margin(torch_top),
                candidate_top2_margin=top_margin(candidate_top),
            )
        request_reports.append(report)
    return {
        "exact": exact_requests == len(request_reports),
        "exact_requests": exact_requests,
        "total_requests": len(request_reports),
        "candidate_backend": candidate_result["backend"],
        "torch_output_tok_s": torch_result["output_tok_s"],
        "candidate_output_tok_s": candidate_result["output_tok_s"],
        "speedup": candidate_result["output_tok_s"] / torch_result["output_tok_s"],
        "requests": request_reports,
    }


def compact_backend_result(result: dict[str, Any]) -> dict[str, Any]:
    """Omit full token/logprob arrays already summarized by the comparison."""
    keys = (
        "backend",
        "elapsed_s",
        "samples_s",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "output_tok_s",
        "total_tok_s",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "gpu",
        "software",
        "request_metrics",
        "prefix_cache",
        "engine_memory",
    )
    return {key: result[key] for key in keys if key in result}


def main() -> None:
    args = parse_args()
    if args.worker is not None:
        run_worker(args)
        return

    torch_result = run_backend(args, "torch")
    candidate_result = run_backend(args, args.candidate_backend)
    comparison = compare_results(torch_result, candidate_result)
    print(json.dumps(comparison, indent=2))
    if args.jsonl_out is not None:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": "rwkv7-kernel-benchmark-v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tag": args.benchmark_tag,
            "parameters": {
                "model": args.model,
                "dtype": args.dtype,
                "max_tokens": args.max_tokens,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "warmup_runs": args.warmup_runs,
                "repeats": args.repeats,
                "candidate_backend": args.candidate_backend,
                "prompt_count": len(load_prompts(args)),
                "enforce_eager": args.enforce_eager,
                "async_scheduling": args.async_scheduling,
                "enable_chunked_prefill": True,
                "prefix_cache": args.prefix_cache,
                "mamba_cache_mode": args.mamba_cache_mode,
            },
            "torch": compact_backend_result(torch_result),
            "candidate": compact_backend_result(candidate_result),
            "comparison": comparison,
        }
        with args.jsonl_out.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")
    if args.require_exact and not comparison["exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
