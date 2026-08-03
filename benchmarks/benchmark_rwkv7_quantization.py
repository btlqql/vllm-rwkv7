# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark RWKV7 16-bit and weight-only quantization implementations.

Every setting runs in a fresh process.  Besides median output throughput, the
report extracts vLLM's model-resident GPU-memory measurement and compares
greedy tokens with the 16-bit run.  The setting name ``fp16`` is retained for
CLI compatibility, while ``--dtype`` selects FP16 or BF16.  The default gates
encode the production target: model memory must decrease, throughput must not
regress, and generated tokens must match the reference. An explicit fixed-prompt
log-probability gate can qualify policies whose continuations diverge.
"""

import argparse
import json
import math
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

MODEL_MEMORY_PATTERN = re.compile(r"Model loading took ([0-9.]+) GiB memory")
ENGINE_MEMORY_PATTERN = re.compile(
    r"Actual usage is ([0-9.]+) GiB for consumed memory .*?"
    r"([0-9.]+) GiB for peak activation, and ([0-9.]+) GiB for CUDAGraph "
    r"memory\..*?Current kv cache memory in use is ([0-9.]+) GiB"
)
NOISY_BNB_MESSAGES = ("MatMul8bitLt: inputs will be cast",)
EXCEPTION_SUMMARY_PATTERN = re.compile(
    r"(?:AssertionError|ImportError|ModuleNotFoundError|RuntimeError|ValueError): .+"
)
SETTINGS = (
    "fp16",
    "ct-w8",
    "ct-w4",
    "online-int8",
    "torchao-int8",
    "torchao-int4",
    "bnb-int8",
    "bnb-nf4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="16-bit reference checkpoint")
    parser.add_argument("--tokenizer")
    parser.add_argument(
        "--ct-w8-model", help="Compressed-tensors pack-quantized W8A16 model"
    )
    parser.add_argument(
        "--ct-w4-model", help="Compressed-tensors pack-quantized W4A16 model"
    )
    parser.add_argument("--int8-model", help="Pre-quantized BitsAndBytes INT8 model")
    parser.add_argument(
        "--int4-model",
        help="Pre-quantized BitsAndBytes NF4 model; omit for inflight NF4",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=SETTINGS,
        default=("fp16", "online-int8"),
    )
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument(
        "--prompt-token-ids-file",
        type=Path,
        help="JSON array of token-ID arrays for tokenizer-independent evaluation.",
    )
    parser.add_argument(
        "--prompt-token-length",
        action="append",
        type=int,
        default=[],
        help="Use a deterministic token-ID prompt of this length; repeat for batches.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--logprobs", type=int, default=1)
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="full-engine warmups; three also settles lazy BitsAndBytes state",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", default="half")
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
        "--ignore-eos", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-gates", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-repeatable", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--repeat-logprob-margin-tolerance",
        type=float,
        default=0.0,
        help=(
            "Allow a repeated greedy-token flip only when both runs place the "
            "two competing tokens within this log-probability margin."
        ),
    )
    parser.add_argument(
        "--record-unsupported",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record unsupported quantization settings instead of aborting the report.",
    )
    parser.add_argument("--min-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-memory-reduction", type=float, default=0.01)
    parser.add_argument("--min-token-agreement", type=float, default=1.0)
    parser.add_argument("--max-prompt-logprob-mean-abs-error", type=float)
    parser.add_argument("--max-prompt-perplexity-ratio", type=float)
    parser.add_argument(
        "--online-int8-target-regex",
        help=(
            "Full-match regex selecting LinearBase modules for online INT8; "
            "all other linear modules stay in the model dtype."
        ),
    )
    parser.add_argument(
        "--rwkv7-backend",
        choices=("auto", "torch", "triton"),
        default="torch",
        help=(
            "RWKV7 recurrent backend used by every quantization setting. "
            "Torch remains the default so weight-format comparisons preserve "
            "the established baseline."
        ),
    )
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        help="Append one machine-readable benchmark record to this JSONL file.",
    )
    parser.add_argument("--benchmark-tag", default="")
    parser.add_argument(
        "--worker-setting",
        choices=SETTINGS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-model", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-inflight-nf4", action="store_true", help=argparse.SUPPRESS
    )
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[Any]:
    if args.warmup_runs < 0 or args.repeats < 1:
        raise ValueError("--warmup-runs must be >= 0 and --repeats must be >= 1")
    if args.prompt_token_length:
        if (
            args.prompt
            or args.prompts_file is not None
            or args.prompt_token_ids_file is not None
        ):
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
    if args.prompt_token_ids_file is not None:
        if args.prompt or args.prompts_file is not None:
            raise ValueError(
                "--prompt-token-ids-file cannot be combined with text prompts"
            )
        loaded = json.loads(args.prompt_token_ids_file.read_text())
        if not (
            isinstance(loaded, list)
            and loaded
            and all(
                isinstance(prompt, list)
                and prompt
                and all(
                    isinstance(token_id, int) and token_id >= 0 for token_id in prompt
                )
                for prompt in loaded
            )
        ):
            raise ValueError(
                "--prompt-token-ids-file must contain a non-empty JSON array "
                "of non-empty, non-negative integer arrays"
            )
        return [{"prompt_token_ids": prompt} for prompt in loaded]
    prompts = list(args.prompt)
    if args.prompts_file is not None:
        loaded = json.loads(args.prompts_file.read_text())
        if not isinstance(loaded, list) or not all(isinstance(x, str) for x in loaded):
            raise ValueError("--prompts-file must contain a JSON array of strings")
        prompts.extend(loaded)
    return prompts or DEFAULT_PROMPTS


def repeat_mismatch_is_tied(mismatch: dict[str, Any], margin_tolerance: float) -> bool:
    margins = (mismatch.get("reference_margin"), mismatch.get("candidate_margin"))
    return margin_tolerance > 0 and all(
        isinstance(margin, (int, float)) and 0 <= margin <= margin_tolerance
        for margin in margins
    )


def run_worker(args: argparse.Namespace) -> None:
    assert args.worker_setting is not None
    assert args.worker_model is not None

    # Keep one recurrent backend across all settings so comparisons isolate
    # the selected weight format. Torch remains the fail-closed default.
    import os

    os.environ["VLLM_RWKV7_KERNEL"] = args.rwkv7_backend
    if args.require_repeatable:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.require_repeatable and args.worker_setting == "ct-w4":
        disabled_kernels = list(
            filter(None, os.environ.get("VLLM_DISABLED_KERNELS", "").split(","))
        )
        if "RDNA3W4A16LinearKernel" not in disabled_kernels:
            disabled_kernels.append("RDNA3W4A16LinearKernel")
        os.environ["VLLM_DISABLED_KERNELS"] = ",".join(disabled_kernels)
    # Keep the benchmark self-contained on CUDA hosts without a full nvcc
    # toolchain; sampling is outside the measured model/kernel scope.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import torch

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.platforms import current_platform

    if args.require_repeatable:
        torch.use_deterministic_algorithms(True)

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
    if args.worker_setting == "online-int8":
        # vLLM's online-quant configuration was generalized from the original
        # scheme/override API to QuantSpec. Supporting both keeps this benchmark
        # usable while bisecting releases and downstream integration branches.
        from vllm.config import quantization as online_quant_config

        if hasattr(online_quant_config, "QuantizationConfigArgs"):
            quantization_config = {"linear": "int8_per_channel_static"}
            if args.online_int8_target_regex is not None:
                target = args.online_int8_target_regex
                quantization_config["ignore"] = [rf"re:^(?!(?:{target})$).*"]
        else:
            quantization_config = {
                "linear_scheme_override": "int8_per_channel_weight_only"
            }
        llm_kwargs.update(
            quantization="online",
            quantization_config=quantization_config,
        )
    elif args.worker_setting == "bnb-nf4" and args.worker_inflight_nf4:
        llm_kwargs["quantization"] = "bitsandbytes"
    elif args.worker_setting.startswith("torchao-"):
        from torchao.core.config import config_to_dict
        from torchao.quantization import (
            Int4WeightOnlyConfig,
            Int8WeightOnlyConfig,
        )

        if args.worker_setting == "torchao-int8":
            torchao_config = Int8WeightOnlyConfig()
        else:
            if args.dtype not in ("bfloat16", "bf16"):
                raise ValueError("torchao-int4 requires --dtype bfloat16")
            torchao_config = Int4WeightOnlyConfig(
                group_size=128,
                int4_packing_format="tile_packed_to_4d",
            )
        llm_kwargs.update(
            quantization="torchao",
            hf_overrides={
                "quantization_config_dict_json": json.dumps(
                    config_to_dict(torchao_config)
                )
            },
        )

    load_started = time.perf_counter()
    uses_token_ids = bool(args.prompt_token_length or args.prompt_token_ids_file)
    if uses_token_ids:
        llm_kwargs["skip_tokenizer_init"] = True
    llm = LLM(
        model=args.worker_model,
        dtype=args.dtype,
        trust_remote_code=False,
        enforce_eager=args.enforce_eager,
        async_scheduling=args.async_scheduling,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=False,
        tokenizer=(None if uses_token_ids else args.tokenizer or args.model),
        **llm_kwargs,
    )
    load_s = time.perf_counter() - load_started
    prompts = load_prompts(args)
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
        prompt_logprobs=args.prompt_logprobs,
        ignore_eos=args.ignore_eos,
    )
    for _ in range(args.warmup_runs):
        llm.generate(prompts, sampling_params, use_tqdm=False)

    samples = []
    signatures = []
    outputs_by_run = []
    outputs = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        samples.append(time.perf_counter() - started)
        outputs_by_run.append(outputs)
        signatures.append([tuple(item.outputs[0].token_ids) for item in outputs])
    assert outputs is not None
    repeat_mismatch = None
    tolerated_repeat_mismatches = []
    exact_repeatable = all(signature == signatures[0] for signature in signatures[1:])
    for run_index, signature in enumerate(signatures[1:], start=1):
        if signature == signatures[0]:
            continue
        mismatch = None
        for request_index, (reference_ids, candidate_ids) in enumerate(
            zip(signatures[0], signature)
        ):
            if reference_ids == candidate_ids:
                continue
            common = min(len(reference_ids), len(candidate_ids))
            token_index = next(
                (
                    index
                    for index in range(common)
                    if reference_ids[index] != candidate_ids[index]
                ),
                common,
            )
            mismatch = {
                "run": run_index,
                "request": request_index,
                "token": token_index,
            }
            if token_index < common:
                reference_token = reference_ids[token_index]
                candidate_token = candidate_ids[token_index]
                reference_logprobs = (
                    outputs_by_run[0][request_index].outputs[0].logprobs or []
                )
                candidate_logprobs = (
                    outputs_by_run[run_index][request_index].outputs[0].logprobs or []
                )
                reference_step = (
                    reference_logprobs[token_index]
                    if token_index < len(reference_logprobs)
                    else {}
                )
                candidate_step = (
                    candidate_logprobs[token_index]
                    if token_index < len(candidate_logprobs)
                    else {}
                )

                def margin(step_logprobs, selected_token, competing_token):
                    selected = step_logprobs.get(selected_token)
                    competing = step_logprobs.get(competing_token)
                    if selected is None or competing is None:
                        return None
                    return float(selected.logprob - competing.logprob)

                mismatch.update(
                    reference_token=reference_token,
                    candidate_token=candidate_token,
                    reference_margin=margin(
                        reference_step, reference_token, candidate_token
                    ),
                    candidate_margin=margin(
                        candidate_step, candidate_token, reference_token
                    ),
                )
            break
        if mismatch is None:
            mismatch = {"run": run_index, "request": None, "token": None}
        if repeat_mismatch_is_tied(mismatch, args.repeat_logprob_margin_tolerance):
            tolerated_repeat_mismatches.append(mismatch)
            continue
        repeat_mismatch = mismatch
        break
    if repeat_mismatch is not None and args.require_repeatable:
        raise RuntimeError(
            f"{args.worker_setting} output changed across repeated runs: "
            f"{repeat_mismatch}"
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
    generated_token_logprobs = []
    prompt_token_logprobs = []
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
            }
        )
        token_logprobs = []
        output_logprobs = item.outputs[0].logprobs or []
        for token_id, step_logprobs in zip(item.outputs[0].token_ids, output_logprobs):
            token_entry = step_logprobs.get(token_id)
            token_logprobs.append(
                None if token_entry is None else float(token_entry.logprob)
            )
        generated_token_logprobs.append(token_logprobs)
        prompt_logprobs = []
        for token_id, step_logprobs in zip(
            item.prompt_token_ids, item.prompt_logprobs or []
        ):
            token_entry = None if step_logprobs is None else step_logprobs.get(token_id)
            prompt_logprobs.append(
                None if token_entry is None else float(token_entry.logprob)
            )
        prompt_token_logprobs.append(prompt_logprobs)
    result = {
        "setting": args.worker_setting,
        "dtype": args.dtype,
        "model": args.worker_model,
        "load_s": load_s,
        "elapsed_s": elapsed,
        "samples_s": samples,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_tok_s": output_tokens / elapsed,
        "total_tok_s": (input_tokens + output_tokens) / elapsed,
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
        "generated_token_logprobs": generated_token_logprobs,
        "prompt_token_logprobs": prompt_token_logprobs,
        "repeatable": repeat_mismatch is None,
        "exact_repeatable": exact_repeatable,
        "repeat_mismatch": repeat_mismatch,
        "tolerated_repeat_mismatches": tolerated_repeat_mismatches,
        "rwkv7_backend": args.rwkv7_backend,
        "online_int8_target_regex": args.online_int8_target_regex,
        "requests": [list(item.outputs[0].token_ids) for item in outputs],
    }
    print("RESULT_JSON " + json.dumps(result), flush=True)


def _setting_model(args: argparse.Namespace, setting: str) -> tuple[str, bool]:
    if setting == "fp16":
        return args.model, False
    if setting == "ct-w8":
        if args.ct_w8_model is None:
            raise ValueError("--ct-w8-model is required for ct-w8")
        return args.ct_w8_model, False
    if setting == "ct-w4":
        if args.ct_w4_model is None:
            raise ValueError("--ct-w4-model is required for ct-w4")
        return args.ct_w4_model, False
    if setting == "online-int8" or setting.startswith("torchao-"):
        return args.model, False
    if setting == "bnb-int8":
        if args.int8_model is None:
            raise ValueError("--int8-model is required for bnb-int8")
        return args.int8_model, False
    if args.int4_model is None:
        return args.model, True
    return args.int4_model, False


def select_unsupported_reason(exception_summaries: list[str]) -> str:
    unique_summaries = list(dict.fromkeys(exception_summaries))
    specific_summaries = [
        summary
        for summary in unique_summaries
        if "Engine core initialization failed" not in summary
    ]
    if specific_summaries:
        return specific_summaries[0]
    if unique_summaries:
        return unique_summaries[0]
    return "worker exited without an exception summary"


def run_setting(args: argparse.Namespace, setting: str) -> dict[str, Any]:
    model, inflight_nf4 = _setting_model(args, setting)
    command = [
        sys.executable,
        __file__,
        "--model",
        args.model,
        "--worker-model",
        model,
        "--worker-setting",
        setting,
    ]
    for name in (
        "max_tokens",
        "logprobs",
        "prompt_logprobs",
        "warmup_runs",
        "repeats",
        "repeat_logprob_margin_tolerance",
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
    if args.prompt_token_ids_file is not None:
        command.extend(["--prompt-token-ids-file", str(args.prompt_token_ids_file)])
    if args.online_int8_target_regex is not None:
        command.extend(["--online-int8-target-regex", args.online_int8_target_regex])
    command.extend(["--rwkv7-backend", args.rwkv7_backend])
    command.append("--enforce-eager" if args.enforce_eager else "--no-enforce-eager")
    command.append(
        "--async-scheduling" if args.async_scheduling else "--no-async-scheduling"
    )
    command.append("--ignore-eos" if args.ignore_eos else "--no-ignore-eos")
    command.append(
        "--require-repeatable" if args.require_repeatable else "--no-require-repeatable"
    )
    if inflight_nf4:
        command.append("--worker-inflight-nf4")

    result = None
    model_memory_gib = None
    engine_memory = None
    output_tail: list[str] = []
    exception_summaries: list[str] = []
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        output_tail.append(line.rstrip())
        output_tail = output_tail[-20:]
        if match := EXCEPTION_SUMMARY_PATTERN.search(line):
            exception_summaries.append(match.group(0).strip())
        if match := MODEL_MEMORY_PATTERN.search(line):
            model_memory_gib = float(match.group(1))
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
        elif not any(message in line for message in NOISY_BNB_MESSAGES):
            print(f"[{setting}] {line}", end="")
    return_code = process.wait()
    if return_code != 0:
        if args.record_unsupported and setting != "fp16":
            return {
                "setting": setting,
                "model": model,
                "supported": False,
                "unsupported_reason": select_unsupported_reason(exception_summaries),
                "error": "\n".join(output_tail),
            }
        raise RuntimeError(f"{setting} worker exited with status {return_code}")
    if result is None:
        raise RuntimeError(f"{setting} worker produced no result")
    result["model_memory_gib"] = model_memory_gib
    result["engine_memory"] = engine_memory
    result["supported"] = True
    return result


def compare_with_reference(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    if len(reference["requests"]) != len(candidate["requests"]):
        raise RuntimeError(
            "16-bit and quantized settings returned different batch sizes"
        )

    matching_tokens = 0
    total_tokens = 0
    exact_requests = 0
    request_reports = []
    logprob_errors = []
    prompt_logprob_errors = []
    reference_prompt_nlls = []
    candidate_prompt_nlls = []
    for index, (ref_ids, candidate_ids) in enumerate(
        zip(reference["requests"], candidate["requests"])
    ):
        common = min(len(ref_ids), len(candidate_ids))
        first_difference = next(
            (pos for pos in range(common) if ref_ids[pos] != candidate_ids[pos]),
            None,
        )
        if first_difference is None and len(ref_ids) != len(candidate_ids):
            first_difference = common
        exact = first_difference is None
        exact_requests += int(exact)
        matching_tokens += sum(
            ref_token == candidate_token
            for ref_token, candidate_token in zip(ref_ids, candidate_ids)
        )
        total_tokens += max(len(ref_ids), len(candidate_ids))
        reference_logprobs = reference["generated_token_logprobs"][index]
        candidate_logprobs = candidate["generated_token_logprobs"][index]
        for token_index in range(common):
            if ref_ids[token_index] != candidate_ids[token_index]:
                continue
            ref_logprob = reference_logprobs[token_index]
            candidate_logprob = candidate_logprobs[token_index]
            if ref_logprob is not None and candidate_logprob is not None:
                logprob_errors.append(abs(candidate_logprob - ref_logprob))
        request_reports.append(
            {
                "request": index,
                "exact": exact,
                "first_difference": first_difference,
                "reference_length": len(ref_ids),
                "candidate_length": len(candidate_ids),
            }
        )
        reference_prompt_logprobs = reference["prompt_token_logprobs"][index]
        candidate_prompt_logprobs = candidate["prompt_token_logprobs"][index]
        for ref_logprob, candidate_logprob in zip(
            reference_prompt_logprobs, candidate_prompt_logprobs
        ):
            if ref_logprob is None or candidate_logprob is None:
                continue
            prompt_logprob_errors.append(abs(candidate_logprob - ref_logprob))
            reference_prompt_nlls.append(-ref_logprob)
            candidate_prompt_nlls.append(-candidate_logprob)

    memory_reduction = None
    if (
        reference["model_memory_gib"] is not None
        and candidate["model_memory_gib"] is not None
    ):
        memory_reduction = 1.0 - (
            candidate["model_memory_gib"] / reference["model_memory_gib"]
        )
    return {
        "setting": candidate["setting"],
        "speed_ratio": candidate["output_tok_s"] / reference["output_tok_s"],
        "memory_reduction": memory_reduction,
        "token_agreement": matching_tokens / total_tokens if total_tokens else 1.0,
        "generated_token_logprob_max_abs_error": (
            max(logprob_errors) if logprob_errors else None
        ),
        "generated_token_logprob_mean_abs_error": (
            statistics.mean(logprob_errors) if logprob_errors else None
        ),
        "prompt_logprob_max_abs_error": (
            max(prompt_logprob_errors) if prompt_logprob_errors else None
        ),
        "prompt_logprob_mean_abs_error": (
            statistics.mean(prompt_logprob_errors) if prompt_logprob_errors else None
        ),
        "prompt_perplexity_ratio": (
            math.exp(
                statistics.mean(candidate_prompt_nlls)
                - statistics.mean(reference_prompt_nlls)
            )
            if reference_prompt_nlls
            else None
        ),
        "exact_requests": exact_requests,
        "total_requests": len(request_reports),
        "reference_dtype": reference["dtype"],
        "reference_output_tok_s": reference["output_tok_s"],
        "candidate_output_tok_s": candidate["output_tok_s"],
        "reference_model_memory_gib": reference["model_memory_gib"],
        "candidate_model_memory_gib": candidate["model_memory_gib"],
        "requests": request_reports,
    }


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep JSONL evidence useful without duplicating token/logprob arrays."""
    keys = (
        "supported",
        "unsupported_reason",
        "dtype",
        "model",
        "model_memory_gib",
        "output_tok_s",
        "total_tok_s",
        "input_tokens",
        "output_tokens",
        "elapsed_s",
        "samples_s",
        "repeatable",
        "exact_repeatable",
        "repeat_mismatch",
        "tolerated_repeat_mismatches",
        "rwkv7_backend",
        "online_int8_target_regex",
        "engine_memory",
    )
    return {key: result[key] for key in keys if key in result}


def main() -> None:
    args = parse_args()
    if args.worker_setting is not None:
        run_worker(args)
        return
    if not 0.0 <= args.min_token_agreement <= 1.0:
        raise ValueError("--min-token-agreement must be between 0 and 1")
    if args.repeat_logprob_margin_tolerance < 0:
        raise ValueError("--repeat-logprob-margin-tolerance must be non-negative")
    if (
        args.max_prompt_logprob_mean_abs_error is not None
        and args.max_prompt_logprob_mean_abs_error < 0
    ):
        raise ValueError("--max-prompt-logprob-mean-abs-error must be non-negative")
    if (
        args.max_prompt_perplexity_ratio is not None
        and args.max_prompt_perplexity_ratio <= 0
    ):
        raise ValueError("--max-prompt-perplexity-ratio must be positive")
    if "fp16" not in args.settings:
        raise ValueError("--settings must include fp16 as the reference")

    results = {setting: run_setting(args, setting) for setting in args.settings}
    comparisons = [
        compare_with_reference(results["fp16"], results[setting])
        for setting in args.settings
        if setting != "fp16" and results[setting].get("supported", True)
    ]
    report = {"results": results, "comparisons": comparisons}
    print(json.dumps(report, indent=2))

    failed = [
        f"{setting} unsupported"
        for setting in args.settings
        if setting != "fp16" and not results[setting].get("supported", True)
    ]
    for comparison in comparisons:
        if comparison["speed_ratio"] < args.min_speed_ratio:
            failed.append(
                f"{comparison['setting']} speed_ratio={comparison['speed_ratio']:.4f}"
            )
        reduction = comparison["memory_reduction"]
        if reduction is None or reduction < args.min_memory_reduction:
            failed.append(f"{comparison['setting']} memory_reduction={reduction}")
        if comparison["token_agreement"] < args.min_token_agreement:
            failed.append(
                f"{comparison['setting']} "
                f"token_agreement={comparison['token_agreement']:.4f}"
            )
        prompt_error = comparison["prompt_logprob_mean_abs_error"]
        if args.max_prompt_logprob_mean_abs_error is not None and (
            prompt_error is None
            or prompt_error > args.max_prompt_logprob_mean_abs_error
        ):
            failed.append(
                f"{comparison['setting']} prompt_logprob_mean_abs_error={prompt_error}"
            )
        perplexity_ratio = comparison["prompt_perplexity_ratio"]
        if args.max_prompt_perplexity_ratio is not None and (
            perplexity_ratio is None
            or perplexity_ratio > args.max_prompt_perplexity_ratio
        ):
            failed.append(
                f"{comparison['setting']} prompt_perplexity_ratio={perplexity_ratio}"
            )

    if args.jsonl_out is not None:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        identity = next(
            (
                result
                for result in results.values()
                if result.get("supported", True) and result.get("gpu") is not None
            ),
            {},
        )
        record = {
            "schema": "rwkv7-quantization-benchmark-v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tag": args.benchmark_tag,
            "gpu": identity.get("gpu"),
            "software": identity.get("software"),
            "parameters": {
                "model": args.model,
                "dtype": args.dtype,
                "settings": args.settings,
                "max_tokens": args.max_tokens,
                "logprobs": args.logprobs,
                "prompt_logprobs": args.prompt_logprobs,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "warmup_runs": args.warmup_runs,
                "repeats": args.repeats,
                "repeat_logprob_margin_tolerance": (
                    args.repeat_logprob_margin_tolerance
                ),
                "require_repeatable": args.require_repeatable,
                "require_gates": args.require_gates,
                "prompt_count": len(load_prompts(args)),
                "prompt_token_ids_file": (
                    None
                    if args.prompt_token_ids_file is None
                    else str(args.prompt_token_ids_file)
                ),
                "enforce_eager": args.enforce_eager,
                "async_scheduling": args.async_scheduling,
                "ignore_eos": args.ignore_eos,
                "enable_chunked_prefill": True,
                "rwkv7_backend": args.rwkv7_backend,
                "online_int8_target_regex": args.online_int8_target_regex,
                "min_speed_ratio": args.min_speed_ratio,
                "min_memory_reduction": args.min_memory_reduction,
                "min_token_agreement": args.min_token_agreement,
                "max_prompt_logprob_mean_abs_error": (
                    args.max_prompt_logprob_mean_abs_error
                ),
                "max_prompt_perplexity_ratio": args.max_prompt_perplexity_ratio,
            },
            "results": {
                setting: compact_result(result) for setting, result in results.items()
            },
            "comparisons": comparisons,
            "gate_passed": not failed,
            "failed_gates": failed,
        }
        with args.jsonl_out.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")

    if args.require_gates and failed:
        print("FAILED_GATES " + "; ".join(failed), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
