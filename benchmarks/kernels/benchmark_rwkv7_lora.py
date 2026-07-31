# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise RWKV-7 runtime LoRA load, switch, generation, and removal."""

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--prompt-token-length", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--jsonl-out", type=Path)
    parser.add_argument("--benchmark-tag", default="")
    return parser.parse_args()


def create_adapter(
    output_dir: Path,
    *,
    hidden_size: int,
    rank: int,
    seed: int,
) -> None:
    import torch
    from safetensors.torch import save_file

    generator = torch.Generator(device="cpu").manual_seed(seed)
    lora_a = torch.randn(
        rank, hidden_size, generator=generator, dtype=torch.float16
    ).mul_(0.1)
    lora_b = torch.randn(
        hidden_size, rank, generator=generator, dtype=torch.float16
    ).mul_(0.1)
    prefix = "base_model.model.model.layers.0.attn.r_proj"
    weights = {
        f"{prefix}.lora_A.weight": lora_a,
        f"{prefix}.lora_B.weight": lora_b,
    }
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": "rwkv7-runtime-smoke",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": ["r_proj"],
    }
    output_dir.mkdir(parents=True)
    (output_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    save_file(weights, output_dir / "adapter_model.safetensors")


def token_signature(outputs) -> tuple[int, ...]:
    return tuple(outputs[0].outputs[0].token_ids)


def timed_generate(llm, prompts, sampling, **kwargs):
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False, **kwargs)
    return token_signature(outputs), time.perf_counter() - started


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=False,
        skip_tokenizer_init=True,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=args.rank,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    hidden_size = int(llm.llm_engine.model_config.hf_config.hidden_size)
    prompt = {
        "prompt_token_ids": [
            3 + (token_index % 124) for token_index in range(args.prompt_token_length)
        ]
    }
    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    with tempfile.TemporaryDirectory(prefix="rwkv7-lora-") as temp_dir:
        root = Path(temp_dir)
        adapter_one = root / "adapter-one"
        adapter_two = root / "adapter-two"
        create_adapter(
            adapter_one,
            hidden_size=hidden_size,
            rank=args.rank,
            seed=1,
        )
        create_adapter(
            adapter_two,
            hidden_size=hidden_size,
            rank=args.rank,
            seed=2,
        )

        request_one = LoRARequest("rwkv7-one", 1, str(adapter_one))
        request_two = LoRARequest("rwkv7-two", 2, str(adapter_two))
        base_before, base_before_s = timed_generate(llm, [prompt], sampling)
        assert llm.llm_engine.add_lora(request_one)
        assert llm.llm_engine.add_lora(request_two)
        assert set(llm.llm_engine.list_loras()) == {1, 2}

        output_one, adapter_one_s = timed_generate(
            llm,
            [prompt],
            sampling,
            lora_request=request_one,
        )
        output_two, adapter_two_s = timed_generate(
            llm,
            [prompt],
            sampling,
            lora_request=request_two,
        )
        assert output_one != output_two
        assert output_one != base_before or output_two != base_before

        assert llm.llm_engine.remove_lora(1)
        assert set(llm.llm_engine.list_loras()) == {2}
        output_two_after_switch, adapter_two_after_switch_s = timed_generate(
            llm,
            [prompt],
            sampling,
            lora_request=request_two,
        )
        assert output_two_after_switch == output_two

        assert llm.llm_engine.remove_lora(2)
        assert set(llm.llm_engine.list_loras()) == set()
        base_after, base_after_s = timed_generate(llm, [prompt], sampling)
        assert base_after == base_before

    import torch

    import vllm

    report = {
        "schema": "rwkv7-lora-benchmark-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": args.benchmark_tag,
        "model": args.model,
        "dtype": args.dtype,
        "gpu": {
            "name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "vllm": vllm.__version__,
        },
        "prompt_token_length": args.prompt_token_length,
        "max_tokens": args.max_tokens,
        "rank": args.rank,
        "target_module": "model.layers.0.attn.r_proj",
        "base_restored": base_after == base_before,
        "any_adapter_effective": (
            output_one != base_before or output_two != base_before
        ),
        "adapters_distinct": output_one != output_two,
        "switch_stable": output_two_after_switch == output_two,
        "loaded_after_remove": sorted(llm.llm_engine.list_loras()),
        "base_tokens": list(base_before),
        "adapter_one_tokens": list(output_one),
        "adapter_two_tokens": list(output_two),
        "latency_s": {
            "base_before": base_before_s,
            "adapter_one": adapter_one_s,
            "adapter_two": adapter_two_s,
            "adapter_two_after_switch": adapter_two_after_switch_s,
            "base_after": base_after_s,
        },
    }
    print(json.dumps(report, indent=2))
    if args.jsonl_out is not None:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl_out.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(report, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
