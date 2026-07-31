from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
LOCAL_LINK = re.compile(r"\[[^]]+\]\((?!https?://|mailto:|#)([^)#]+)(?:#[^)]+)?\)")


def test_local_markdown_links_resolve() -> None:
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    documents.extend(sorted((ROOT / "bench").glob("*.md")))
    missing = []
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for target in LOCAL_LINK.findall(text):
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert missing == []


def test_recorded_blackwell_result_is_passing_and_scoped() -> None:
    result_path = ROOT / "bench" / "results" / "blackwell_5070_fla_20260730.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["scope"] == "operator_correctness_and_microbenchmark_only"
    assert result["end_to_end_vllm_claim"] is False
    assert result["vllm_interface_baseline"] == ("837eae64580c885101ee95b073aafb27a485e7ce")
    assert result["correctness_thresholds"] == {
        "output_min_cosine": 0.9999,
        "state_max_abs_diff": 0.02,
    }
    assert result["environment"]["gpu"] == "NVIDIA GeForce RTX 5070 Laptop GPU"
    assert result["environment"]["compute_capability"] == [12, 0]
    assert len(result["rows"]) == 5
    assert all(row["passed"] for row in result["rows"])
    for row in result["rows"]:
        assert row["output"]["min_cosine"] >= 0.9999
        assert row["state"]["max_abs_diff"] <= 0.02


def test_recorded_remote_vllm_result_is_passing_sanitized_and_scoped() -> None:
    result_path = ROOT / "bench" / "results" / "remote_gpu_vllm_20260731.json"
    raw_result = result_path.read_text(encoding="utf-8")
    result = json.loads(raw_result)

    assert result["scope"] == "real_vllm_plugin_cache_and_engine_correctness"
    assert result["end_to_end_generation_validated"] is True
    assert result["performance_claim"] is False
    assert result["upstream"] == {
        "precompiled_binary_commit": "553fcb82d5602c75fb6ab41b6dc3c46f480c1785",
        "python_source_commit": "837eae64580c885101ee95b073aafb27a485e7ce",
        "version": "0.26.1rc1.dev146+g837eae645.d20260731",
    }

    checkpoint = result["checkpoint_contract"]
    assert checkpoint["trust_remote_code"] is False
    assert checkpoint["remote_code_files_transferred"] is False
    assert checkpoint["fixed_revision_claimed"] is False
    assert checkpoint["weights_committed"] is False
    assert checkpoint["engine_checkpoint"]["size_bytes"] == 382110672
    assert checkpoint["engine_checkpoint"]["sha256"] == (
        "12d208adf2880927615656c2dc3f6fb6a3ea3120a9ed9fdfeeea55c841723d79"
    )

    environments = {environment["alias"]: environment for environment in result["environments"]}
    assert set(environments) == {"gpu4080", "WZU_Server"}
    assert environments["gpu4080"]["compute_capability"] == [8, 9]
    assert environments["gpu4080"]["engine_gpu_memory_utilization"] == 0.15
    assert environments["gpu4080"]["default_0_75_engine_previously_passed"] is True
    assert environments["WZU_Server"]["compute_capability"] == [7, 0]
    assert environments["WZU_Server"]["engine_gpu_memory_utilization"] == 0.05
    assert all(
        suite == {"failed": 0, "passed": 76, "skipped": 4}
        for suite in result["test_suites"].values()
    )

    assert result["observed_results"]["prefix_cache"] == {
        "cache_mode": "align",
        "cold_generation_equal_to_cached_generation": True,
        "minimum_observed_contract": "num_cached_tokens >= 16",
        "reset_cold_num_cached_tokens": 0,
        "status": "passed_on_both",
    }
    assert "/home/" not in raw_result
    assert re.search(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)", raw_result) is None


def test_pull_request_ci_checks_out_the_reviewed_head_commit() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "ref: ${{ github.event.pull_request.head.sha || github.sha }}" in workflow
    assert "python -m pip install torch transformers pytest ruff" in workflow
