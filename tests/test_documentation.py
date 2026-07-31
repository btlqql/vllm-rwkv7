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


def test_pull_request_ci_checks_out_the_reviewed_head_commit() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "ref: ${{ github.event.pull_request.head.sha || github.sha }}" in workflow
