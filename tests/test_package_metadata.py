from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from vllm_rwkv7 import __version__

ROOT = Path(__file__).parents[1]


def test_distribution_metadata_and_entry_point() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["name"] == "vllm-rwkv7"
    assert metadata["project"]["version"] == __version__
    assert metadata["project"]["authors"] == [{"name": "btlqql", "email": "2977859784@qq.com"}]
    assert metadata["project"]["entry-points"]["vllm.general_plugins"] == {
        "rwkv7": "vllm_rwkv7.plugin:register"
    }
    assert "vllm>=0.11.1" in metadata["project"]["dependencies"]


def test_authors_file_has_only_the_repository_author() -> None:
    authors = [
        line.strip()
        for line in (ROOT / "AUTHORS.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("- ")
    ]

    assert authors == ["- btlqql <2977859784@qq.com>"]


def test_codeowners_assigns_every_path_only_to_btlqql() -> None:
    codeowners = (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")

    assert codeowners.splitlines() == ["* @btlqql"]
