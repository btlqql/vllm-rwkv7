#!/usr/bin/env python3
"""Fail when repository history violates the btlqql-only author contract."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vllm_rwkv7.provenance import audit_repository  # noqa: E402


def main() -> int:
    try:
        audit_repository(ROOT)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print("provenance check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
