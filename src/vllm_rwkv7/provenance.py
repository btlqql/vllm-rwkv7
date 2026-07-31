"""Repository-history checks for the single-author clean-room contract."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

ALLOWED_AUTHOR_NAME = "btlqql"
ALLOWED_AUTHOR_EMAIL = "2977859784@qq.com"


@dataclass(frozen=True, slots=True)
class CommitIdentity:
    commit: str
    author_name: str
    author_email: str
    committer_name: str
    committer_email: str
    message: str


def validate_commit_identities(commits: list[CommitIdentity]) -> None:
    errors = []
    for commit in commits:
        if commit.author_name != ALLOWED_AUTHOR_NAME or commit.author_email != ALLOWED_AUTHOR_EMAIL:
            errors.append(
                f"{commit.commit}: unexpected author {commit.author_name} <{commit.author_email}>"
            )
        if (
            commit.committer_name != ALLOWED_AUTHOR_NAME
            or commit.committer_email != ALLOWED_AUTHOR_EMAIL
        ):
            errors.append(
                f"{commit.commit}: unexpected committer "
                f"{commit.committer_name} <{commit.committer_email}>"
            )
        if "co-authored-by:" in commit.message.lower():
            errors.append(f"{commit.commit}: Co-authored-by trailers are not allowed")
    if errors:
        raise ValueError("\n".join(errors))


def read_git_identities(repository: Path) -> list[CommitIdentity]:
    has_head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    if has_head.returncode:
        return []

    result = subprocess.run(
        ["git", "log", "--format=%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%B%x1e"],
        cwd=repository,
        capture_output=True,
        check=True,
    )
    records = result.stdout.decode("utf-8", errors="strict").split("\x1e")
    commits = []
    for record in records:
        record = record.strip("\r\n")
        if not record:
            continue
        commit, author_name, author_email, committer_name, committer_email, message = record.split(
            "\x1f", maxsplit=5
        )
        commits.append(
            CommitIdentity(
                commit=commit,
                author_name=author_name,
                author_email=author_email,
                committer_name=committer_name,
                committer_email=committer_email,
                message=message,
            )
        )
    return commits


def audit_repository(repository: Path) -> None:
    validate_commit_identities(read_git_identities(repository))
