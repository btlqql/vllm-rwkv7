from __future__ import annotations

import pytest

from vllm_rwkv7.provenance import CommitIdentity, validate_commit_identities


def test_btlqql_identity_is_accepted() -> None:
    validate_commit_identities(
        [
            CommitIdentity(
                commit="abc123",
                author_name="btlqql",
                author_email="2977859784@qq.com",
                committer_name="btlqql",
                committer_email="2977859784@qq.com",
                message="feat: initialize clean-room plugin",
            )
        ]
    )


def test_other_author_is_rejected() -> None:
    with pytest.raises(ValueError, match="unexpected author"):
        validate_commit_identities(
            [
                CommitIdentity(
                    commit="def456",
                    author_name="someone-else",
                    author_email="else@example.com",
                    committer_name="btlqql",
                    committer_email="2977859784@qq.com",
                    message="feat: foreign implementation",
                )
            ]
        )


def test_coauthor_trailer_is_rejected() -> None:
    with pytest.raises(ValueError, match="Co-authored-by"):
        validate_commit_identities(
            [
                CommitIdentity(
                    commit="abc123",
                    author_name="btlqql",
                    author_email="2977859784@qq.com",
                    committer_name="btlqql",
                    committer_email="2977859784@qq.com",
                    message=(
                        "feat: initialize clean-room plugin\n\n"
                        "Co-authored-by: someone <else@example.com>"
                    ),
                )
            ]
        )


def test_other_committer_is_rejected() -> None:
    with pytest.raises(ValueError, match="unexpected committer"):
        validate_commit_identities(
            [
                CommitIdentity(
                    commit="abc123",
                    author_name="btlqql",
                    author_email="2977859784@qq.com",
                    committer_name="someone-else",
                    committer_email="else@example.com",
                    message="feat: commit through another identity",
                )
            ]
        )
