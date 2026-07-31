from __future__ import annotations

import pytest

from vllm_rwkv7.cache import PAD_STATE_SLOT, plan_packed_state_spans


def test_plan_distinguishes_fresh_prefill_from_cached_continuation() -> None:
    spans = plan_packed_state_spans(
        query_start_loc=[0, 2, 3],
        state_slots=[3, 1],
        sequence_lengths=[2, 8],
        total_tokens=3,
        num_cache_slots=4,
    )

    assert [(span.start, span.end, span.state_slot) for span in spans] == [
        (0, 2, 3),
        (2, 3, 1),
    ]
    assert spans[0].query_length == 2
    assert spans[0].has_cached_prefix is False
    assert spans[1].query_length == 1
    assert spans[1].has_cached_prefix is True
    assert all(span.active for span in spans)


def test_plan_preserves_scheduler_reordering_by_state_slot() -> None:
    spans = plan_packed_state_spans(
        query_start_loc=[0, 1, 3, 4],
        state_slots=[2, 0, 1],
        sequence_lengths=[9, 2, 5],
        total_tokens=4,
        num_cache_slots=3,
    )

    assert [span.state_slot for span in spans] == [2, 0, 1]
    assert [span.request_index for span in spans] == [0, 1, 2]


def test_plan_accepts_vllm_padding_without_touching_a_cache_slot() -> None:
    spans = plan_packed_state_spans(
        query_start_loc=[0, 1, 1],
        state_slots=[0, PAD_STATE_SLOT],
        sequence_lengths=[1, 0],
        total_tokens=1,
        num_cache_slots=2,
    )

    assert spans[0].active is True
    assert spans[1].active is False
    assert spans[1].query_length == 0
    assert spans[1].has_cached_prefix is False


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {
                "query_start_loc": [1, 2],
                "state_slots": [0],
                "sequence_lengths": [1],
                "total_tokens": 2,
                "num_cache_slots": 1,
            },
            "start at zero",
        ),
        (
            {
                "query_start_loc": [0, 2, 1],
                "state_slots": [0, 1],
                "sequence_lengths": [2, 1],
                "total_tokens": 1,
                "num_cache_slots": 2,
            },
            "nondecreasing",
        ),
        (
            {
                "query_start_loc": [0, 1],
                "state_slots": [0],
                "sequence_lengths": [1],
                "total_tokens": 2,
                "num_cache_slots": 1,
            },
            "total_tokens",
        ),
        (
            {
                "query_start_loc": [0, 1, 2],
                "state_slots": [0],
                "sequence_lengths": [1, 1],
                "total_tokens": 2,
                "num_cache_slots": 2,
            },
            "request count",
        ),
        (
            {
                "query_start_loc": [0, 2],
                "state_slots": [0],
                "sequence_lengths": [1],
                "total_tokens": 2,
                "num_cache_slots": 1,
            },
            "shorter than query",
        ),
        (
            {
                "query_start_loc": [0, 1],
                "state_slots": [-2],
                "sequence_lengths": [1],
                "total_tokens": 1,
                "num_cache_slots": 1,
            },
            "state slot",
        ),
        (
            {
                "query_start_loc": [0, 1],
                "state_slots": [1],
                "sequence_lengths": [1],
                "total_tokens": 1,
                "num_cache_slots": 1,
            },
            "out of range",
        ),
        (
            {
                "query_start_loc": [0, 1, 2],
                "state_slots": [0, 0],
                "sequence_lengths": [1, 1],
                "total_tokens": 2,
                "num_cache_slots": 1,
            },
            "duplicate",
        ),
        (
            {
                "query_start_loc": [0, 0],
                "state_slots": [0],
                "sequence_lengths": [0],
                "total_tokens": 0,
                "num_cache_slots": 1,
            },
            "empty active",
        ),
    ],
)
def test_invalid_packed_metadata_fails_before_cache_mutation(
    arguments: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        plan_packed_state_spans(**arguments)
