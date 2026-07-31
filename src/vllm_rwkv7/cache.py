"""Validated packed-request planning for vLLM recurrent state slots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

PAD_STATE_SLOT = -1


def _integer(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    resolved = int(value)
    if minimum is not None and resolved < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {resolved}")
    return resolved


@dataclass(frozen=True, slots=True)
class PackedStateSpan:
    """One request's token range and scheduler-owned recurrent state slot."""

    request_index: int
    start: int
    end: int
    state_slot: int
    sequence_length: int
    has_cached_prefix: bool

    @property
    def query_length(self) -> int:
        return self.end - self.start

    @property
    def active(self) -> bool:
        return self.state_slot != PAD_STATE_SLOT


def plan_packed_state_spans(
    *,
    query_start_loc: Sequence[int],
    state_slots: Sequence[int],
    sequence_lengths: Sequence[int],
    total_tokens: int,
    num_cache_slots: int,
) -> tuple[PackedStateSpan, ...]:
    """Validate vLLM linear-attention metadata before touching cache tensors.

    vLLM owns slot allocation, reordering, and release. This plan preserves the
    scheduler's request order while making slot reuse explicit: a request whose
    total sequence length equals its query length is fresh and must start from
    zero state even if the selected slot contains data from a released request.
    """

    total_tokens = _integer("total_tokens", total_tokens, minimum=0)
    num_cache_slots = _integer("num_cache_slots", num_cache_slots, minimum=0)
    starts = tuple(_integer("query_start_loc", value, minimum=0) for value in query_start_loc)
    slots = tuple(_integer("state slot", value) for value in state_slots)
    lengths = tuple(_integer("sequence length", value, minimum=0) for value in sequence_lengths)

    if not starts:
        raise ValueError("query_start_loc must contain at least the initial zero")
    if starts[0] != 0:
        raise ValueError("query_start_loc must start at zero")
    if any(end < start for start, end in zip(starts, starts[1:], strict=False)):
        raise ValueError("query_start_loc must be nondecreasing")
    if starts[-1] != total_tokens:
        raise ValueError(
            f"query_start_loc must end at total_tokens={total_tokens}, got {starts[-1]}"
        )

    request_count = len(starts) - 1
    if len(slots) != request_count or len(lengths) != request_count:
        raise ValueError(
            "packed metadata request count mismatch: "
            f"starts={request_count}, slots={len(slots)}, lengths={len(lengths)}"
        )

    spans = []
    used_slots: set[int] = set()
    for request_index, (start, end, state_slot, sequence_length) in enumerate(
        zip(starts, starts[1:], slots, lengths, strict=False)
    ):
        query_length = end - start
        if state_slot == PAD_STATE_SLOT:
            if query_length:
                raise ValueError("a padding state slot cannot own query tokens")
            spans.append(
                PackedStateSpan(
                    request_index=request_index,
                    start=start,
                    end=end,
                    state_slot=state_slot,
                    sequence_length=sequence_length,
                    has_cached_prefix=False,
                )
            )
            continue
        if state_slot < PAD_STATE_SLOT:
            raise ValueError(f"state slot must be {PAD_STATE_SLOT} or nonnegative")
        if state_slot >= num_cache_slots:
            raise ValueError(
                f"state slot {state_slot} is out of range for {num_cache_slots} cache slots"
            )
        if state_slot in used_slots:
            raise ValueError(f"duplicate active state slot {state_slot} in one packed batch")
        if query_length == 0:
            raise ValueError("empty active request span is not supported")
        if sequence_length < query_length:
            raise ValueError(
                f"sequence length {sequence_length} is shorter than query length {query_length}"
            )
        used_slots.add(state_slot)
        spans.append(
            PackedStateSpan(
                request_index=request_index,
                start=start,
                end=end,
                state_slot=state_slot,
                sequence_length=sequence_length,
                has_cached_prefix=sequence_length > query_length,
            )
        )

    return tuple(spans)
