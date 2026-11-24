"""The status transition rules and the reason rules on their own (no database, no HTTP)."""
import itertools

import pytest

from app.models.case import CaseStatus
from app.services import review

ALL = [status.value for status in CaseStatus]

# Written out here on purpose (not read from the module), so the test is not a copy of the code under test.
ALLOWED = {
    ("pending", "approved"),
    ("pending", "rejected"),
    ("approved", "found"),
    ("approved", "closed"),
    ("found", "closed"),
    ("rejected", "closed"),
}
PAIRS = list(itertools.product(ALL, ALL))
PAIR_IDS = [f"{from_status}->{to_status}" for from_status, to_status in PAIRS]


def test_the_allowed_transitions_are_exactly_the_documented_ones():
    actual = {
        (from_status.value, to_status.value)
        for from_status, targets in review.ALLOWED_TRANSITIONS.items()
        for to_status in targets
    }
    assert actual == ALLOWED
    assert {status.value for status in review.ALLOWED_TRANSITIONS} == set(ALL)      # every status has an entry


def test_the_transition_table_cannot_be_changed_at_runtime():
    with pytest.raises(TypeError):
        review.ALLOWED_TRANSITIONS[CaseStatus.CLOSED] = frozenset({CaseStatus.PENDING})


@pytest.mark.parametrize("from_status, to_status", PAIRS, ids=PAIR_IDS)
def test_can_transition_agrees_with_the_documented_table_for_every_pair(from_status, to_status):
    assert review.can_transition(from_status, to_status) is ((from_status, to_status) in ALLOWED)


def test_check_transition_accepts_allowed_pairs_and_names_the_pair_it_refuses():
    for from_status, to_status in PAIRS:
        if (from_status, to_status) in ALLOWED:
            assert review.check_transition(from_status, to_status) is None
        else:
            with pytest.raises(review.InvalidTransitionError) as exc_info:
                review.check_transition(from_status, to_status)
            assert exc_info.value.from_status == from_status
            assert exc_info.value.to_status == to_status


def test_closed_is_terminal_and_staying_in_the_same_status_is_not_a_transition():
    assert review.ALLOWED_TRANSITIONS[CaseStatus.CLOSED] == frozenset()
    for status in ALL:
        assert review.can_transition("closed", status) is False
        assert review.can_transition(status, status) is False


def test_nothing_can_move_back_to_pending():
    assert not any(review.can_transition(status, "pending") for status in ALL)


# ------------------------------------------------------------------ reasons
def test_the_reason_constants_match_the_specification():
    assert review.REASON_REQUIRED_FOR == {CaseStatus.REJECTED, CaseStatus.CLOSED}
    assert review.REVIEW_DECISIONS == {CaseStatus.APPROVED, CaseStatus.REJECTED}
    assert (review.REASON_MIN_LENGTH, review.REASON_MAX_LENGTH) == (5, 1000)


@pytest.mark.parametrize("to_status", ["rejected", "closed"])
@pytest.mark.parametrize("missing", [None, "", "   ", "\t "])
def test_a_reason_is_required_for_rejected_and_closed(to_status, missing):
    with pytest.raises(review.ReasonRequiredError):
        review.normalise_reason(to_status, missing)


@pytest.mark.parametrize("to_status", ["approved", "found", "pending"])
@pytest.mark.parametrize("missing", [None, "", "   "])
def test_a_reason_is_optional_for_the_other_statuses(to_status, missing):
    assert review.normalise_reason(to_status, missing) is None


@pytest.mark.parametrize("to_status", ALL)
def test_the_length_limits_apply_after_trimming_for_every_status(to_status):
    assert review.normalise_reason(to_status, "  abcde  ") == "abcde"                    # 5 characters after trimming
    assert review.normalise_reason(to_status, "a" * 1000) == "a" * 1000
    assert review.normalise_reason(to_status, "  " + "a" * 1000 + "  ") == "a" * 1000     # padding does not count
    for too_short in ("abcd", "  abcd  ", "a"):
        with pytest.raises(review.InvalidReasonError):
            review.normalise_reason(to_status, too_short)
    with pytest.raises(review.InvalidReasonError):
        review.normalise_reason(to_status, "a" * 1001)
