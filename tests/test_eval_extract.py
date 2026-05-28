r"""Tests for `open_geofm.eval.extract_answer`.

The extractor is Phase 8's regex fallback for MathVista-GPS-style answers
(MCQ letter A-E or a clean number). It is also the documented gate against
having to pay for `gpt-4o` as a judge — getting this right is load-bearing
for the **$0 eval path** that the wiki advertises.
"""

from __future__ import annotations

import pytest

from open_geofm.eval.extract_answer import extract

# ---------------------------------------------------------------------------
# MCQ letter rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A", "A"),
        ("B", "B"),
        ("(C)", "C"),
        ("answer: D", "D"),
        ("Answer: E", "E"),
        ("The answer is A.", "A"),
        ("final = B", "B"),
        ("option (D)", "D"),
        ("the answer is c", "C"),   # case-insensitive
    ],
)
def test_extract_recognises_mcq_letters(text: str, expected: str) -> None:
    assert extract(text) == expected


def test_extract_mcq_letter_takes_priority_over_numbers() -> None:
    """`(D)` must beat the trailing `4` so we don't confuse MCQA + reasoning."""
    text = "4 + 5 = 9, but the answer choice is (D)"
    assert extract(text) == "D"


def test_extract_last_line_wins_over_reasoning_trail() -> None:
    """A reasoning trace mentions A and B; the conclusion on the last line is C."""
    text = "Option A says X, option B says Y, but they're wrong.\nanswer: C"
    assert extract(text) == "C"


def test_extract_letter_F_is_not_matched() -> None:
    """MathVista-GPS only ranges A-E; F is a real noun (e.g. 'F = ma'), not an option."""
    assert extract("The answer is F.") != "F"


# ---------------------------------------------------------------------------
# Numeric fallback (no MCQ letter present)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is 5", "5"),
        ("y = 15", "15"),
        ("So, x = 1.5", "1.5"),
        ("we get -3", "-3"),
        ("…and finally 42.", "42"),
    ],
)
def test_extract_returns_clean_numbers(text: str, expected: str) -> None:
    assert extract(text) == expected


def test_extract_picks_last_number_when_many_present() -> None:
    text = "step 1: 4 + 5 = 9; step 2: 9 - 7 = 2; therefore the answer is 2."
    assert extract(text) == "2"


def test_extract_handles_negative_decimal() -> None:
    assert extract("the slope is -0.5") == "-0.5"


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_extract_returns_none_on_empty_string() -> None:
    assert extract("") is None


def test_extract_returns_none_on_whitespace_only() -> None:
    assert extract("   \n\t  ") is None


def test_extract_returns_none_on_pure_words() -> None:
    """A solution that is qualitative (no MCQ letter, no number) yields None."""
    assert extract("the triangle is equilateral, no value to report") is None


# ---------------------------------------------------------------------------
# Behaviour the wiki documents as the "last-line rule"
# ---------------------------------------------------------------------------


def test_extract_last_line_scan_finds_mcq_after_long_reasoning() -> None:
    text = (
        "Step 1: examine triangle ABC.\n"
        "Step 2: apply Pythagoras.\n"
        "Step 3: rule out option A and option B.\n"
        "\n"
        "answer: D"
    )
    assert extract(text) == "D"


def test_extract_returns_letter_even_when_not_on_last_line() -> None:
    """Letter regex falls back to scanning the full text if the last line
    is non-MCQ."""
    text = "The answer is C.\nThis concludes the proof."
    assert extract(text) == "C"
