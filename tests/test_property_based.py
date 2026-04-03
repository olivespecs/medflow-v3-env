"""
Property-based tests using Hypothesis.

Targets boundary conditions and crash guarantees that unit tests miss:
  - _age_group: every integer age maps to exactly one bucket
  - _age_group: boundaries (39 → "18-40", 40 → "41-60", etc.)
  - normalize_date: never crashes regardless of input

Install test deps:  pip install -e ".[test]"
Run with: pytest tests/test_property_based.py -v
"""

from __future__ import annotations

from datetime import datetime

import pytest

# Skip the entire module if hypothesis is not installed.
# Install with: pip install -e ".[test]"  or  pip install hypothesis
hypothesis = pytest.importorskip("hypothesis", minversion="6.0.0", reason="hypothesis not installed")
given = hypothesis.given
settings = hypothesis.settings
st = hypothesis.strategies

CURRENT_YEAR = datetime.now().year  # dynamic so tests stay correct each year


# ---------------------------------------------------------------------------
# _age_group — bucket coverage and boundaries
# ---------------------------------------------------------------------------


@given(st.integers(min_value=18, max_value=100))
@settings(max_examples=200)
def test_age_group_covers_all_ages(age: int) -> None:
    """Every integer age 18-100 must map to exactly one bucket with no gaps."""
    from src.record_processors import _age_group

    dob = f"{CURRENT_YEAR - age}-06-15"  # mid-year avoids most leap-year edge cases
    group = _age_group(dob)
    assert group in {"18-40", "41-60", "61-75", "76+"}, (
        f"Unexpected bucket {group!r} for age={age}, dob={dob}"
    )


def test_age_group_boundary_39_vs_40() -> None:
    """Boundary: age 39 → '18-40', age 40 → '41-60' (code uses strict <)."""
    from src.record_processors import _age_group

    # Use January dates to ensure birthday has passed (current date is March)
    dob_39 = f"{CURRENT_YEAR - 39}-01-15"
    dob_40 = f"{CURRENT_YEAR - 40}-01-15"

    assert _age_group(dob_39) == "18-40", f"Age 39 should be '18-40', got {_age_group(dob_39)}"
    assert _age_group(dob_40) == "41-60", f"Age 40 should be '41-60', got {_age_group(dob_40)}"


def test_age_group_boundary_59_vs_60() -> None:
    """Boundary: age 59 → '41-60', age 60 → '61-75'."""
    from src.record_processors import _age_group

    # Use January dates to ensure birthday has passed
    dob_59 = f"{CURRENT_YEAR - 59}-01-15"
    dob_60 = f"{CURRENT_YEAR - 60}-01-15"

    assert _age_group(dob_59) == "41-60"
    assert _age_group(dob_60) == "61-75"


def test_age_group_boundary_74_vs_75() -> None:
    """Boundary: age 74 → '61-75', age 75 → '76+'."""
    from src.record_processors import _age_group

    # Use January dates to ensure birthday has passed
    dob_74 = f"{CURRENT_YEAR - 74}-01-15"
    dob_75 = f"{CURRENT_YEAR - 75}-01-15"

    assert _age_group(dob_74) == "61-75"
    assert _age_group(dob_75) == "76+"


def test_age_group_none_returns_unknown() -> None:
    """None/empty DOB must not crash — returns 'unknown'."""
    from src.record_processors import _age_group

    assert _age_group(None) == "unknown"
    assert _age_group("") == "unknown"
    assert _age_group("not-a-date") == "unknown"


# ---------------------------------------------------------------------------
# normalize_date — crash guarantee
# ---------------------------------------------------------------------------


@given(st.text(max_size=200))
@settings(max_examples=500)
def test_normalize_date_never_crashes(s: str) -> None:
    """normalize_date must never raise — unknown formats return None."""
    from src.utils import normalize_date

    result = normalize_date(s)
    # Function returns ISO date string or None for unparseable/empty input
    assert result is None or isinstance(result, str), f"normalize_date({s!r}) returned unexpected type: {result!r}"


@given(st.none())
def test_normalize_date_handles_none_gracefully(s: None) -> None:
    """normalize_date called with None must not crash."""
    from src.utils import normalize_date

    # None input should return None (not crash)
    result = normalize_date(s)  # type: ignore[arg-type]
    assert result is None, f"normalize_date(None) should return None, got {result!r}"

