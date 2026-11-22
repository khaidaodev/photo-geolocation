"""
Tests for check_balance.py, using made-up counts instead of the real dataset folders.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import check_balance


def test_even_counts_have_no_uneven_countries():
    assert check_balance.summarise_counts({"GB": 50, "FR": 50, "JP": 50}) == (3, 50, 50, [])


def test_a_short_country_is_named():
    assert check_balance.summarise_counts({"GB": 50, "FR": 50, "JP": 48}) == (3, 48, 50, ["JP"])


def test_a_single_country_is_fine():
    assert check_balance.summarise_counts({"GB": 10}) == (1, 10, 10, [])
