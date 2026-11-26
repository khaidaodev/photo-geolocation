"""
Tests for chance_baseline.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import chance_baseline


def test_one_guess_out_of_four_countries():
    assert chance_baseline.chance_at(1, 4) == 0.25


def test_more_guesses_raise_the_chance():
    gap = chance_baseline.chance_at(5, 211) - 5 * chance_baseline.chance_at(1, 211)
    assert abs(gap) < 1e-12


def test_guessing_every_country_is_certain():
    assert chance_baseline.chance_at(211, 211) == 1.0
