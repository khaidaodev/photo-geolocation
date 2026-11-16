"""
Tests for continents.py, checking the table has no repeats and gets a few known countries right.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from continents import CONTINENT_OF, CONTINENTS, continent_of


def test_no_code_is_in_two_continents():
    total = sum(len(codes.split()) for codes in CONTINENTS.values())
    assert total == len(CONTINENT_OF)


def test_every_code_is_two_capital_letters():
    assert all(len(code) == 2 and code.isalpha() and code.isupper() for code in CONTINENT_OF)


def test_known_countries():
    assert continent_of("GB") == "Europe"
    assert continent_of("JP") == "Asia"
    assert continent_of("BR") == "South America"
    assert continent_of("AU") == "Oceania"


def test_namibia_is_not_confused_with_north_america():
    assert continent_of("NA") == "Africa"


def test_lowercase_works_and_unknown_code_raises():
    assert continent_of("fr") == "Europe"
    with pytest.raises(KeyError):
        continent_of("ZZ")
