"""
Tests for the pure logic in src/data_loading.py, the parts that don't need the actual ~11GB
dataset downloaded. Downloading, and everything after that, needs a human with a real internet
connection and some patience, so that's left for a manual run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loading import STARTER_COUNTRIES, country_name, list_downloaded_countries  # noqa: E402


def test_country_name_converts_known_codes_to_readable_names():
    assert country_name("US") == "United States"
    assert country_name("GB") == "United Kingdom"
    assert country_name("JP") == "Japan"


def test_country_name_falls_back_to_the_code_for_unrecognised_input():
    assert country_name("ZZ") == "ZZ"


def test_starter_countries_are_all_real_recognised_codes():
    for code in STARTER_COUNTRIES:
        assert country_name(code) != code, f"{code} wasn't recognised as a real country code"


def test_starter_countries_has_no_duplicates():
    assert len(STARTER_COUNTRIES) == len(set(STARTER_COUNTRIES))


def test_list_downloaded_countries_returns_empty_for_a_missing_directory(tmp_path):
    assert list_downloaded_countries(tmp_path / "does_not_exist") == []


def test_list_downloaded_countries_only_lists_directories_not_stray_files(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    (split_dir / "US").mkdir()
    (split_dir / "GB").mkdir()
    (split_dir / "notes.txt").write_text("not a country folder")

    assert list_downloaded_countries(tmp_path) == ["GB", "US"]
