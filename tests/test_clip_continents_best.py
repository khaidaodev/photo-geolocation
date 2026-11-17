"""
Tests for clip_continents_best.py, using a handful of made-up predictions instead of the real
saved CLIP features or a trained classifier.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_continents_best


def test_scores_on_a_small_example():
    true = ["GB", "GB", "JP", "BR"]
    predicted = ["GB", "FR", "KR", "AR"]
    country, continent, wrong_share, _ = clip_continents_best.continent_scores(true, predicted)
    assert country == 0.25
    assert continent == 1.0
    assert wrong_share == 1.0


def test_wrong_continent_is_counted():
    country, continent, wrong_share, _ = clip_continents_best.continent_scores(
        ["GB", "JP"], ["GB", "BR"]
    )
    assert country == 0.5
    assert continent == 0.5
    assert wrong_share == 0.0


def test_per_continent_split():
    _, _, _, per_continent = clip_continents_best.continent_scores(
        ["GB", "FR", "JP"], ["GB", "DE", "BR"]
    )
    assert per_continent["Europe"] == (2, 0.5, 1.0)
    assert per_continent["Asia"] == (1, 0.0, 0.0)


def test_no_wrong_answers_gives_zero_share():
    _, _, wrong_share, _ = clip_continents_best.continent_scores(["GB"], ["GB"])
    assert wrong_share == 0.0
