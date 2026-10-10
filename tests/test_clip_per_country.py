"""
Tests for clip_per_country.py, using small made-up numbers instead of the real saved CLIP
features or a trained classifier.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_per_country


def test_per_country_accuracy_gives_each_country_its_own_fraction():
    true = np.array([0, 0, 1, 1, 1, 2])
    predicted = np.array([0, 1, 1, 1, 0, 2])

    result = clip_per_country.per_country_accuracy(true, predicted, num_classes=3)

    assert np.allclose(result, [0.5, 2 / 3, 1.0])


def test_per_country_accuracy_gives_zero_when_nothing_was_right_or_there_were_no_photos():
    true = np.array([0, 0, 1, 1])
    predicted = np.array([1, 1, 1, 1])

    result = clip_per_country.per_country_accuracy(true, predicted, num_classes=3)

    assert np.allclose(result, [0.0, 1.0, 0.0])


def test_best_and_worst_return_the_top_and_bottom_countries():
    best, worst = clip_per_country.best_and_worst([0.9, 0.1, 0.5, 0.0], ["AA", "BB", "CC", "DD"], n=2)

    assert best == [("AA", 0.9), ("CC", 0.5)]
    assert worst == [("DD", 0.0), ("BB", 0.1)]


def test_best_and_worst_keep_tied_countries_in_the_order_given():
    best, worst = clip_per_country.best_and_worst([0.5, 0.5, 0.5], ["AA", "BB", "CC"], n=2)

    assert best == [("AA", 0.5), ("BB", 0.5)]
    assert worst == [("AA", 0.5), ("BB", 0.5)]


def test_summarise_counts_the_middle_and_the_low_and_high_scorers():
    result = clip_per_country.summarise([0.0, 0.05, 0.1, 0.5, 0.9])

    assert result == {
        "median": 0.1,
        "at_zero": 1,
        "below_10_percent": 2,
        "at_least_50_percent": 2,
    }
