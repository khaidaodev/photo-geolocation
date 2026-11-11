"""
Tests for clip_bootstrap_best.py, using small made-up right and wrong lists instead of the real
saved CLIP features or trained classifiers.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_bootstrap_best


def test_bootstrap_accuracies_gives_one_score_per_resample_for_each_setup():
    correct = np.array([True, False, True, True])

    first, second = clip_bootstrap_best.bootstrap_accuracies([correct, correct], n_resamples=50)

    assert first.shape == (50,)
    assert second.shape == (50,)


def test_bootstrap_accuracies_is_one_when_every_photo_is_right_and_zero_when_every_photo_is_wrong():
    right = np.ones(20, dtype=bool)
    wrong = np.zeros(20, dtype=bool)

    all_right, all_wrong = clip_bootstrap_best.bootstrap_accuracies([right, wrong], n_resamples=10)

    assert np.all(all_right == 1.0)
    assert np.all(all_wrong == 0.0)


def test_bootstrap_accuracies_picks_the_same_photos_for_every_setup():
    right_on = np.array([True, False, True, False, True, True, False, False])

    first, second = clip_bootstrap_best.bootstrap_accuracies([right_on, ~right_on], n_resamples=100)

    assert np.allclose(first + second, 1.0)


def test_bootstrap_accuracies_gives_the_same_scores_for_the_same_seed():
    correct = np.array([True, False, True, True, False, False])

    first = clip_bootstrap_best.bootstrap_accuracies([correct], n_resamples=30, seed=3)[0]
    second = clip_bootstrap_best.bootstrap_accuracies([correct], n_resamples=30, seed=3)[0]

    assert np.array_equal(first, second)


def test_middle_range_cuts_off_the_lowest_and_highest_share_of_the_values():
    values = np.arange(101)

    low, high = clip_bootstrap_best.middle_range(values, level=0.9)

    assert np.isclose(low, 5.0)
    assert np.isclose(high, 95.0)
