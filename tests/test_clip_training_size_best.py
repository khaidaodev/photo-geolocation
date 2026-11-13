"""
Tests for clip_training_size_best.py, using small made-up scores instead of the real saved CLIP
features or trained classifiers.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_training_size_best


def test_spread_gives_the_average_lowest_and_highest_score():
    average, low, high = clip_training_size_best.spread([0.30, 0.34, 0.32])

    assert np.isclose(average, 0.32)
    assert low == 0.30
    assert high == 0.34


def test_spread_of_a_single_score_has_the_same_average_lowest_and_highest():
    assert clip_training_size_best.spread([0.4]) == (0.4, 0.4, 0.4)
