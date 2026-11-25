"""
Tests for check_features_again.py, using a few made-up feature rows instead of the real saved
CLIP features.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import check_features_again

SAVED = np.array([[1.0, 0.0], [0.0, 1.0]])


def test_identical_features_have_no_difference():
    biggest, lowest = check_features_again.compare_features(SAVED, SAVED.copy())
    assert biggest == 0.0
    assert abs(lowest - 1.0) < 1e-12


def test_a_small_change_shows_up_in_both_numbers():
    new = np.array([[1.0, 0.1], [0.0, 1.0]])
    biggest, lowest = check_features_again.compare_features(new, SAVED)
    assert abs(biggest - 0.1) < 1e-12
    assert lowest < 1.0


def test_a_flipped_photo_gives_a_low_similarity():
    new = np.array([[0.0, 1.0], [0.0, 1.0]])
    _, lowest = check_features_again.compare_features(new, SAVED)
    assert lowest == 0.0
