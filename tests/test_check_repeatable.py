"""
Tests for check_repeatable.py, using a few made-up score rows instead of the real saved CLIP
features or a trained classifier.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import check_repeatable

SCORES = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]])


def test_identical_runs_agree_completely():
    assert check_repeatable.compare_runs(SCORES, SCORES.copy()) == (1.0, 0.0)


def test_a_flipped_answer_is_counted():
    other = SCORES.copy()
    other[1] = [0.6, 0.1, 0.3]
    same, _ = check_repeatable.compare_runs(SCORES, other)
    assert same == 0.5


def test_biggest_difference_is_reported():
    other = SCORES + np.array([[0.0, 0.01, 0.0], [0.0, 0.0, 0.0]])
    _, biggest = check_repeatable.compare_runs(SCORES, other)
    assert abs(biggest - 0.01) < 1e-12
