"""
Tests for clip_joined_l14_all.py, using small made-up numbers instead of the real saved CLIP
features or a trained classifier.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_joined_l14_all


def test_pick_columns_with_one_part_keeps_only_that_part():
    X = np.arange(12).reshape(2, 6)

    result = clip_joined_l14_all.pick_columns(X, [slice(2, 4)])

    assert np.array_equal(result, [[2, 3], [8, 9]])


def test_pick_columns_with_two_parts_sticks_them_together():
    X = np.arange(12).reshape(2, 6)

    result = clip_joined_l14_all.pick_columns(X, [slice(0, 2), slice(4, 6)])

    assert np.array_equal(result, [[0, 1, 4, 5], [6, 7, 10, 11]])


def test_pick_columns_keeps_the_parts_in_the_order_given():
    X = np.arange(12).reshape(2, 6)

    result = clip_joined_l14_all.pick_columns(X, [slice(4, 6), slice(0, 2)])

    assert np.array_equal(result, [[4, 5, 0, 1], [10, 11, 6, 7]])
