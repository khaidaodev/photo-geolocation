"""
Tests for clip_joined_l14.py, using small made-up numbers instead of the real saved CLIP
features.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_joined_l14


def test_join_features_sticks_the_two_sets_of_numbers_side_by_side():
    first = {"features": np.array([[1.0, 2.0], [3.0, 4.0]]), "labels": np.array([0, 1])}
    second = {"features": np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]), "labels": np.array([0, 1])}

    result = clip_joined_l14.join_features(first, second)

    assert result.shape == (2, 5)
    assert np.array_equal(result, [[1, 2, 5, 6, 7], [3, 4, 8, 9, 10]])


def test_join_features_stops_when_the_photos_are_in_a_different_order():
    first = {"features": np.zeros((2, 2)), "labels": np.array([0, 1])}
    second = {"features": np.zeros((2, 2)), "labels": np.array([1, 0])}

    with pytest.raises(AssertionError):
        clip_joined_l14.join_features(first, second)


def test_join_features_can_be_chained_to_join_a_third_set():
    first = {"features": np.array([[1.0, 2.0]]), "labels": np.array([0])}
    second = {"features": np.array([[3.0]]), "labels": np.array([0])}
    third = {"features": np.array([[4.0, 5.0]]), "labels": np.array([0])}

    pair = {"features": clip_joined_l14.join_features(first, second), "labels": first["labels"]}
    result = clip_joined_l14.join_features(pair, third)

    assert np.array_equal(result, [[1, 2, 3, 4, 5]])
