"""
Tests for clip_training_size_l14.py, using small made-up labels instead of the real saved CLIP
features or a trained classifier.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_training_size_l14


def test_subsample_per_country_takes_the_same_share_from_each_country():
    labels = np.array([0] * 10 + [1] * 20)

    picked = clip_training_size_l14.subsample_per_country(labels, 0.5)

    assert np.array_equal(np.bincount(labels[picked]), [5, 10])


def test_subsample_per_country_gives_sorted_positions_with_no_repeats_and_the_same_pick_each_time():
    labels = np.array([0] * 10 + [1] * 20)

    first = clip_training_size_l14.subsample_per_country(labels, 0.5)
    second = clip_training_size_l14.subsample_per_country(labels, 0.5)

    assert np.array_equal(first, np.sort(first))
    assert len(set(first.tolist())) == len(first)
    assert np.array_equal(first, second)


def test_subsample_per_country_keeps_at_least_one_photo_per_country_for_a_tiny_share():
    labels = np.array([0, 0, 0, 1, 1, 1])

    picked = clip_training_size_l14.subsample_per_country(labels, 0.1)

    assert np.array_equal(np.bincount(labels[picked]), [1, 1])


def test_subsample_per_country_with_a_share_of_one_keeps_every_photo():
    labels = np.array([0, 1, 1, 2, 2, 2])

    picked = clip_training_size_l14.subsample_per_country(labels, 1.0)

    assert np.array_equal(picked, np.arange(6))
