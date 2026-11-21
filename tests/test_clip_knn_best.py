"""
Tests for clip_knn_best.py, using a few made-up points instead of the real saved CLIP features.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_knn_best

TRAIN_X = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
TRAIN_Y = np.array([0, 0, 1, 1])


def test_a_query_votes_for_the_country_it_is_closest_to():
    votes = clip_knn_best.knn_votes(TRAIN_X, TRAIN_Y, np.array([[1.0, 0.1], [0.1, 1.0]]), 2, 2)
    assert votes.argmax(axis=1).tolist() == [0, 1]


def test_k_of_one_only_counts_the_single_closest_photo():
    votes = clip_knn_best.knn_votes(TRAIN_X, TRAIN_Y, np.array([[1.0, 0.0]]), 1, 2)
    assert votes[0, 1] == 0.0
    assert votes[0, 0] > 0.0


def test_one_row_per_query_and_one_column_per_country():
    votes = clip_knn_best.knn_votes(TRAIN_X, TRAIN_Y, np.array([[1.0, 0.0]] * 3), 2, 5, chunk_size=2)
    assert votes.shape == (3, 5)
