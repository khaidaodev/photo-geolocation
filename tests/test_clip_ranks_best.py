"""
Tests for clip_ranks_best.py, using a few made-up score rows instead of the real saved CLIP
features or a trained classifier.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_ranks_best

SCORES = np.array([
    [0.7, 0.2, 0.1],
    [0.2, 0.5, 0.3],
    [0.1, 0.3, 0.6],
])


def test_rank_is_one_when_the_true_country_scores_highest():
    assert clip_ranks_best.true_country_ranks(SCORES, [0, 1, 2]).tolist() == [1, 1, 1]


def test_rank_counts_the_countries_scoring_higher():
    assert clip_ranks_best.true_country_ranks(SCORES, [2, 0, 0]).tolist() == [3, 3, 3]


def test_share_within_each_cutoff():
    shares = clip_ranks_best.share_within([1, 2, 3, 3], cutoffs=[1, 2, 3])
    assert shares == [(1, 0.25), (2, 0.5), (3, 1.0)]
