"""
Tests for clip_valid_vs_test_best.py, using made-up shares instead of the real saved CLIP
features or a trained classifier.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_valid_vs_test_best


def test_gap_is_test_minus_valid():
    rows = clip_valid_vs_test_best.score_gaps([(1, 0.40), (3, 0.60)], [(1, 0.50), (3, 0.55)])
    assert rows[0][:3] == (1, 0.40, 0.50)
    assert abs(rows[0][3] - 0.10) < 1e-12
    assert abs(rows[1][3] + 0.05) < 1e-12


def test_one_row_per_cutoff():
    rows = clip_valid_vs_test_best.score_gaps([(1, 0.4), (3, 0.6), (5, 0.7)], [(1, 0.4), (3, 0.6), (5, 0.7)])
    assert [row[0] for row in rows] == [1, 3, 5]


def test_equal_scores_have_no_gap():
    rows = clip_valid_vs_test_best.score_gaps([(1, 0.4)], [(1, 0.4)])
    assert rows[0][3] == 0.0
