"""
Tests for clip_confidence_l14.py, using small made-up numbers instead of the real saved CLIP
features or a trained classifier.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_confidence_l14


def test_confidence_bins_groups_photos_by_confidence_and_reports_count_average_and_fraction_right():
    confidence = np.array([0.05, 0.15, 0.95, 0.97])
    correct = np.array([False, True, True, True])

    rows = clip_confidence_l14.confidence_bins(confidence, correct)

    assert [row[2] for row in rows] == [1, 1, 2]
    low, high, count, average, right = rows[2]
    assert (round(low, 2), round(high, 2)) == (0.9, 1.0)
    assert count == 2
    assert np.isclose(average, 0.96)
    assert right == 1.0
    assert rows[0][4] == 0.0


def test_confidence_bins_leaves_empty_bins_out_and_puts_a_confidence_of_exactly_one_in_the_last_bin():
    confidence = np.array([0.45, 1.0])
    correct = np.array([True, False])

    rows = clip_confidence_l14.confidence_bins(confidence, correct)

    assert len(rows) == 2
    assert round(rows[0][0], 2) == 0.4
    assert round(rows[1][0], 2) == 0.9
    assert rows[1][2] == 1


def test_confidence_bins_can_use_a_different_number_of_bins():
    confidence = np.array([0.2, 0.8])
    correct = np.array([True, False])

    rows = clip_confidence_l14.confidence_bins(confidence, correct, num_bins=2)

    assert [(low, high) for low, high, _, _, _ in rows] == [(0.0, 0.5), (0.5, 1.0)]
    assert [row[4] for row in rows] == [1.0, 0.0]
