"""
Tests for clip_coverage_best.py, using small made-up confidences and right and wrong lists
instead of the real saved CLIP features or a trained classifier.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_coverage_best


def test_accuracy_at_coverage_keeps_the_most_confident_photos_first():
    confidence = [0.9, 0.5, 0.7, 0.6]
    correct = [True, False, True, False]

    (coverage, count, cutoff, accuracy), = clip_coverage_best.accuracy_at_coverage(confidence, correct, [0.5])

    assert coverage == 0.5
    assert count == 2
    assert np.isclose(cutoff, 0.7)
    assert accuracy == 1.0


def test_accuracy_at_coverage_of_everything_is_the_overall_accuracy():
    confidence = [0.9, 0.5, 0.7, 0.6]
    correct = [True, False, True, False]

    (coverage, count, cutoff, accuracy), = clip_coverage_best.accuracy_at_coverage(confidence, correct, [1.0])

    assert count == 4
    assert np.isclose(cutoff, 0.5)
    assert accuracy == 0.5


def test_accuracy_at_coverage_keeps_at_least_one_photo_for_a_tiny_share():
    (coverage, count, cutoff, accuracy), = clip_coverage_best.accuracy_at_coverage([0.4, 0.8], [False, True], [0.01])

    assert count == 1
    assert np.isclose(cutoff, 0.8)
    assert accuracy == 1.0


def test_accuracy_at_coverage_gives_one_row_per_share_in_the_order_given():
    rows = clip_coverage_best.accuracy_at_coverage([0.9, 0.8, 0.7, 0.6], [True, True, False, False], [0.25, 1.0, 0.5])

    assert [row[0] for row in rows] == [0.25, 1.0, 0.5]
