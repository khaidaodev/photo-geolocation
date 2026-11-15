"""
Tests for clip_confident_mistakes_best.py, using small made-up confidences and right and wrong
lists instead of the real saved CLIP features or a trained classifier.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_confident_mistakes_best


def test_most_confident_wrong_gives_the_wrong_guesses_with_the_highest_confidence_first():
    confidence = [0.9, 0.8, 0.95, 0.6, 0.7]
    correct = [True, False, False, False, True]

    picked = clip_confident_mistakes_best.most_confident_wrong(confidence, correct, 2)

    assert picked.tolist() == [2, 1]


def test_most_confident_wrong_ignores_right_guesses_even_when_they_are_the_most_confident():
    picked = clip_confident_mistakes_best.most_confident_wrong([0.99, 0.5], [True, False], 5)

    assert picked.tolist() == [1]


def test_most_confident_wrong_is_empty_when_every_guess_was_right():
    picked = clip_confident_mistakes_best.most_confident_wrong([0.9, 0.5], [True, True], 3)

    assert picked.tolist() == []


def test_wrong_at_least_counts_the_wrong_guesses_at_or_above_the_level():
    confidence = [0.95, 0.9, 0.85, 0.4]
    correct = [True, False, False, False]

    assert clip_confident_mistakes_best.wrong_at_least(confidence, correct, 0.9) == (1, 2)
    assert clip_confident_mistakes_best.wrong_at_least(confidence, correct, 0.5) == (2, 3)


def test_wrong_at_least_is_zero_of_zero_when_no_photo_reaches_the_level():
    assert clip_confident_mistakes_best.wrong_at_least([0.2, 0.3], [False, True], 0.9) == (0, 0)
