"""
Tests for clip_blend_l14.py, using small made-up numbers instead of the real saved CLIP
features or a trained classifier.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_blend_l14


def test_log_softmax_rows_turn_into_probabilities_that_add_up_to_one():
    scores = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])

    result = clip_blend_l14.log_softmax(scores)

    assert np.allclose(np.exp(result).sum(axis=1), [1.0, 1.0])
    assert np.allclose(np.exp(result[1]), [1 / 3, 1 / 3, 1 / 3])


def test_log_softmax_is_not_changed_by_adding_the_same_number_to_a_row_and_does_not_overflow():
    scores = np.array([[1.0, 2.0, 3.0]])

    assert np.allclose(clip_blend_l14.log_softmax(scores + 1000.0), clip_blend_l14.log_softmax(scores))
    assert np.isfinite(clip_blend_l14.log_softmax(np.array([[1000.0, 1001.0]]))).all()


def test_blend_mixes_the_two_sets_of_scores_by_the_weight():
    classifier_part = np.array([[1.0, 2.0]])
    zero_shot_part = np.array([[3.0, 6.0]])

    assert np.allclose(clip_blend_l14.blend(classifier_part, zero_shot_part, 0.0), classifier_part)
    assert np.allclose(clip_blend_l14.blend(classifier_part, zero_shot_part, 1.0), zero_shot_part)
    assert np.allclose(clip_blend_l14.blend(classifier_part, zero_shot_part, 0.25), [[1.5, 3.0]])


def test_blend_weight_decides_which_country_wins_when_the_two_disagree():
    classifier_part = np.array([[-1.0, -1.5]])
    zero_shot_part = np.array([[-3.0, -0.5]])

    assert clip_blend_l14.blend(classifier_part, zero_shot_part, 0.0).argmax() == 0
    assert clip_blend_l14.blend(classifier_part, zero_shot_part, 1.0).argmax() == 1
