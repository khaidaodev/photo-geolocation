"""
Tests for clip_predict_best.py, using a tiny made-up classifier and made-up photo numbers instead
of the real CLIP models.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_blend_l14
import clip_predict_best


class FakeClassifier:
    """Stands in for the trained classifier, always giving the same probabilities."""

    def __init__(self, probabilities):
        self.probabilities = np.array(probabilities)

    def predict_proba(self, joined):
        return self.probabilities


def made_up_inputs():
    joined = np.zeros((2, 768 + 4))
    joined[0, 0] = 1.0
    joined[1, 1] = 1.0
    sentences = np.zeros((3, 768))
    sentences[0, 0] = 1.0
    sentences[1, 1] = 1.0
    sentences[2, 2] = 1.0
    clf = FakeClassifier([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    return clf, joined, sentences


def test_blended_log_probs_rows_turn_into_probabilities_that_add_up_to_one():
    clf, joined, sentences = made_up_inputs()

    result = np.exp(clip_predict_best.blended_log_probs(clf, joined, sentences))

    assert np.allclose(result.sum(axis=1), [1.0, 1.0])


def test_blended_log_probs_with_weight_zero_gives_the_classifier_alone():
    clf, joined, sentences = made_up_inputs()

    result = np.exp(clip_predict_best.blended_log_probs(clf, joined, sentences, weight=0.0))

    assert np.allclose(result, clf.probabilities, atol=1e-6)


def test_blended_log_probs_with_weight_one_gives_the_sentence_scores_alone():
    clf, joined, sentences = made_up_inputs()
    expected = np.exp(clip_blend_l14.log_softmax(clip_blend_l14.LOGIT_SCALE * (joined[:, :768] @ sentences.T)))

    result = np.exp(clip_predict_best.blended_log_probs(clf, joined, sentences, weight=1.0))

    assert np.allclose(result, expected)


def test_blended_log_probs_only_reads_the_openai_half_of_the_joined_numbers_for_the_sentences():
    clf, joined, sentences = made_up_inputs()
    changed = joined.copy()
    changed[:, 768:] = 5.0

    first = clip_predict_best.blended_log_probs(clf, joined, sentences, weight=1.0)
    second = clip_predict_best.blended_log_probs(clf, changed, sentences, weight=1.0)

    assert np.allclose(first, second)
