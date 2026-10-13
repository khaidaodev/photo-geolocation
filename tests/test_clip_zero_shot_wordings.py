"""
Tests for clip_zero_shot_wordings.py, using small made-up numbers instead of the real CLIP.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_zero_shot
import clip_zero_shot_wordings as wordings


def test_sentences_for_fills_every_country_in_the_order_given():
    assert wordings.sentences_for("a photo of {}", ["JP", "ZZ"]) == ["a photo of Japan", "a photo of ZZ"]


def test_ensemble_averages_the_wordings_and_rescales_each_row_to_length_one():
    first = np.array([[1.0, 0.0], [0.0, 1.0]])
    second = np.array([[0.0, 1.0], [0.0, 1.0]])

    result = wordings.ensemble([first, second])

    assert np.allclose(result, [[0.7071, 0.7071], [0.0, 1.0]], atol=1e-3)
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_pick_best_returns_the_highest_scoring_wording():
    assert wordings.pick_best({"a": 0.1, "b": 0.3, "c": 0.2}) == "b"


def test_pick_best_gives_a_tie_to_the_one_listed_first():
    assert wordings.pick_best({"a": 0.3, "b": 0.3}) == "a"


def test_the_original_wording_stays_first_so_the_comparison_is_against_the_earlier_result():
    assert wordings.TEMPLATES[0] == clip_zero_shot.PROMPT
