"""
Tests for clip_confusions.py, using small made-up grids instead of the real saved CLIP features
or a trained classifier.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_confusions
from confusion_matrix import build_confusion_matrix

NAMES = ["AA", "BB", "CC", "DD"]


def test_top_wrong_guesses_lists_the_most_common_first_and_skips_the_right_answer_and_zeros():
    matrix = [
        [5, 3, 0, 7],
        [0, 9, 0, 0],
        [0, 0, 9, 0],
        [0, 0, 0, 9],
    ]

    assert clip_confusions.top_wrong_guesses(matrix, 0, NAMES, n=3) == [("DD", 7), ("BB", 3)]


def test_top_wrong_guesses_stops_at_n():
    matrix = [[1, 4, 3, 2]] + [[0, 0, 0, 0]] * 3

    assert clip_confusions.top_wrong_guesses(matrix, 0, NAMES, n=2) == [("BB", 4), ("CC", 3)]


def test_top_wrong_guesses_keeps_tied_countries_in_the_order_given():
    matrix = [[1, 4, 4, 0]] + [[0, 0, 0, 0]] * 3

    assert clip_confusions.top_wrong_guesses(matrix, 0, NAMES, n=3) == [("BB", 4), ("CC", 4)]


def test_top_wrong_guesses_reads_rows_as_the_real_country_and_columns_as_the_guess():
    true = [0, 0, 0, 1]
    guessed = [1, 1, 0, 1]
    matrix = build_confusion_matrix(true, guessed, num_classes=2)

    assert clip_confusions.top_wrong_guesses(matrix, 0, ["AA", "BB"]) == [("BB", 2)]
    assert clip_confusions.top_wrong_guesses(matrix, 1, ["AA", "BB"]) == []
