"""
Tests for confusion_matrix.py's matrix-building and mix-up-finding logic, using small,
made-up prediction lists instead of a real model or real photos.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import confusion_matrix


def test_build_confusion_matrix_counts_correctly():
    # 3 classes: 0, 1, 2. Model got the first two right, then guessed 1 when it should've
    # been 2, twice in a row.
    true_labels = [0, 1, 2, 2]
    predicted_labels = [0, 1, 1, 1]

    matrix = confusion_matrix.build_confusion_matrix(true_labels, predicted_labels, num_classes=3)

    assert matrix[0][0] == 1  # correctly guessed class 0 once
    assert matrix[1][1] == 1  # correctly guessed class 1 once
    assert matrix[2][1] == 2  # guessed 1 when it should've been 2, twice
    assert matrix[2][2] == 0  # never correctly guessed class 2


def test_find_most_confused_pairs_ignores_correct_guesses():
    class_names = ["US", "GB", "FR"]
    matrix = [
        [5, 1, 0],  # 5 correct US guesses, 1 US photo wrongly guessed as GB
        [0, 3, 2],  # 3 correct GB guesses, 2 GB photos wrongly guessed as FR
        [0, 0, 4],  # 4 correct FR guesses
    ]

    mix_ups = confusion_matrix.find_most_confused_pairs(matrix, class_names, top_n=5)

    assert ("GB", "FR", 2) in mix_ups
    assert ("US", "GB", 1) in mix_ups
    assert not any(actual == guessed for actual, guessed, _ in mix_ups)


def test_find_most_confused_pairs_sorts_most_frequent_first():
    class_names = ["US", "GB", "FR"]
    matrix = [
        [5, 1, 0],
        [0, 3, 4],
        [0, 0, 4],
    ]

    mix_ups = confusion_matrix.find_most_confused_pairs(matrix, class_names, top_n=1)

    assert mix_ups == [("GB", "FR", 4)]
