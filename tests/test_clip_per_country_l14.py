"""
Tests for clip_per_country_l14.py, using tiny made-up saved photo numbers written to a temporary
folder instead of the real CLIP features.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_per_country_l14


def write_made_up_features(folder):
    """Two well separated countries. The last test photo is deliberately labelled wrong."""
    train_features = np.array([[1.0, 0.0], [0.9, 0.1], [1.0, 0.1], [0.95, 0.0], [1.0, 0.05],
                               [0.0, 1.0], [0.1, 0.9], [0.1, 1.0], [0.0, 0.95], [0.05, 1.0]])
    train_labels = np.array([0] * 5 + [1] * 5)
    test_features = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    test_labels = np.array([0, 0, 1, 0])
    np.savez(folder / "train.npz", features=train_features, labels=train_labels)
    np.savez(folder / "test.npz", features=test_features, labels=test_labels)


def test_train_and_score_gives_the_overall_accuracy_and_one_score_per_country(tmp_path):
    write_made_up_features(tmp_path)

    overall, per_country = clip_per_country_l14.train_and_score(tmp_path, num_classes=2)

    assert np.isclose(overall, 0.75)
    assert np.allclose(per_country, [2 / 3, 1.0])


def test_train_and_score_stops_when_the_number_of_countries_does_not_match_the_saved_labels(tmp_path):
    write_made_up_features(tmp_path)

    with pytest.raises(AssertionError):
        clip_per_country_l14.train_and_score(tmp_path, num_classes=3)
