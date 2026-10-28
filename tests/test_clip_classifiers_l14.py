"""
Tests for clip_classifiers_l14.py, using small made-up numbers instead of the real saved CLIP
features or a trained classifier.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_classifiers_l14


def test_class_centroids_gives_one_row_per_country_pointing_along_that_countrys_average():
    features = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
    labels = np.array([0, 0, 1])

    result = clip_classifiers_l14.class_centroids(features, labels, num_classes=2)

    assert np.allclose(result, [[1.0, 0.0], [0.0, 1.0]])


def test_class_centroids_rows_have_length_one_even_when_photos_point_in_different_directions():
    features = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 0])

    result = clip_classifiers_l14.class_centroids(features, labels, num_classes=1)

    assert np.allclose(result, [[2 ** -0.5, 2 ** -0.5]])
    assert np.isclose(np.linalg.norm(result[0]), 1.0)


def test_centroid_accuracy_is_the_fraction_of_photos_whose_closest_centroid_is_the_real_country():
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
    features = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    labels = np.array([0, 1, 1])

    assert np.isclose(clip_classifiers_l14.centroid_accuracy(centroids, features, labels), 2 / 3)


def test_centroid_accuracy_is_one_when_every_photo_sits_on_its_own_centroid():
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]])
    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    labels = np.array([0, 1, 0])

    assert clip_classifiers_l14.centroid_accuracy(centroids, features, labels) == 1.0
