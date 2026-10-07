"""
Tests for clip_probe.py, using a fake CLIP and made-up numbers instead of the real model.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_probe


class FakeClip:
    """Stands in for CLIP: turns each photo into 4 arbitrary numbers, not scaled to length 1."""

    def encode_image(self, images):
        return images.flatten(start_dim=1)[:, :4] * 3.0


def test_extract_clip_features_returns_unit_length_rows_and_matching_labels():
    photos = torch.rand(6, 3, 8, 8) + 0.1
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    loader = DataLoader(TensorDataset(photos, labels), batch_size=4)

    features, got_labels = clip_probe.extract_clip_features(FakeClip(), loader)

    assert features.shape == (6, 4)
    assert np.allclose(np.linalg.norm(features, axis=1), 1.0)
    assert list(got_labels) == [0, 1, 2, 0, 1, 2]


def test_cached_features_extracts_once_then_reuses_the_saved_file(tmp_path):
    calls = []

    def extract():
        calls.append(1)
        return np.ones((3, 2)), np.array([0, 1, 2])

    path = tmp_path / "nested" / "train.npz"
    first = clip_probe.cached_features(path, extract)
    second = clip_probe.cached_features(path, extract)

    assert len(calls) == 1
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_top_k_scores_counts_a_photo_when_the_real_country_is_in_the_top_k():
    probabilities = np.array([[0.1, 0.5, 0.4], [0.9, 0.06, 0.04]])
    labels = np.array([2, 1])

    results = clip_probe.top_k_scores(probabilities, labels, ks=(1, 2))

    assert results == {1: 0.0, 2: 1.0}
