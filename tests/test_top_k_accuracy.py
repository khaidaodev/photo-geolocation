"""
Tests for top_k_accuracy.py, using made-up scores and a fake model instead of a real trained
model or real photos.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import top_k_accuracy


class FixedScoresModel(nn.Module):
    """A stand-in model that returns the same made-up scores whatever photo it's given."""

    def __init__(self, scores):
        super().__init__()
        self.scores = scores

    def forward(self, x):
        return self.scores[: x.size(0)]


SCORES = torch.tensor([
    [0.1, 0.5, 0.4],    # best guess is country 1, second best is country 2
    [0.9, 0.06, 0.04],  # best guess is country 0, second best is country 1
])
LABELS = torch.tensor([2, 1])  # the real countries for those two photos


def test_top_k_correct_counts_only_when_real_country_is_in_the_top_k():
    assert top_k_accuracy.top_k_correct(SCORES, LABELS, k=1) == 0
    assert top_k_accuracy.top_k_correct(SCORES, LABELS, k=2) == 2


def test_top_k_accuracy_returns_a_fraction_for_each_k():
    photos = torch.rand(2, 3, 8, 8)
    loader = DataLoader(TensorDataset(photos, LABELS), batch_size=2)
    model = FixedScoresModel(SCORES)

    results = top_k_accuracy.top_k_accuracy(model, loader, ks=(1, 2))

    assert results == {1: 0.0, 2: 1.0}
