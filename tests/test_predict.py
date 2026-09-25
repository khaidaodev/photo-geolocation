"""
Tests for predict.py: turning raw model scores into ranked, readable predictions, and
correctly rebuilding a saved model from a checkpoint. Uses fake models and fake checkpoints,
no real training or downloads needed.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import predict


class FakeModel(nn.Module):
    """A stand-in model that always returns the same fixed scores regardless of the actual
    photo given to it, so predict_top_k's own logic can be tested on its own, fast and
    predictably, without needing a real trained model."""

    def __init__(self, fixed_logits):
        super().__init__()
        self.fixed_logits = fixed_logits

    def forward(self, x):
        return self.fixed_logits.unsqueeze(0)


def test_predict_top_k_returns_highest_confidence_first(tmp_path):
    fake_logits = torch.tensor([1.0, 5.0, 2.0, 0.5])  # index 1 should clearly win
    model = FakeModel(fake_logits)
    class_names = ["US", "GB", "FR", "JP"]

    fake_image_path = tmp_path / "fake.jpg"
    Image.new("RGB", (50, 50)).save(fake_image_path)

    results = predict.predict_top_k(model, class_names, fake_image_path, k=2)

    assert results[0][0] == "GB"
    assert results[0][1] > results[1][1]


def test_predict_top_k_confidences_are_even_when_scores_are_even(tmp_path):
    fake_logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
    model = FakeModel(fake_logits)
    class_names = ["US", "GB", "FR", "JP"]

    fake_image_path = tmp_path / "fake.jpg"
    Image.new("RGB", (50, 50)).save(fake_image_path)

    results = predict.predict_top_k(model, class_names, fake_image_path, k=4)

    for _, confidence in results:
        assert abs(confidence - 0.25) < 0.01


def test_load_trained_model_rebuilds_correct_architecture(tmp_path):
    dummy_model = torchvision.models.resnet50(weights=None)
    dummy_model.fc = torch.nn.Linear(dummy_model.fc.in_features, 3)
    checkpoint_path = tmp_path / "dummy_model.pt"
    torch.save({
        "model_state_dict": dummy_model.state_dict(),
        "class_names": ["US", "GB", "FR"],
        "valid_accuracy": 0.5,
    }, checkpoint_path)

    model, class_names = predict.load_trained_model(checkpoint_path)

    assert class_names == ["US", "GB", "FR"]
    assert model.fc.out_features == 3
