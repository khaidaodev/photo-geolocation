"""
Tests for baseline_model.py: load_split()'s filtering logic, and the actual model-building and
feature-extraction functions, using tiny fake data instead of the real 11GB dataset.
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import baseline_model


def _make_fake_country211(root: Path):
    """Builds a tiny fake dataset: 3 countries, 2 photos each, same folder layout as the real
    one (split/country_code/photo.jpg) but tiny enough to run instantly."""
    for country in ["US", "GB", "ZZ"]:  # ZZ isn't in STARTER_COUNTRIES, should get filtered out
        folder = root / "train" / country
        folder.mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (10, 10)).save(folder / f"{i}.jpg")


def test_load_split_only_keeps_starter_countries(tmp_path, monkeypatch):
    _make_fake_country211(tmp_path)
    monkeypatch.setattr(baseline_model, "COUNTRY211_DIR", tmp_path)

    subset = baseline_model.load_split("train")
    dataset = subset.dataset

    kept_labels = {dataset.classes[dataset.samples[i][1]] for i in subset.indices}
    assert kept_labels == {"US", "GB"}
    assert "ZZ" not in kept_labels


def test_build_feature_extractor_outputs_512_numbers_per_photo():
    """A real photo goes in, and instead of a class guess, we should get back a plain list of
    512 numbers summarising it, since the final classification layer was swapped out."""
    model = baseline_model.build_feature_extractor()
    fake_photo_batch = torch.rand(1, 3, 224, 224)  # 1 fake photo, 3 colour channels, 224x224

    with torch.no_grad():
        output = model(fake_photo_batch)

    assert output.shape == (1, 512)


def test_extract_features_matches_labels_to_correct_photos():
    """Builds a tiny fake dataset directly (2 fake photos, 2 different labels) and checks the
    features and labels that come back are the same length and in the right order, not that
    the actual numbers mean anything specific."""
    model = baseline_model.build_feature_extractor()
    fake_photos = torch.rand(2, 3, 224, 224)
    fake_labels = torch.tensor([0, 1])
    fake_dataset = TensorDataset(fake_photos, fake_labels)

    features, labels = baseline_model.extract_features(fake_dataset, model, batch_size=2)

    assert features.shape == (2, 512)
    assert list(labels) == [0, 1]
