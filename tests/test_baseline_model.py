"""
Tests for baseline_model.py's load_split() filtering logic, using a tiny fake folder
structure instead of the real 11GB dataset.
"""

import sys
from pathlib import Path

from PIL import Image

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
