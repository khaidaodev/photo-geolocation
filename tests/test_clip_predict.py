"""
Tests for clip_predict.py, using a tiny made-up classifier and a fake CLIP instead of the real
saved classifier or the real model.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_predict


def _tiny_classifier():
    photos = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = np.array([0, 0, 1, 1])
    return LogisticRegression().fit(photos, labels), photos


def test_save_then_load_gives_back_a_classifier_that_makes_the_same_guesses(tmp_path):
    clf, photos = _tiny_classifier()
    path = tmp_path / "nested" / "classifier.joblib"

    clip_predict.save_classifier(clf, ["AA", "BB"], path)
    loaded, codes = clip_predict.load_classifier(path)

    assert codes == ["AA", "BB"]
    assert np.array_equal(loaded.predict(photos), clf.predict(photos))


def test_load_classifier_refuses_when_the_codes_do_not_match_the_classifier(tmp_path):
    clf, _ = _tiny_classifier()
    path = tmp_path / "classifier.joblib"

    clip_predict.save_classifier(clf, ["AA", "BB", "CC"], path)

    with pytest.raises(ValueError):
        clip_predict.load_classifier(path)


def test_top_guesses_returns_the_most_likely_countries_first():
    result = clip_predict.top_guesses(np.array([0.1, 0.6, 0.3]), ["AA", "BB", "CC"], k=2)

    assert [code for code, _ in result] == ["BB", "CC"]
    assert result[0][1] == pytest.approx(0.6)


def test_format_guess_shows_the_full_country_name_and_the_code():
    assert clip_predict.format_guess("JP", 0.256) == "Japan (JP): 25.6%"


def test_photo_features_gives_one_row_scaled_to_length_one(tmp_path):
    class FakeClip:
        def encode_image(self, images):
            return images.flatten(start_dim=1)[:, :4] * 3.0

    def fake_preprocess(image):
        return torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0 + 0.1

    photo_path = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), (120, 60, 30)).save(photo_path)

    features = clip_predict.photo_features(FakeClip(), fake_preprocess, photo_path)

    assert features.shape == (1, 4)
    assert np.allclose(np.linalg.norm(features, axis=1), 1.0)
