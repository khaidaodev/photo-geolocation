"""
Tests for clip_zero_shot.py, using a fake pycountry and a fake CLIP text model instead of the
real ones.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_zero_shot


def _fake_pycountry(countries):
    """Stands in for the pycountry library, looking codes up in a small made-up dictionary."""
    return SimpleNamespace(countries=SimpleNamespace(get=lambda alpha_2: countries.get(alpha_2)))


def test_readable_name_uses_the_short_common_name_when_there_is_one(monkeypatch):
    fake = _fake_pycountry({"TW": SimpleNamespace(name="Taiwan, Province of China", common_name="Taiwan")})
    monkeypatch.setattr(clip_zero_shot, "pycountry", fake)
    assert clip_zero_shot.readable_name("TW") == "Taiwan"


def test_readable_name_falls_back_to_the_normal_name_then_to_the_code(monkeypatch):
    fake = _fake_pycountry({"KR": SimpleNamespace(name="Korea, Republic of")})
    monkeypatch.setattr(clip_zero_shot, "pycountry", fake)
    assert clip_zero_shot.readable_name("KR") == "Korea, Republic of"
    assert clip_zero_shot.readable_name("ZZ") == "ZZ"


def test_build_prompts_keeps_the_order_of_the_codes():
    prompts = clip_zero_shot.build_prompts(["JP", "ZZ"])
    assert prompts == ["a photo taken in Japan", "a photo taken in ZZ"]


def test_text_features_returns_unit_length_rows_in_the_order_given():
    def fake_tokenizer(sentences):
        return torch.tensor([[float(len(s)), 1.0] for s in sentences])

    class FakeTextModel:
        def encode_text(self, tokens):
            return tokens * 3.0

    features = clip_zero_shot.text_features(FakeTextModel(), fake_tokenizer, ["aa", "bbbb"])

    assert isinstance(features, np.ndarray)
    assert features.shape == (2, 2)
    assert np.allclose(np.linalg.norm(features, axis=1), 1.0)
    assert features[0][0] < features[1][0]
