"""
Tests for clip_blend_by_continent_best.py, using a handful of made-up predictions instead of
the real saved CLIP features or a trained classifier.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_blend_by_continent_best


def test_each_setup_is_scored_inside_each_continent():
    true = ["GB", "FR", "JP", "KR"]
    predictions = {
        "a": ["GB", "FR", "JP", "CN"],
        "b": ["DE", "FR", "KR", "CN"],
    }
    table = clip_blend_by_continent_best.accuracy_by_continent(true, predictions)
    assert table["Europe"] == (2, {"a": 1.0, "b": 0.5})
    assert table["Asia"] == (2, {"a": 0.5, "b": 0.0})


def test_only_continents_that_appear_are_listed():
    table = clip_blend_by_continent_best.accuracy_by_continent(["GB"], {"a": ["GB"]})
    assert list(table) == ["Europe"]
