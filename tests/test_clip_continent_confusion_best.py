"""
Tests for clip_continent_confusion_best.py, using a handful of made-up predictions instead of
the real saved CLIP features or a trained classifier.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_continent_confusion_best

NAMES = ["Europe", "Asia", "Africa"]


def test_rows_show_where_photos_were_sent():
    grid = clip_continent_confusion_best.continent_confusion(
        ["GB", "GB", "GB", "GB", "JP"], ["GB", "FR", "JP", "EG", "JP"], NAMES
    )
    assert grid[0].tolist() == [0.5, 0.25, 0.25]
    assert grid[1].tolist() == [0.0, 1.0, 0.0]


def test_a_continent_with_no_photos_is_a_row_of_zeros():
    grid = clip_continent_confusion_best.continent_confusion(["GB"], ["GB"], NAMES)
    assert grid[2].tolist() == [0.0, 0.0, 0.0]


def test_each_row_with_photos_adds_to_one():
    grid = clip_continent_confusion_best.continent_confusion(
        ["GB", "JP", "EG", "GB"], ["JP", "EG", "GB", "FR"], NAMES
    )
    assert np.allclose(grid.sum(axis=1), 1.0)
