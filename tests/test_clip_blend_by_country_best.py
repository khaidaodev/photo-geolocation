"""
Tests for clip_blend_by_country_best.py, using a handful of made-up predictions instead of the
real saved CLIP features or a trained classifier.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_blend_by_country_best

LABELS = [0, 0, 0, 0, 1, 1, 1, 1]
WITHOUT = [0, 0, 1, 1, 1, 1, 1, 0]
WITH = [0, 0, 0, 1, 1, 0, 0, 0]


def test_each_country_gets_a_row_with_its_change():
    rows = clip_blend_by_country_best.country_changes(LABELS, WITHOUT, WITH)
    by_country = {row[0]: row for row in rows}
    assert by_country[0] == (0, 4, 0.5, 0.75, 1)
    assert by_country[1] == (1, 4, 0.75, 0.25, -2)


def test_rows_run_from_biggest_gain_to_biggest_loss():
    rows = clip_blend_by_country_best.country_changes(LABELS, WITHOUT, WITH)
    assert [row[0] for row in rows] == [0, 1]


def test_no_change_gives_zero():
    rows = clip_blend_by_country_best.country_changes(LABELS, WITHOUT, WITHOUT)
    assert all(row[4] == 0 for row in rows)
