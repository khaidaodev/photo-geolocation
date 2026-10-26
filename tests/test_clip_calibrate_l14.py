"""
Tests for clip_calibrate_l14.py, using small made-up numbers instead of the real saved CLIP
features or a trained classifier.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import clip_calibrate_l14


def test_rescale_with_a_temperature_of_one_leaves_log_probabilities_unchanged():
    log_probs = np.log(np.array([[0.2, 0.3, 0.5]]))

    assert np.allclose(clip_calibrate_l14.rescale(log_probs, 1.0), log_probs)


def test_rescale_with_a_temperature_under_one_sharpens_and_rows_still_add_up_to_one():
    log_probs = np.log(np.array([[0.2, 0.3, 0.5]]))

    result = np.exp(clip_calibrate_l14.rescale(log_probs, 0.5))

    assert np.allclose(result.sum(axis=1), [1.0])
    assert np.allclose(result, np.array([[0.04, 0.09, 0.25]]) / 0.38)


def test_rescale_never_changes_which_country_is_on_top_for_each_photo():
    probs = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.2, 0.35, 0.15, 0.3]])
    log_probs = np.log(probs)

    for temperature in (0.5, 0.9, 2.0):
        rescaled = clip_calibrate_l14.rescale(log_probs, temperature)
        assert (rescaled.argmax(axis=1) == log_probs.argmax(axis=1)).all()


def test_log_loss_is_minus_the_average_log_probability_given_to_the_real_country():
    log_probs = np.log(np.array([[0.5, 0.5], [0.25, 0.75]]))
    labels = np.array([0, 1])

    result = clip_calibrate_l14.log_loss(log_probs, labels)

    assert np.isclose(result, -(np.log(0.5) + np.log(0.75)) / 2)


def test_log_loss_is_zero_when_all_the_probability_is_on_the_real_country():
    log_probs = np.array([[0.0, -30.0], [-30.0, 0.0]])
    labels = np.array([0, 1])

    assert np.isclose(clip_calibrate_l14.log_loss(log_probs, labels), 0.0)


def test_calibration_gap_weights_each_bin_by_how_many_photos_are_in_it():
    rows = [(0.0, 0.5, 10, 0.4, 0.5), (0.5, 1.0, 30, 0.9, 0.8)]

    assert np.isclose(clip_calibrate_l14.calibration_gap(rows, 40), 0.1)


def test_calibration_gap_is_zero_when_calibrated_and_counts_over_and_under_confidence_the_same():
    calibrated = [(0.0, 1.0, 10, 0.6, 0.6)]
    too_modest = [(0.0, 1.0, 10, 0.5, 0.7)]
    too_sure = [(0.0, 1.0, 10, 0.7, 0.5)]

    assert np.isclose(clip_calibrate_l14.calibration_gap(calibrated, 10), 0.0)
    assert np.isclose(clip_calibrate_l14.calibration_gap(too_modest, 10), 0.2)
    assert np.isclose(clip_calibrate_l14.calibration_gap(too_sure, 10), 0.2)


def test_summarise_gives_accuracy_the_bins_and_the_gap_for_a_small_example():
    probs = np.array([[0.95, 0.05], [0.25, 0.75], [0.65, 0.35], [0.15, 0.85]])
    labels = np.array([0, 1, 1, 1])

    accuracy, rows, gap = clip_calibrate_l14.summarise(np.log(probs), labels)

    assert np.isclose(accuracy, 0.75)
    assert sum(row[2] for row in rows) == 4
    assert len(rows) == 4
    assert np.isclose(gap, (0.05 + 0.25 + 0.65 + 0.15) / 4)
