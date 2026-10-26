"""Fixes the ViT-L-14 classifier's confidence. clip_confidence_l14.py showed it is under-confident:
when it says 65% it is right about 77% of the time. This divides the log-probabilities by one
number (the temperature, picked on the valid split only by lowest log loss) and renormalises.
A temperature under 1 sharpens the probabilities. The order of the countries for each photo never
changes, so accuracy must stay the same. Then the test split is read once to see if the gap
between what it says and how often it is right got smaller. Uses the saved photo numbers.

Run it with:
    python3 src/clip_calibrate_l14.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from clip_blend_l14 import log_softmax
from clip_confidence_l14 import confidence_bins
from clip_per_country import BEST_C
from clip_probe import COUNTRY211_DIR
from clip_probe_l14 import FEATURES_DIR_L14

TEMPERATURES = [round(0.3 + 0.05 * i, 2) for i in range(19)]


def rescale(log_probs, temperature):
    """Divides the log-probabilities by the temperature and renormalises each row."""
    return log_softmax(log_probs / temperature)


def log_loss(log_probs, labels):
    """Average of minus the log-probability given to the real country. Lower is better."""
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def calibration_gap(rows, total):
    """Average gap between what the classifier said and how often it was right, across the
    confidence bins from confidence_bins(), weighted by how many photos are in each bin."""
    return sum(count / total * abs(average - right) for _, _, count, average, right in rows)


def summarise(log_probs, labels):
    """Returns the top-1 accuracy, the confidence bins and the calibration gap."""
    probabilities = np.exp(log_probs)
    correct = probabilities.argmax(axis=1) == labels
    rows = confidence_bins(probabilities.max(axis=1), correct)
    return float(correct.mean()), rows, calibration_gap(rows, len(labels))

if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    folder = FEATURES_DIR_L14 / f"{len(codes)}_countries"
    train, valid, test = (np.load(folder / f"{split}.npz") for split in ("train", "valid", "test"))
    assert test["labels"].max() + 1 == len(codes), "Saved photo numbers and country folders don't match"

    print(f"Training the classifier on {len(train['labels'])} saved photo numbers (C={BEST_C})...")
    clf = LogisticRegression(C=BEST_C, max_iter=1000)
    clf.fit(train["features"], train["labels"])
    assert list(clf.classes_) == list(range(len(codes)))
    valid_log = np.log(clf.predict_proba(valid["features"]) + 1e-12)
    test_log = np.log(clf.predict_proba(test["features"]) + 1e-12)

    print("Temperature and valid log loss (lower is better, 1.0 is no change, picked on valid only):")
    best_t, best_loss = 1.0, float("inf")
    for t in TEMPERATURES:
        loss = log_loss(rescale(valid_log, t), valid["labels"])
        print(f"  {t:.2f}: {loss:.4f}")
        if loss < best_loss:
            best_t, best_loss = t, loss

    before_acc, _, before_gap = summarise(rescale(test_log, 1.0), test["labels"])
    after_acc, after_rows, after_gap = summarise(rescale(test_log, best_t), test["labels"])
    print(f"Best temperature was {best_t}. Test split, never used until now:")
    print(f"Top-1 accuracy before {before_acc:.1%}, after {after_acc:.1%} (should be the same, 40.4%)")
    print(f"Average gap between what it said and how often it was right: before {before_gap:.1%}, after {after_gap:.1%}")

    print("\nAfter rescaling, confidence of the top guess, how many test photos, average confidence, how often right:")
    for low, high, count, average, right in after_rows:
        print(f"  {low:.0%}-{high:.0%}: {count} photos, said {average:.1%}, was right {right:.1%}")
