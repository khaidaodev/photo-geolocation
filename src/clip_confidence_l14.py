"""Checks whether the ViT-L-14 classifier's confidence can be trusted. For every test photo it
takes the probability of the top guess, sorts the photos into ten bins by that number, and prints
how often the top guess was actually right in each bin. If it says 90% and is right about 90% of
the time, the percentages mean what they say. Uses the saved photo numbers, so nothing gets
extracted again.

Run it with:
    python3 src/clip_confidence_l14.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from clip_per_country import BEST_C
from clip_probe import COUNTRY211_DIR
from clip_probe_l14 import FEATURES_DIR_L14


def confidence_bins(confidence, correct, num_bins=10):
    """Splits photos into equal-width bins by confidence (0-10%, 10-20%, ...) and returns, for
    each bin, (low edge, high edge, photos in it, average confidence, fraction right). Empty bins
    are left out."""
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    which = np.minimum(np.digitize(confidence, edges[1:-1]), num_bins - 1)
    rows = []
    for b in range(num_bins):
        mask = which == b
        if mask.any():
            rows.append((edges[b], edges[b + 1], int(mask.sum()), float(confidence[mask].mean()), float(correct[mask].mean())))
    return rows


if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    folder = FEATURES_DIR_L14 / f"{len(codes)}_countries"
    train, test = np.load(folder / "train.npz"), np.load(folder / "test.npz")
    assert test["labels"].max() + 1 == len(codes), "Saved photo numbers and country folders don't match"

    print(f"Training the classifier on {len(train['labels'])} saved photo numbers (C={BEST_C})...")
    clf = LogisticRegression(C=BEST_C, max_iter=1000)
    clf.fit(train["features"], train["labels"])
    probabilities = clf.predict_proba(test["features"])
    predicted = probabilities.argmax(axis=1)
    correct = predicted == test["labels"]
    confidence = probabilities.max(axis=1)
    print(f"Overall test accuracy: {correct.mean():.1%} (clip_probe_l14.py got 40.4%, so these should match)")

    print("\nConfidence of the top guess, how many test photos, average confidence, how often right:")
    for low, high, count, average, right in confidence_bins(confidence, correct):
        print(f"  {low:.0%}-{high:.0%}: {count} photos, said {average:.1%}, was right {right:.1%}")
