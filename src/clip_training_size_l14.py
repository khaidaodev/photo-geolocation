"""Checks how the ViT-L-14 classifier's accuracy changes with the number of training photos.
Trains it again on 10%, 25%, 50% and 100% of the training photos (C=10, same as before), taking
the same share from every country so none gets left out. Each share is one random pick with a
fixed seed, so the small ones can wobble a little. Nothing is chosen from these numbers, so both
valid and test are just printed. Uses the saved photo numbers, so nothing gets extracted again.

Run it with:
    python3 src/clip_training_size_l14.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from clip_per_country import BEST_C
from clip_probe import COUNTRY211_DIR
from clip_probe_l14 import FEATURES_DIR_L14

FRACTIONS = [0.1, 0.25, 0.5, 1.0]
SEED = 0


def subsample_per_country(labels, fraction, seed=SEED):
    """Returns the positions of a random share of the photos, taking the same share from each
    country (at least one photo per country). Positions come back sorted."""
    rng = np.random.default_rng(seed)
    picked = []
    for country in np.unique(labels):
        positions = np.where(labels == country)[0]
        count = max(1, round(len(positions) * fraction))
        picked.extend(rng.choice(positions, size=count, replace=False))
    return np.sort(np.array(picked))


if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    folder = FEATURES_DIR_L14 / f"{len(codes)}_countries"
    train, valid, test = (np.load(folder / f"{split}.npz") for split in ("train", "valid", "test"))
    assert test["labels"].max() + 1 == len(codes), "Saved photo numbers and country folders don't match"

    print("Share of the training photos, photos used, photos per country, valid top-1, test top-1:")
    for fraction in FRACTIONS:
        keep = subsample_per_country(train["labels"], fraction)
        clf = LogisticRegression(C=BEST_C, max_iter=1000)
        clf.fit(train["features"][keep], train["labels"][keep])
        valid_acc = clf.score(valid["features"], valid["labels"])
        test_acc = clf.score(test["features"], test["labels"])
        print(f"  {fraction:.0%}: {len(keep)} photos, about {len(keep) / len(codes):.0f} per country, "
              f"valid {valid_acc:.1%}, test {test_acc:.1%}")
