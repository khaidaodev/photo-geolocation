"""Tries a finer set of C values for the ViT-L-14 classifier, around 10 which won in
clip_probe_l14.py (that grid was only 1, 10 and 100). C controls how closely the classifier is
allowed to fit the training photos. The best C is picked on the valid split only, then the test
split is read once at the end. Uses the saved photo numbers, so nothing gets extracted again.

Run it with:
    python3 src/clip_probe_l14_c_grid.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from clip_probe import COUNTRY211_DIR, top_k_scores
from clip_probe_l14 import FEATURES_DIR_L14

C_GRID = [3.0, 5.0, 10.0, 20.0, 30.0]

if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    folder = FEATURES_DIR_L14 / f"{len(codes)}_countries"
    train, valid, test = (np.load(folder / f"{split}.npz") for split in ("train", "valid", "test"))
    assert test["labels"].max() + 1 == len(codes), "Saved photo numbers and country folders don't match"

    best_c, best_valid, best_clf = None, -1.0, None
    for c in C_GRID:
        print(f"Training the classifier with C={c}...")
        clf = LogisticRegression(C=c, max_iter=1000)
        clf.fit(train["features"], train["labels"])
        valid_acc = clf.score(valid["features"], valid["labels"])
        print(f"  C={c}: valid acc {valid_acc:.1%}")
        if valid_acc > best_valid:
            best_c, best_valid, best_clf = c, valid_acc, clf

    results = top_k_scores(best_clf.predict_proba(test["features"]), test["labels"])
    print(f"Best C was {best_c} (valid acc {best_valid:.1%}). Test split, never used until now:")
    for k, accuracy in results.items():
        print(f"Right country in the top {k} guesses: {accuracy:.1%}")
