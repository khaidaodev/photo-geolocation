"""Sticks the saved ViT-L-14 photo numbers from the OpenAI weights (768 per photo) and from the
DFN-2B weights (768 per photo) side by side, so each photo has 1536 numbers, then trains the same
logistic regression on that. The two models may be wrong on different photos, so together they
might beat either one alone. C is picked on the valid split only, then the test split is read
once at the end. Two checks train on each half alone, and should match the earlier scripts.
Nothing gets extracted again.

Run it with:
    python3 src/clip_joined_l14.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from clip_per_country import BEST_C
from clip_probe import C_VALUES, COUNTRY211_DIR, top_k_scores
from clip_probe_l14_dfn2b import FEATURES_DIR_L14_DFN2B
from clip_probe_l14 import FEATURES_DIR_L14


def join_features(first, second):
    """Sticks two sets of saved photo numbers side by side. Stops if the photos are not in the
    same order in both, since the numbers would then belong to different photos."""
    assert np.array_equal(first["labels"], second["labels"]), "The two sets list the photos in a different order"
    return np.concatenate([first["features"], second["features"]], axis=1)


if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    folder_name = f"{len(codes)}_countries"
    data = {}
    for split in ("train", "valid", "test"):
        first = np.load(FEATURES_DIR_L14 / folder_name / f"{split}.npz")
        second = np.load(FEATURES_DIR_L14_DFN2B / folder_name / f"{split}.npz")
        data[split] = (join_features(first, second), first["labels"])
    X_train, y_train = data["train"]
    X_valid, y_valid = data["valid"]
    X_test, y_test = data["test"]
    assert y_test.max() + 1 == len(codes), "Saved photo numbers and country folders don't match"
    print(f"Each photo now has {X_train.shape[1]} numbers.")

    for name, columns, expected in (("OpenAI half only", slice(0, 768), "40.4%"),
                                    ("DFN-2B half only", slice(768, 1536), "36.7%")):
        clf = LogisticRegression(C=BEST_C, max_iter=1000)
        clf.fit(X_train[:, columns], y_train)
        print(f"Check, {name}: test top-1 {clf.score(X_test[:, columns], y_test):.1%} (should be {expected})")

    best_c, best_valid, best_clf = None, -1.0, None
    for c in C_VALUES:
        print(f"Training the classifier on both halves with C={c}...")
        clf = LogisticRegression(C=c, max_iter=1000)
        clf.fit(X_train, y_train)
        valid_acc = clf.score(X_valid, y_valid)
        print(f"  C={c}: valid acc {valid_acc:.1%}")
        if valid_acc > best_valid:
            best_c, best_valid, best_clf = c, valid_acc, clf

    assert list(best_clf.classes_) == list(range(len(codes)))
    results = top_k_scores(best_clf.predict_proba(X_test), y_test)
    print(f"Best C was {best_c} (valid acc {best_valid:.1%}). Test split, never used until now:")
    for k, accuracy in results.items():
        print(f"Right country in the top {k} guesses: {accuracy:.1%}")
