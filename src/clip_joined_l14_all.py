"""Joins the saved ViT-L-14 photo numbers from three sets of weights: OpenAI (768 per photo),
DFN-2B (768) and LAION-2B (768). It tries two joins: OpenAI with LAION-2B, and all three together
(2304 numbers per photo). Same logistic regression as before. C is picked on the valid split
only, then the test split is read once for each join. Two checks should match earlier scripts:
OpenAI with DFN-2B (clip_joined_l14.py) and LAION-2B alone. Nothing gets extracted again.

Run it with:
    python3 src/clip_joined_l14_all.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from clip_joined_l14 import join_features
from clip_per_country import BEST_C
from clip_probe import C_VALUES, COUNTRY211_DIR, top_k_scores
from clip_probe_l14 import FEATURES_DIR_L14
from clip_probe_l14_dfn2b import FEATURES_DIR_L14_DFN2B
from clip_probe_l14_laion2b import FEATURES_DIR_L14_LAION2B

OPENAI, DFN2B, LAION2B = slice(0, 768), slice(768, 1536), slice(1536, 2304)


def pick_columns(X, columns):
    """Keeps only the chosen sets of columns from the joined photo numbers, in the order given."""
    return np.concatenate([X[:, part] for part in columns], axis=1)


def fit_and_report(name, columns, data):
    """Trains on the chosen columns for each C, picks C on valid only, then prints the test
    scores once."""
    X_train, y_train = data["train"]
    X_valid, y_valid = data["valid"]
    X_test, y_test = data["test"]
    best_c, best_valid, best_clf = None, -1.0, None
    print(f"\n{name}:")
    for c in C_VALUES:
        clf = LogisticRegression(C=c, max_iter=1000)
        clf.fit(pick_columns(X_train, columns), y_train)
        valid_acc = clf.score(pick_columns(X_valid, columns), y_valid)
        print(f"  C={c}: valid acc {valid_acc:.1%}")
        if valid_acc > best_valid:
            best_c, best_valid, best_clf = c, valid_acc, clf
    assert list(best_clf.classes_) == list(range(len(np.unique(y_test))))
    results = top_k_scores(best_clf.predict_proba(pick_columns(X_test, columns)), y_test)
    print(f"  Best C was {best_c} (valid acc {best_valid:.1%}). Test split, never used until now:")
    for k, accuracy in results.items():
        print(f"  Right country in the top {k} guesses: {accuracy:.1%}")


if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    folder_name = f"{len(codes)}_countries"
    data = {}
    for split in ("train", "valid", "test"):
        first = np.load(FEATURES_DIR_L14 / folder_name / f"{split}.npz")
        second = np.load(FEATURES_DIR_L14_DFN2B / folder_name / f"{split}.npz")
        third = np.load(FEATURES_DIR_L14_LAION2B / folder_name / f"{split}.npz")
        pair = {"features": join_features(first, second), "labels": first["labels"]}
        data[split] = (join_features(pair, third), first["labels"])
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]
    assert y_test.max() + 1 == len(codes), "Saved photo numbers and country folders don't match"
    print(f"Each photo now has {X_train.shape[1]} numbers.")

    for name, columns, expected in (("OpenAI with DFN-2B", [OPENAI, DFN2B], "42.3%"),
                                    ("LAION-2B alone", [LAION2B], "35.2%")):
        clf = LogisticRegression(C=BEST_C, max_iter=1000)
        clf.fit(pick_columns(X_train, columns), y_train)
        print(f"Check, {name}: test top-1 {clf.score(pick_columns(X_test, columns), y_test):.1%} (should be {expected})")

    fit_and_report("OpenAI with LAION-2B", [OPENAI, LAION2B], data)
    fit_and_report("All three ViT-L-14 sets", [OPENAI, DFN2B, LAION2B], data)
