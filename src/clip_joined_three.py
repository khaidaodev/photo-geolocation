"""Sticks together the saved photo numbers from three CLIP sets: ViT-B-16 (512 per photo), the
OpenAI ViT-L-14 (768) and the DFN-2B ViT-L-14 (768), so each photo has 2048 numbers. Then it
trains the same logistic regression on that. C is picked on the valid split only, then the test
split is read once at the end. A check trains on the two ViT-L-14 sets alone, which should match
clip_joined_l14.py. Nothing gets extracted again.

Run it with:
    python3 src/clip_joined_three.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from clip_joined_l14 import join_features
from clip_per_country import BEST_C
from clip_probe import C_VALUES, COUNTRY211_DIR, top_k_scores
from clip_probe_b16 import FEATURES_DIR_B16
from clip_probe_l14 import FEATURES_DIR_L14
from clip_probe_l14_dfn2b import FEATURES_DIR_L14_DFN2B

if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    folder_name = f"{len(codes)}_countries"
    data = {}
    for split in ("train", "valid", "test"):
        first = np.load(FEATURES_DIR_B16 / folder_name / f"{split}.npz")
        second = np.load(FEATURES_DIR_L14 / folder_name / f"{split}.npz")
        third = np.load(FEATURES_DIR_L14_DFN2B / folder_name / f"{split}.npz")
        pair = {"features": join_features(first, second), "labels": first["labels"]}
        data[split] = (join_features(pair, third), first["labels"])
    X_train, y_train = data["train"]
    X_valid, y_valid = data["valid"]
    X_test, y_test = data["test"]
    assert y_test.max() + 1 == len(codes), "Saved photo numbers and country folders don't match"
    print(f"Each photo now has {X_train.shape[1]} numbers.")

    clf = LogisticRegression(C=BEST_C, max_iter=1000)
    clf.fit(X_train[:, 512:], y_train)
    print(f"Check, the two ViT-L-14 sets only: test top-1 {clf.score(X_test[:, 512:], y_test):.1%} (should be 42.3%)")

    best_c, best_valid, best_clf = None, -1.0, None
    for c in C_VALUES:
        print(f"Training the classifier on all three sets with C={c}...")
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
