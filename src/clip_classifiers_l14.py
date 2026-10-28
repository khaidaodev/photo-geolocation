"""Compares other simple classifiers on the same saved ViT-L-14 photo numbers as the logistic
regression in clip_probe_l14.py. Nearest class centroid averages the training photos of each
country and guesses the country whose average sits closest to the photo. k-nearest-neighbours
finds the k most similar training photos and lets them vote. The k is picked on the valid split
only. Valid and test are printed for all three, and nothing else is chosen from them. Nothing
gets extracted again.

Run it with:
    python3 src/clip_classifiers_l14.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from clip_per_country import BEST_C
from clip_probe import COUNTRY211_DIR
from clip_probe_l14 import FEATURES_DIR_L14

K_GRID = [1, 5, 10, 20, 50]


def class_centroids(features, labels, num_classes):
    """Returns one row per country: the average of that country's photo numbers, scaled so each
    row has length 1 (the same scaling the photo numbers already have)."""
    sums = np.zeros((num_classes, features.shape[1]))
    np.add.at(sums, labels, features)
    return sums / np.linalg.norm(sums, axis=1, keepdims=True)


def centroid_accuracy(centroids, features, labels):
    """Fraction of photos whose closest country average is the real country."""
    return float(((features @ centroids.T).argmax(axis=1) == labels).mean())


if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    folder = FEATURES_DIR_L14 / f"{len(codes)}_countries"
    train, valid, test = (np.load(folder / f"{split}.npz") for split in ("train", "valid", "test"))
    assert test["labels"].max() + 1 == len(codes), "Saved photo numbers and country folders don't match"

    print(f"Logistic regression (C={BEST_C}), the one used so far:")
    clf = LogisticRegression(C=BEST_C, max_iter=1000)
    clf.fit(train["features"], train["labels"])
    print(f"  valid {clf.score(valid['features'], valid['labels']):.1%}, "
          f"test {clf.score(test['features'], test['labels']):.1%} (should be 41.1% and 40.4%)")

    print("Nearest class centroid:")
    centroids = class_centroids(train["features"], train["labels"], len(codes))
    print(f"  valid {centroid_accuracy(centroids, valid['features'], valid['labels']):.1%}, "
          f"test {centroid_accuracy(centroids, test['features'], test['labels']):.1%}")

    print("k-nearest-neighbours, k picked on valid only:")
    best_k, best_valid, best_knn = None, -1.0, None
    for k in K_GRID:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
        knn.fit(train["features"], train["labels"])
        valid_acc = knn.score(valid["features"], valid["labels"])
        print(f"  k={k}: valid {valid_acc:.1%}")
        if valid_acc > best_valid:
            best_k, best_valid, best_knn = k, valid_acc, knn
    print(f"Best k was {best_k}. Test split: {best_knn.score(test['features'], test['labels']):.1%}")
