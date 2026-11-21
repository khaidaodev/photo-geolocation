"""
Tries a different kind of classifier on the same joined features, to see whether the logistic
regression was a fair choice. For each photo it finds the k most similar training photos and
lets them vote for a country, with closer photos counting for more. The best k is picked on the
valid split, and the test split is only scored once at the end.

Run it with:
    python3 src/clip_knn_best.py
"""

import numpy as np

from clip_joined_l14 import join_features
from clip_probe import COUNTRY211_DIR, top_k_scores
from clip_probe_l14 import FEATURES_DIR_L14
from clip_probe_l14_dfn2b import FEATURES_DIR_L14_DFN2B

K_VALUES = [1, 5, 10, 20, 50]


def knn_votes(train_x, train_y, query_x, k, n_classes, chunk_size=500):
    """Gives back a grid with one row per query photo and one column per country, holding the
    summed similarity of that country's photos among the k most similar training photos."""
    train_x = train_x / np.linalg.norm(train_x, axis=1, keepdims=True)
    query_x = query_x / np.linalg.norm(query_x, axis=1, keepdims=True)
    votes = np.zeros((len(query_x), n_classes))
    for start in range(0, len(query_x), chunk_size):
        similarity = query_x[start:start + chunk_size] @ train_x.T
        nearest = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        for row, neighbours in enumerate(nearest):
            np.add.at(votes[start + row], train_y[neighbours], similarity[row, neighbours])
    return votes


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

    print("Picking k on the valid split:")
    best_k, best_score = None, -1.0
    for k in K_VALUES:
        votes = knn_votes(X_train, y_train, X_valid, k, len(codes))
        score = float((votes.argmax(axis=1) == y_valid).mean())
        print(f"  k={k}: valid top-1 {score:.1%}")
        if score > best_score:
            best_k, best_score = k, score

    print(f"\nScoring the test split once, with k={best_k}:")
    votes = knn_votes(X_train, y_train, X_test, best_k, len(codes))
    scores = top_k_scores(votes, y_test)
    print(f"  Test top-1 {scores[1]:.1%}, top-3 {scores[3]:.1%}, top-5 {scores[5]:.1%}")
