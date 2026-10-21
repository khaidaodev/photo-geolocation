"""Checks whether the bigger ViT-L-14 model still mixes up the same countries as ViT-B-32.

Trains both classifiers again from the saved photo numbers (C=10, the value that won on the
valid split) and prints each overall score so I can check it against what the earlier scripts
printed. It takes the 15 most common mix-ups for ViT-B-32 and shows how many times ViT-L-14
made the same mistake. It also lists the 15 most common mix-ups for ViT-L-14 and where the
photos of the low scoring countries went. Each country only has 100 test photos, so a count of
3 or 4 is weak evidence and a count of 20 is not.

Run it with:
    python3 src/clip_confusions_l14.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from clip_confusions import CHECK_CODES, top_wrong_guesses
from clip_per_country import BEST_C
from clip_probe import COUNTRY211_DIR, FEATURES_DIR
from clip_probe_l14 import FEATURES_DIR_L14
from clip_zero_shot import readable_name
from confusion_matrix import build_confusion_matrix, find_most_confused_pairs


def train_and_predict(folder, num_classes):
    """Trains the classifier on the saved train photo numbers in folder, then returns the true
    labels and the guessed labels for the test photos."""
    train, test = np.load(folder / "train.npz"), np.load(folder / "test.npz")
    X_train, y_train = train["features"], train["labels"]
    X_test, y_test = test["features"], test["labels"]
    assert y_test.max() + 1 == num_classes, "Saved photo numbers and country folders don't match"
    print(f"  Training on {len(y_train)} saved photo numbers (C={BEST_C})...")
    clf = LogisticRegression(C=BEST_C, max_iter=1000)
    clf.fit(X_train, y_train)
    return y_test, clf.predict(X_test)


if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    names = [readable_name(code) for code in codes]
    folder_name = f"{len(codes)}_countries"

    print("ViT-B-32:")
    y_b32, guessed_b32 = train_and_predict(FEATURES_DIR / folder_name, len(codes))
    print(f"  Overall test accuracy: {(guessed_b32 == y_b32).mean():.1%} (should be 25.9%)")
    print("ViT-L-14:")
    y_l14, guessed_l14 = train_and_predict(FEATURES_DIR_L14 / folder_name, len(codes))
    print(f"  Overall test accuracy: {(guessed_l14 == y_l14).mean():.1%} (should be 40.4%)")

    matrix_b32 = build_confusion_matrix(y_b32.tolist(), guessed_b32.tolist(), len(codes))
    matrix_l14 = build_confusion_matrix(y_l14.tolist(), guessed_l14.tolist(), len(codes))

    print("\nThe 15 most common ViT-B-32 mix-ups, and how many times ViT-L-14 made the same one:")
    dropped = 0
    for actual, guessed, count in find_most_confused_pairs(matrix_b32, names, top_n=15):
        new = matrix_l14[names.index(actual)][names.index(guessed)]
        dropped += int(new < count)
        print(f"  Real {actual}, guessed {guessed}: {count} -> {new}")
    print(f"ViT-L-14 made fewer of these mistakes in {dropped} of the 15 pairs.")

    print("\nMost common ViT-L-14 mix-ups across all countries:")
    for actual, guessed, count in find_most_confused_pairs(matrix_l14, names, top_n=15):
        print(f"  Real {actual}, guessed {guessed}: {count} times")

    print("\nWhere the photos of the low scoring countries went with ViT-L-14:")
    for code in CHECK_CODES:
        index = codes.index(code)
        right = matrix_l14[index][index]
        wrong = top_wrong_guesses(matrix_l14, index, names)
        guesses = ", ".join(f"{name} ({count})" for name, count in wrong)
        print(f"  {names[index]}: {right} of {sum(matrix_l14[index])} right, wrong ones mostly went to {guesses}")
