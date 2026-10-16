"""
Shows where CLIP's wrong guesses go, to check whether low scoring countries get mixed up with
look-alike neighbours or with something less obvious.

The classifier is trained again from the photo numbers clip_probe.py saved, with the same C, so
the overall accuracy should match clip_probe.py. It then reuses the confusion matrix code from
the ResNet scripts: a grid where row A, column B counts the test photos of country A that were
guessed as country B. Each country only has 100 test photos, so a count of 3 or 4 is weak
evidence and a count of 20 is not.

Run it with:
    python3 src/clip_confusions.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from clip_per_country import BEST_C
from clip_probe import COUNTRY211_DIR, FEATURES_DIR
from clip_zero_shot import readable_name
from confusion_matrix import build_confusion_matrix, find_most_confused_pairs

# Countries that scored lowest in clip_per_country.py
CHECK_CODES = ["AT", "DE", "PL", "RO", "BG", "CL", "CO"]


def top_wrong_guesses(matrix, index, names, n=3):
    """For one country (its position in the list), returns the n countries its photos were most
    often wrongly guessed as, as (name, count) pairs, most common first. Ties stay in the order
    given, and countries with a count of zero are left out."""
    row = matrix[index]
    wrong = [(names[j], count) for j, count in enumerate(row) if j != index and count > 0]
    return sorted(wrong, key=lambda pair: -pair[1])[:n]


if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    names = [readable_name(code) for code in codes]
    folder = FEATURES_DIR / f"{len(codes)}_countries"
    train, test = np.load(folder / "train.npz"), np.load(folder / "test.npz")
    X_train, y_train = train["features"], train["labels"]
    X_test, y_test = test["features"], test["labels"]
    assert y_test.max() + 1 == len(codes), "Saved photo numbers and country folders don't match"

    print(f"Training the classifier on {len(y_train)} saved photo numbers (C={BEST_C})...")
    clf = LogisticRegression(C=BEST_C, max_iter=1000)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(f"Overall test accuracy: {(predicted == y_test).mean():.1%} "
          f"(clip_probe.py got 25.9%, so these should match)")

    matrix = build_confusion_matrix(y_test.tolist(), predicted.tolist(), len(codes))

    print("\nMost common mix-ups across all countries:")
    for actual, guessed, count in find_most_confused_pairs(matrix, names, top_n=15):
        print(f"  Real {actual}, guessed {guessed}: {count} times")

    print("\nWhere the photos of some low scoring countries went:")
    for code in CHECK_CODES:
        index = codes.index(code)
        right = matrix[index][index]
        wrong = top_wrong_guesses(matrix, index, names)
        guesses = ", ".join(f"{name} ({count})" for name, count in wrong)
        print(f"  {names[index]}: {right} of {sum(matrix[index])} right, wrong ones mostly went to {guesses}")
