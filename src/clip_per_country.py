"""
Works out how well CLIP does on each country separately, instead of one average across all 211.

The classifier from clip_probe.py wasn't saved, so this trains it again from the photo numbers
that script saved, using the same C that won on the valid split. Every country has 100 test
photos, so each country's score is a fraction out of 100. With that few photos, one country's
score can wobble by several points, so the pattern across countries is more trustworthy than
the exact order of two close countries.

Run it with:
    python src/clip_per_country.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from clip_probe import COUNTRY211_DIR, FEATURES_DIR
from clip_zero_shot import readable_name

BEST_C = 10.0  # the value that won on the valid split in clip_probe.py


def per_country_accuracy(true_labels, predicted_labels, num_classes):
    """Returns one number per country: the fraction of that country's test photos the
    classifier got right. A country with no photos gets 0.0, which can't happen here because
    every Country211 test folder has 100."""
    totals = np.bincount(true_labels, minlength=num_classes)
    correct = np.bincount(true_labels[predicted_labels == true_labels], minlength=num_classes)
    return correct / np.maximum(totals, 1)


def best_and_worst(accuracies, codes, n=10):
    """Returns the n countries with the highest scores and the n with the lowest, each as
    (country_code, score) pairs. Countries with the same score stay in the order given."""
    ranked = list(zip(codes, (float(a) for a in accuracies)))
    best = sorted(ranked, key=lambda pair: -pair[1])[:n]
    worst = sorted(ranked, key=lambda pair: pair[1])[:n]
    return best, worst


def summarise(accuracies):
    """Boils the per-country scores down to a few counts: the middle country's score, how many
    countries scored exactly zero, how many scored under 10%, and how many scored 50% or more."""
    accuracies = np.asarray(accuracies)
    return {
        "median": float(np.median(accuracies)),
        "at_zero": int((accuracies == 0).sum()),
        "below_10_percent": int((accuracies < 0.10).sum()),
        "at_least_50_percent": int((accuracies >= 0.50).sum()),
    }


if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
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

    accuracies = per_country_accuracy(y_test, predicted, len(codes))
    best, worst = best_and_worst(accuracies, codes)
    summary = summarise(accuracies)

    print(f"\nThe middle country scored {summary['median']:.0%}.")
    print(f"{summary['at_least_50_percent']} of {len(codes)} countries scored 50% or more, "
          f"{summary['below_10_percent']} scored under 10%, {summary['at_zero']} scored exactly 0%.")

    print("\nBest 10 countries:")
    for code, accuracy in best:
        print(f"  {readable_name(code)} ({code}): {accuracy:.0%}")
    print("\nWorst 10 countries:")
    for code, accuracy in worst:
        print(f"  {readable_name(code)} ({code}): {accuracy:.0%}")
