"""Per-country scores for the ViT-L-14 CLIP classifier, next to ViT-B-32 on the same countries.

Trains both classifiers again from the saved photo numbers (C=10, the value that won on the
valid split) and prints each overall score so I can check it against what clip_probe.py and
clip_probe_l14.py printed. Then it shows whether the bigger model fixes the countries ViT-B-32
did worst on, or only scores higher on countries that were already easy. Each country has 100
test photos, so one country's score can wobble by several points.

Run it with:
    python3 src/clip_per_country_l14.py
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from clip_per_country import BEST_C, best_and_worst, per_country_accuracy, summarise
from clip_probe import COUNTRY211_DIR, FEATURES_DIR
from clip_probe_l14 import FEATURES_DIR_L14
from clip_zero_shot import readable_name


def train_and_score(folder, num_classes):
    """Trains the classifier on the saved train photo numbers in folder, then returns the
    overall test accuracy and one score per country."""
    train, test = np.load(folder / "train.npz"), np.load(folder / "test.npz")
    X_train, y_train = train["features"], train["labels"]
    X_test, y_test = test["features"], test["labels"]
    assert y_test.max() + 1 == num_classes, "Saved photo numbers and country folders don't match"
    print(f"  Training on {len(y_train)} saved photo numbers (C={BEST_C})...")
    clf = LogisticRegression(C=BEST_C, max_iter=1000)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    return float((predicted == y_test).mean()), per_country_accuracy(y_test, predicted, num_classes)


if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    folder_name = f"{len(codes)}_countries"

    print("ViT-B-32:")
    overall_b32, acc_b32 = train_and_score(FEATURES_DIR / folder_name, len(codes))
    print(f"  Overall test accuracy: {overall_b32:.1%} (clip_probe.py got 25.9%, so these should match)")
    print("ViT-L-14:")
    overall_l14, acc_l14 = train_and_score(FEATURES_DIR_L14 / folder_name, len(codes))
    print(f"  Overall test accuracy: {overall_l14:.1%} (clip_probe_l14.py got 40.4%, so these should match)")

    for label, accuracies in (("ViT-B-32", acc_b32), ("ViT-L-14", acc_l14)):
        summary = summarise(accuracies)
        print(f"\n{label}: the middle country scored {summary['median']:.0%}. "
              f"{summary['at_least_50_percent']} of {len(codes)} countries scored 50% or more, "
              f"{summary['below_10_percent']} scored under 10%, {summary['at_zero']} scored exactly 0%.")

    best, worst = best_and_worst(acc_l14, codes)
    print("\nViT-L-14 best 10 countries:")
    for code, accuracy in best:
        print(f"  {readable_name(code)} ({code}): {accuracy:.0%}")
    print("\nViT-L-14 worst 10 countries:")
    for code, accuracy in worst:
        print(f"  {readable_name(code)} ({code}): {accuracy:.0%}")

    _, b32_worst = best_and_worst(acc_b32, codes)
    print("\nThe 10 countries ViT-B-32 scored lowest on, and what ViT-L-14 got on them:")
    for code, old in b32_worst:
        new = float(acc_l14[codes.index(code)])
        print(f"  {readable_name(code)} ({code}): {old:.0%} -> {new:.0%}")

    better = int((acc_l14 > acc_b32).sum())
    same = int((acc_l14 == acc_b32).sum())
    worse = int((acc_l14 < acc_b32).sum())
    print(f"\nViT-L-14 scored higher than ViT-B-32 on {better} countries, the same on {same}, lower on {worse}.")
