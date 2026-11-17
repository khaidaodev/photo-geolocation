"""
Asks how much of the best setup's error is "right continent, wrong country". Each test photo
gets the country the model picks, and that country and the true one are both turned into
continents with continents.py. Then it prints the share of photos with the right country, the
share with the right continent, and, out of the photos with the wrong country, how many still
landed on the right continent.

Run it with:
    python3 src/clip_continents_best.py
"""

from collections import defaultdict

import numpy as np
import open_clip
from sklearn.linear_model import LogisticRegression

from clip_joined_l14 import join_features
from clip_per_country import BEST_C
from clip_predict_best import blended_log_probs
from clip_probe import COUNTRY211_DIR
from clip_probe_l14 import CLIP_MODEL_L14, FEATURES_DIR_L14
from clip_probe_l14_dfn2b import FEATURES_DIR_L14_DFN2B
from clip_zero_shot import build_prompts, text_features
from continents import continent_of


def continent_scores(true_codes, predicted_codes):
    """Gives back (country accuracy, continent accuracy, share of the wrong-country photos that
    still had the right continent, and a dict of continent -> (photos, country acc, continent acc))."""
    true_codes = list(true_codes)
    predicted_codes = list(predicted_codes)
    country_right = [t == p for t, p in zip(true_codes, predicted_codes)]
    continent_right = [continent_of(t) == continent_of(p) for t, p in zip(true_codes, predicted_codes)]
    wrong = [c for r, c in zip(country_right, continent_right) if not r]
    by_continent = defaultdict(lambda: [0, 0, 0])
    for t, cr, tr in zip(true_codes, country_right, continent_right):
        row = by_continent[continent_of(t)]
        row[0] += 1
        row[1] += cr
        row[2] += tr
    per_continent = {name: (n, a / n, b / n) for name, (n, a, b) in by_continent.items()}
    wrong_share = sum(wrong) / len(wrong) if wrong else 0.0
    return sum(country_right) / len(country_right), sum(continent_right) / len(continent_right), wrong_share, per_continent


if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    folder_name = f"{len(codes)}_countries"
    print(f"Loading CLIP ({CLIP_MODEL_L14}, frozen) for the country sentences...")
    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL_L14, pretrained="openai")
    model.eval()
    sentences = text_features(model, open_clip.get_tokenizer(CLIP_MODEL_L14), build_prompts(codes))

    data = {}
    for split in ("train", "test"):
        first = np.load(FEATURES_DIR_L14 / folder_name / f"{split}.npz")
        second = np.load(FEATURES_DIR_L14_DFN2B / folder_name / f"{split}.npz")
        data[split] = (join_features(first, second), first["labels"])
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]
    assert y_test.max() + 1 == len(codes), "Saved photo numbers and country folders don't match"

    print(f"Training the joined classifier (C={BEST_C}) and blending in the sentences...")
    clf = LogisticRegression(C=BEST_C, max_iter=1000)
    clf.fit(X_train, y_train)
    predicted = blended_log_probs(clf, X_test, sentences).argmax(axis=1)

    country_acc, continent_acc, wrong_share, per_continent = continent_scores(
        [codes[i] for i in y_test], [codes[i] for i in predicted]
    )
    print(f"\n  Right country (top-1): {country_acc:.1%} (should be 43.5%)")
    print(f"  Right continent: {continent_acc:.1%}")
    print(f"  Wrong country but right continent: {wrong_share:.1%} of the wrong answers")
    print("\nPer true continent (photos, right country, right continent):")
    for name, (count, c_acc, k_acc) in sorted(per_continent.items(), key=lambda item: -item[1][0]):
        print(f"  {name}: {count} photos, {c_acc:.1%}, {k_acc:.1%}")
