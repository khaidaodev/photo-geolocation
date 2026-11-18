"""
Shows where the best setup sends photos when it gets the continent wrong. For each true
continent it prints what share of its test photos were guessed as each continent, so a row
reads as "of the photos really from here, this share were guessed as there".

Run it with:
    python3 src/clip_continent_confusion_best.py
"""

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
from continents import CONTINENTS, continent_of


def continent_confusion(true_codes, predicted_codes, names):
    """Gives back a grid where row i, column j is the share of photos truly from names[i] that
    were guessed as names[j]. A continent with no photos gets a row of zeros."""
    position = {name: i for i, name in enumerate(names)}
    counts = np.zeros((len(names), len(names)))
    for true, predicted in zip(true_codes, predicted_codes):
        counts[position[continent_of(true)], position[continent_of(predicted)]] += 1
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)


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

    names = list(CONTINENTS)
    grid = continent_confusion([codes[i] for i in y_test], [codes[i] for i in predicted], names)
    print("\nRows are the true continent, columns are the guessed one (share of that row):")
    print(" " * 15 + "".join(f"{name[:9]:>11}" for name in names))
    for name, row in zip(names, grid):
        print(f"{name:<15}" + "".join(f"{value:>10.1%} " for value in row))
