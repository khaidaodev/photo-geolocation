"""Checks how the best setup's test score changes with the number of training photos (the
logistic regression on the OpenAI and DFN-2B ViT-L-14 photo numbers joined, with the zero-shot
sentences blended in at weight 0.2). clip_training_size_l14.py did this for the OpenAI ViT-L-14
alone with one random pick per share. Here each share is picked 5 times with different seeds, so
the table shows how much a score moves between picks, not just one pick. The same share is taken
from every country. The full set of training photos has nothing to pick, so it runs once.
Nothing is chosen from these numbers, so only the test split is printed. Uses the saved photo
numbers, so nothing gets extracted again.

Run it with:
    python3 src/clip_training_size_best.py
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
from clip_training_size_l14 import subsample_per_country
from clip_zero_shot import build_prompts, text_features

FRACTIONS = [0.1, 0.25, 0.5]
SEEDS = [0, 1, 2, 3, 4]


def spread(scores):
    """The average, lowest and highest of the scores from the different picks."""
    return float(np.mean(scores)), float(np.min(scores)), float(np.max(scores))


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

    def top1_for(keep):
        clf = LogisticRegression(C=BEST_C, max_iter=1000)
        clf.fit(X_train[keep], y_train[keep])
        return (blended_log_probs(clf, X_test, sentences).argmax(axis=1) == y_test).mean()

    print(f"Test top-1 for the best setup. Each share is picked {len(SEEDS)} times with different seeds:")
    for fraction in FRACTIONS:
        scores = [top1_for(subsample_per_country(y_train, fraction, seed)) for seed in SEEDS]
        average, low, high = spread(scores)
        print(f"  {fraction:.0%}: average {average:.1%}, lowest {low:.1%}, highest {high:.1%}")
    full = top1_for(np.arange(len(y_train)))
    print(f"  100%: {full:.1%} (should be 43.5%, nothing to pick so it ran once)")
