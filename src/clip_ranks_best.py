"""
Looks past top-1, top-3 and top-5. For every test photo it finds where the true country sits in
the model's list of guesses (1 means it was the first guess), then prints the share of photos
with the true country inside the top 1, 3, 5, 10, 20 and 50 guesses, plus the median rank.

Run it with:
    python3 src/clip_ranks_best.py
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

CUTOFFS = [1, 3, 5, 10, 20, 50]


def true_country_ranks(scores, labels):
    """For each photo, how many countries scored higher than the true one, plus one."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    true_scores = scores[np.arange(len(labels)), labels]
    return (scores > true_scores[:, None]).sum(axis=1) + 1


def share_within(ranks, cutoffs=CUTOFFS):
    """For each cutoff, the share of photos whose true country ranked at or above it."""
    ranks = np.asarray(ranks)
    return [(cutoff, float((ranks <= cutoff).mean())) for cutoff in cutoffs]


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
    ranks = true_country_ranks(blended_log_probs(clf, X_test, sentences), y_test)

    print("\nShare of test photos with the true country inside the top N guesses:")
    for cutoff, share in share_within(ranks):
        print(f"  top {cutoff}: {share:.1%}")
    print(f"\n  Median rank of the true country: {int(np.median(ranks))} out of {len(codes)}")
