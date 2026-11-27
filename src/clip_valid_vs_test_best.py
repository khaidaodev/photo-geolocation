"""
Compares the best setup's score on the valid split with its score on the test split. C and the
sentence weight were both picked using the valid photos, so if valid came out far above test,
the settings would be tuned to those 10,550 photos and the test score would be the honest one.

Run it with:
    python3 src/clip_valid_vs_test_best.py
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
from clip_ranks_best import share_within, true_country_ranks
from clip_zero_shot import build_prompts, text_features

CUTOFFS = [1, 3, 5]


def score_gaps(valid_shares, test_shares):
    """Takes two lists of (k, share) and gives back (k, valid share, test share, test minus valid)."""
    return [(k, v, t, t - v) for (k, v), (_, t) in zip(valid_shares, test_shares)]


if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    folder_name = f"{len(codes)}_countries"
    print(f"Loading CLIP ({CLIP_MODEL_L14}, frozen) for the country sentences...")
    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL_L14, pretrained="openai")
    model.eval()
    sentences = text_features(model, open_clip.get_tokenizer(CLIP_MODEL_L14), build_prompts(codes))

    data = {}
    for split in ("train", "valid", "test"):
        first = np.load(FEATURES_DIR_L14 / folder_name / f"{split}.npz")
        second = np.load(FEATURES_DIR_L14_DFN2B / folder_name / f"{split}.npz")
        data[split] = (join_features(first, second), first["labels"])
    X_train, y_train = data["train"]

    print(f"Training the joined classifier (C={BEST_C}) and blending in the sentences...")
    clf = LogisticRegression(C=BEST_C, max_iter=1000)
    clf.fit(X_train, y_train)

    shares = {}
    for split in ("valid", "test"):
        X, y = data[split]
        ranks = true_country_ranks(blended_log_probs(clf, X, sentences), y)
        shares[split] = share_within(ranks, CUTOFFS)

    print("\n  Guesses allowed: valid, test, test minus valid")
    for k, v, t, gap in score_gaps(shares["valid"], shares["test"]):
        print(f"  top {k}: {v:.1%}, {t:.1%}, {gap * 100:+.1f} points")
