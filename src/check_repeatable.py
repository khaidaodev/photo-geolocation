"""
Checks that the best setup gives the same answers when it is trained twice from scratch. If a
number changed between two identical runs, none of the other numbers could be trusted to the
decimal place.

Run it with:
    python3 src/check_repeatable.py
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


def compare_runs(first_scores, second_scores):
    """Gives back (share of photos where both runs pick the same country, biggest difference
    between any two scores)."""
    first_scores = np.asarray(first_scores)
    second_scores = np.asarray(second_scores)
    same = (first_scores.argmax(axis=1) == second_scores.argmax(axis=1)).mean()
    return float(same), float(np.abs(first_scores - second_scores).max())


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

    runs = []
    for attempt in (1, 2):
        print(f"Training run {attempt} (C={BEST_C})...")
        clf = LogisticRegression(C=BEST_C, max_iter=1000)
        clf.fit(X_train, y_train)
        runs.append(blended_log_probs(clf, X_test, sentences))
        print(f"  Test top-1: {(runs[-1].argmax(axis=1) == y_test).mean():.1%}")

    same, biggest = compare_runs(runs[0], runs[1])
    print(f"\n  Photos where both runs pick the same country: {same:.1%}")
    print(f"  Biggest difference between any two scores: {biggest:.2e}")
