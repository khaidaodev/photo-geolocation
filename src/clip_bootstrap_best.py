"""Puts a range on the best setup's test score (43.5% top 1: the logistic regression on the OpenAI
and DFN-2B ViT-L-14 photo numbers joined, with the zero-shot sentences blended in at weight 0.2)
and on the OpenAI ViT-L-14 on its own (40.4%). It resamples the 21,100 test photos with
replacement 1000 times, scores both setups on each resample, and reports the middle 95% of the
scores. The same photos are picked for both setups each time, so the gap between them is
measured on matching samples. Nothing gets trained again except the two classifiers, from the
saved photo numbers.

Run it with:
    python3 src/clip_bootstrap_best.py
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

N_RESAMPLES = 1000
SEED = 0


def bootstrap_accuracies(correct_sets, n_resamples=N_RESAMPLES, seed=SEED):
    """Each item in correct_sets says, photo by photo, whether one setup got the photo right.
    Resamples the photos with replacement n_resamples times and gives back one list of accuracies
    per setup. The same photos are picked for every setup within a resample."""
    n_photos = len(correct_sets[0])
    picks = np.random.default_rng(seed).integers(0, n_photos, size=(n_resamples, n_photos), dtype=np.int32)
    return [np.asarray(correct)[picks].mean(axis=1) for correct in correct_sets]


def middle_range(values, level=0.95):
    """The low and high end of the middle share of the values (95% by default)."""
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


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

    print(f"Training the OpenAI ViT-L-14 classifier on its own (C={BEST_C})...")
    clf_openai = LogisticRegression(C=BEST_C, max_iter=1000)
    clf_openai.fit(X_train[:, :768], y_train)
    correct_openai = clf_openai.predict(X_test[:, :768]) == y_test
    print(f"  Test top-1: {correct_openai.mean():.1%} (should be 40.4%)")

    print(f"Training the joined classifier (C={BEST_C}) and blending in the sentences...")
    clf_joined = LogisticRegression(C=BEST_C, max_iter=1000)
    clf_joined.fit(X_train, y_train)
    correct_best = blended_log_probs(clf_joined, X_test, sentences).argmax(axis=1) == y_test
    print(f"  Test top-1: {correct_best.mean():.1%} (should be 43.5%)")

    acc_openai, acc_best = bootstrap_accuracies([correct_openai, correct_best])
    gap = acc_best - acc_openai
    print(f"\nResampling the {len(y_test)} test photos {N_RESAMPLES} times (middle 95% of the scores):")
    for label, accuracies in (("OpenAI ViT-L-14 alone", acc_openai), ("Best setup", acc_best)):
        low, high = middle_range(accuracies)
        print(f"  {label}: {low:.1%} to {high:.1%}")
    low, high = middle_range(gap)
    print(f"  Best setup minus OpenAI alone: {low * 100:.2f} to {high * 100:.2f} points")
    print(f"  The best setup scored higher than OpenAI alone in {(gap > 0).mean():.1%} of the resamples.")
