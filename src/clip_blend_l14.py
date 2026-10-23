"""Blends the two ways of using the ViT-L-14 CLIP model: the trained classifier (40.4% top 1)
and the zero-shot sentences (29.5% top 1). Each one gives every country a score for a photo.
The blend is (1 - weight) * classifier score + weight * zero-shot score, both turned into
log-probabilities first. The weight is picked on the valid split only, then the test split is
read once at the end. Weight 0 is the classifier alone and weight 1 is zero-shot alone, so those
two should match the earlier scripts. No photos get processed again, the numbers are cached.

Run it with:
    python3 src/clip_blend_l14.py
"""

import numpy as np
import open_clip
from sklearn.linear_model import LogisticRegression

from clip_per_country import BEST_C
from clip_probe import COUNTRY211_DIR, top_k_scores
from clip_probe_l14 import CLIP_MODEL_L14, FEATURES_DIR_L14
from clip_zero_shot import build_prompts, text_features

WEIGHTS = [round(0.1 * i, 1) for i in range(11)]
LOGIT_SCALE = 100.0  # the usual size CLIP's similarity scores are multiplied by before a softmax


def log_softmax(scores):
    """Turns each row of scores into log-probabilities that add up to 1 across the countries."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def blend(classifier_part, zero_shot_part, weight):
    """Mixes the two sets of scores. Weight 0 is the trained classifier only, 1 is zero-shot only."""
    return (1 - weight) * classifier_part + weight * zero_shot_part


if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    print(f"Loading CLIP ({CLIP_MODEL_L14}, frozen) for the country sentences...")
    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL_L14, pretrained="openai")
    model.eval()
    sentences = text_features(model, open_clip.get_tokenizer(CLIP_MODEL_L14), build_prompts(codes))

    folder = FEATURES_DIR_L14 / f"{len(codes)}_countries"
    train = np.load(folder / "train.npz")
    print(f"Training the classifier on {len(train['labels'])} saved photo numbers (C={BEST_C})...")
    clf = LogisticRegression(C=BEST_C, max_iter=1000)
    clf.fit(train["features"], train["labels"])
    assert list(clf.classes_) == list(range(len(codes)))

    parts = {}
    for split in ("valid", "test"):
        saved = np.load(folder / f"{split}.npz")
        photos, labels = saved["features"], saved["labels"]
        assert labels.max() + 1 == len(codes), "Saved photo numbers and country folders don't match"
        classifier_part = np.log(clf.predict_proba(photos) + 1e-12)
        zero_shot_part = log_softmax(LOGIT_SCALE * (photos @ sentences.T))
        parts[split] = (classifier_part, zero_shot_part, labels)

    print("Weight on the zero-shot scores, and valid top-1 (the weight is picked on valid only):")
    classifier_part, zero_shot_part, labels = parts["valid"]
    best_weight, best_valid = 0.0, -1.0
    for weight in WEIGHTS:
        top1 = top_k_scores(blend(classifier_part, zero_shot_part, weight), labels, ks=(1,))[1]
        print(f"  {weight:.1f}: {top1:.1%}")
        if top1 > best_valid:
            best_weight, best_valid = weight, top1

    classifier_part, zero_shot_part, labels = parts["test"]
    print(f"Best weight was {best_weight} (valid top-1 {best_valid:.1%}). Test split, never used until now:")
    for k, accuracy in top_k_scores(blend(classifier_part, zero_shot_part, best_weight), labels).items():
        print(f"Right country in the top {k} guesses: {accuracy:.1%}")
    for weight, name, expected in ((0.0, "classifier only", "40.4%"), (1.0, "zero-shot only", "29.5%")):
        top1 = top_k_scores(blend(classifier_part, zero_shot_part, weight), labels, ks=(1,))[1]
        print(f"Check, {name}: test top-1 {top1:.1%} (should be {expected})")
