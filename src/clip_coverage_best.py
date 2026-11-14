"""Checks what happens to the best setup's accuracy if it only answers when it is fairly sure
(the logistic regression on the OpenAI and DFN-2B ViT-L-14 photo numbers joined, with the
zero-shot sentences blended in at weight 0.2). It sorts the test photos by how confident the top
guess was, keeps only the most confident share (all of them, 80%, 60% and so on down to 10%),
and prints the top-1 accuracy on the photos it kept, next to the lowest confidence it kept.
Nothing is chosen from these numbers, so only the test split is printed. Uses the saved photo
numbers, so nothing gets extracted again.

Run it with:
    python3 src/clip_coverage_best.py
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

COVERAGES = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.1]


def accuracy_at_coverage(confidence, correct, coverages=COVERAGES):
    """For each share of the photos, keeps that share with the highest confidence and gives back
    (share, photos kept, lowest confidence kept, share of the kept photos that were right)."""
    confidence = np.asarray(confidence)
    correct = np.asarray(correct)
    order = np.argsort(-confidence, kind="stable")
    sorted_confidence = confidence[order]
    sorted_correct = correct[order]
    rows = []
    for coverage in coverages:
        count = max(1, round(len(confidence) * coverage))
        rows.append((coverage, count, float(sorted_confidence[count - 1]), float(sorted_correct[:count].mean())))
    return rows


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
    probabilities = np.exp(blended_log_probs(clf, X_test, sentences))
    correct = probabilities.argmax(axis=1) == y_test
    print(f"  Test top-1: {correct.mean():.1%} (should be 43.5%)")

    print("\nKeeping only the most confident share of the test photos:")
    for coverage, count, cutoff, accuracy in accuracy_at_coverage(probabilities.max(axis=1), correct):
        print(f"  {coverage:.0%}: {count} photos, confidence at least {cutoff:.1%}, top-1 accuracy {accuracy:.1%}")
