"""Blends the zero-shot sentences into the best setup so far: the logistic regression trained on
the OpenAI and DFN-2B ViT-L-14 photo numbers joined together (42.3% top 1). The sentences are the
same one per country as before ("a photo taken in {}"), scored with the OpenAI ViT-L-14 photo
numbers, which are the first 768 columns of the joined numbers. Same blend as clip_blend_l14.py:
(1 - weight) * classifier score + weight * zero-shot score, both as log-probabilities. The weight
is picked on the valid split only, then the test split is read once at the end. Weight 0 is the
joined classifier alone, which should match clip_joined_l14.py. Nothing gets extracted again.

Run it with:
    python3 src/clip_blend_joined_l14.py
"""

import numpy as np
import open_clip
from sklearn.linear_model import LogisticRegression

from clip_blend_l14 import LOGIT_SCALE, WEIGHTS, blend, log_softmax
from clip_joined_l14 import join_features
from clip_per_country import BEST_C
from clip_probe import COUNTRY211_DIR, top_k_scores
from clip_probe_l14 import CLIP_MODEL_L14, FEATURES_DIR_L14
from clip_probe_l14_dfn2b import FEATURES_DIR_L14_DFN2B
from clip_zero_shot import build_prompts, text_features

OPENAI_COLUMNS = slice(0, 768)

if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    print(f"Loading CLIP ({CLIP_MODEL_L14}, frozen) for the country sentences...")
    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL_L14, pretrained="openai")
    model.eval()
    sentences = text_features(model, open_clip.get_tokenizer(CLIP_MODEL_L14), build_prompts(codes))

    folder_name = f"{len(codes)}_countries"
    data = {}
    for split in ("train", "valid", "test"):
        first = np.load(FEATURES_DIR_L14 / folder_name / f"{split}.npz")
        second = np.load(FEATURES_DIR_L14_DFN2B / folder_name / f"{split}.npz")
        data[split] = (join_features(first, second), first["labels"])
    X_train, y_train = data["train"]
    assert data["test"][1].max() + 1 == len(codes), "Saved photo numbers and country folders don't match"

    print(f"Training the classifier on both sets of photo numbers (C={BEST_C})...")
    clf = LogisticRegression(C=BEST_C, max_iter=1000)
    clf.fit(X_train, y_train)
    assert list(clf.classes_) == list(range(len(codes)))

    parts = {}
    for split in ("valid", "test"):
        X, labels = data[split]
        classifier_part = np.log(clf.predict_proba(X) + 1e-12)
        zero_shot_part = log_softmax(LOGIT_SCALE * (X[:, OPENAI_COLUMNS] @ sentences.T))
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
    top1 = top_k_scores(blend(classifier_part, zero_shot_part, 0.0), labels, ks=(1,))[1]
    print(f"Check, joined classifier only: test top-1 {top1:.1%} (should be 42.3%)")
