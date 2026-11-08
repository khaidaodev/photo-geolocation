"""Checks how honest the best setup's percentages are (43.5% top 1: the logistic regression on
the OpenAI and DFN-2B ViT-L-14 photo numbers joined, with the zero-shot sentences blended in at
weight 0.2). For the single OpenAI ViT-L-14 classifier, clip_calibrate_l14.py cut the average gap
between what it said and how often it was right from 5.3% to 2.3%. This does the same for the
blended scores: the temperature is picked on the valid split only by lowest log loss, then the
test split is read once. The order of the countries for each photo never changes, so top 1 must
stay the same. Uses the saved photo numbers, so nothing gets extracted again.

Run it with:
    python3 src/clip_calibrate_best.py
"""

import numpy as np
import open_clip
from sklearn.linear_model import LogisticRegression

from clip_calibrate_l14 import TEMPERATURES, log_loss, rescale, summarise
from clip_joined_l14 import join_features
from clip_per_country import BEST_C
from clip_predict_best import blended_log_probs
from clip_probe import COUNTRY211_DIR
from clip_probe_l14 import CLIP_MODEL_L14, FEATURES_DIR_L14
from clip_probe_l14_dfn2b import FEATURES_DIR_L14_DFN2B
from clip_zero_shot import build_prompts, text_features

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
    X_valid, y_valid = data["valid"]
    X_test, y_test = data["test"]
    assert y_test.max() + 1 == len(codes), "Saved photo numbers and country folders don't match"

    print(f"Training the joined classifier on {len(y_train)} saved photo numbers (C={BEST_C})...")
    clf = LogisticRegression(C=BEST_C, max_iter=1000)
    clf.fit(X_train, y_train)
    assert list(clf.classes_) == list(range(len(codes)))
    valid_log = blended_log_probs(clf, X_valid, sentences)
    test_log = blended_log_probs(clf, X_test, sentences)

    print("Temperature and valid log loss (lower is better, 1.0 is no change, picked on valid only):")
    best_t, best_loss = 1.0, float("inf")
    for t in TEMPERATURES:
        loss = log_loss(rescale(valid_log, t), y_valid)
        print(f"  {t:.2f}: {loss:.4f}")
        if loss < best_loss:
            best_t, best_loss = t, loss

    before_acc, before_rows, before_gap = summarise(rescale(test_log, 1.0), y_test)
    after_acc, after_rows, after_gap = summarise(rescale(test_log, best_t), y_test)
    print(f"Best temperature was {best_t}. Test split, never used until now:")
    print(f"Top-1 accuracy before {before_acc:.1%}, after {after_acc:.1%} (should be the same, 43.5%)")
    print(f"Average gap between what it said and how often it was right: before {before_gap:.1%}, after {after_gap:.1%}")

    for label, rows in (("Before", before_rows), ("After", after_rows)):
        print(f"\n{label}: confidence of the top guess, how many test photos, average confidence, how often right:")
        for low, high, count, average, right in rows:
            print(f"  {low:.0%}-{high:.0%}: {count} photos, said {average:.1%}, was right {right:.1%}")
