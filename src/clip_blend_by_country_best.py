"""
Follows on from clip_blend_by_continent_best.py. That showed the sentence blend helps most
continents a little, so this looks one level down: which individual countries gain the most from
adding the sentences, and which lose the most. Each country has only 100 test photos, so one photo
is one point and the small differences here are not strong evidence.

Run it with:
    python3 src/clip_blend_by_country_best.py
"""

import numpy as np
import open_clip
from sklearn.linear_model import LogisticRegression

from clip_joined_l14 import join_features
from clip_per_country import BEST_C
from clip_predict_best import BLEND_WEIGHT, blended_log_probs
from clip_probe import COUNTRY211_DIR
from clip_probe_l14 import CLIP_MODEL_L14, FEATURES_DIR_L14
from clip_probe_l14_dfn2b import FEATURES_DIR_L14_DFN2B
from clip_zero_shot import build_prompts, readable_name, text_features

HOW_MANY = 8


def country_changes(labels, without, with_blend):
    """For each country number gives (country number, photos, share right without, share right
    with, change in photos right), sorted from the biggest gain to the biggest loss."""
    labels = np.asarray(labels)
    rows = []
    for country in np.unique(labels):
        mine = labels == country
        right_without = int((np.asarray(without)[mine] == country).sum())
        right_with = int((np.asarray(with_blend)[mine] == country).sum())
        count = int(mine.sum())
        rows.append((int(country), count, right_without / count, right_with / count, right_with - right_without))
    return sorted(rows, key=lambda row: -row[4])


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

    print(f"Training the joined classifier (C={BEST_C})...")
    clf = LogisticRegression(C=BEST_C, max_iter=1000)
    clf.fit(X_train, y_train)
    without = blended_log_probs(clf, X_test, sentences, 0.0).argmax(axis=1)
    with_blend = blended_log_probs(clf, X_test, sentences, BLEND_WEIGHT).argmax(axis=1)
    print(f"  Without sentences: {(without == y_test).mean():.1%}, with: {(with_blend == y_test).mean():.1%}")

    rows = country_changes(y_test, without, with_blend)
    helped = sum(1 for row in rows if row[4] > 0)
    hurt = sum(1 for row in rows if row[4] < 0)
    print(f"\n  Countries helped: {helped}, hurt: {hurt}, unchanged: {len(rows) - helped - hurt}")

    print("\nBiggest gains from the sentences (country, right without, right with, photos gained):")
    for country, count, before, after, change in rows[:HOW_MANY]:
        print(f"  {readable_name(codes[country])}: {before:.0%}, {after:.0%}, {change:+d}")
    print("\nBiggest losses from the sentences:")
    for country, count, before, after, change in rows[::-1][:HOW_MANY]:
        print(f"  {readable_name(codes[country])}: {before:.0%}, {after:.0%}, {change:+d}")
