"""Per-country scores for the best setup so far (43.5% top 1): the logistic regression on the
OpenAI and DFN-2B ViT-L-14 photo numbers joined, with the zero-shot sentences blended in at
weight 0.2. It goes next to the OpenAI ViT-L-14 classifier on its own (40.4%), on the same
countries, to show whether the best setup fixes the countries the single model was weak on or
only scores higher on countries that were already easy. Each country has 100 test photos, so one
country's score can wobble by several points. Uses the saved photo numbers, so nothing gets
extracted again.

Run it with:
    python3 src/clip_per_country_best.py
"""

import numpy as np
import open_clip
from sklearn.linear_model import LogisticRegression

from clip_joined_l14 import join_features
from clip_per_country import BEST_C, best_and_worst, per_country_accuracy, summarise
from clip_predict_best import blended_log_probs
from clip_probe import COUNTRY211_DIR
from clip_probe_l14 import CLIP_MODEL_L14, FEATURES_DIR_L14
from clip_probe_l14_dfn2b import FEATURES_DIR_L14_DFN2B
from clip_zero_shot import build_prompts, readable_name, text_features

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
    guessed_openai = clf_openai.predict(X_test[:, :768])
    print(f"  Overall test accuracy: {(guessed_openai == y_test).mean():.1%} (should be 40.4%)")

    print(f"Training the joined classifier (C={BEST_C}) and blending in the sentences...")
    clf_joined = LogisticRegression(C=BEST_C, max_iter=1000)
    clf_joined.fit(X_train, y_train)
    guessed_best = blended_log_probs(clf_joined, X_test, sentences).argmax(axis=1)
    print(f"  Overall test accuracy: {(guessed_best == y_test).mean():.1%} (should be 43.5%)")

    acc_openai = per_country_accuracy(y_test, guessed_openai, len(codes))
    acc_best = per_country_accuracy(y_test, guessed_best, len(codes))
    for label, accuracies in (("OpenAI ViT-L-14 alone", acc_openai), ("Best setup", acc_best)):
        summary = summarise(accuracies)
        print(f"\n{label}: the middle country scored {summary['median']:.0%}. "
              f"{summary['at_least_50_percent']} of {len(codes)} countries scored 50% or more, "
              f"{summary['below_10_percent']} scored under 10%, {summary['at_zero']} scored exactly 0%.")

    best, worst = best_and_worst(acc_best, codes)
    print("\nBest setup, best 10 countries:")
    for code, accuracy in best:
        print(f"  {readable_name(code)} ({code}): {accuracy:.0%}")
    print("\nBest setup, worst 10 countries:")
    for code, accuracy in worst:
        print(f"  {readable_name(code)} ({code}): {accuracy:.0%}")

    _, openai_worst = best_and_worst(acc_openai, codes)
    print("\nThe 10 countries the OpenAI ViT-L-14 scored lowest on, and what the best setup got on them:")
    for code, old in openai_worst:
        new = float(acc_best[codes.index(code)])
        print(f"  {readable_name(code)} ({code}): {old:.0%} -> {new:.0%}")

    better = int((acc_best > acc_openai).sum())
    same = int((acc_best == acc_openai).sum())
    worse = int((acc_best < acc_openai).sum())
    print(f"\nThe best setup scored higher than the OpenAI ViT-L-14 on {better} countries, "
          f"the same on {same}, lower on {worse}.")
