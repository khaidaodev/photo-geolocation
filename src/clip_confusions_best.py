"""Checks where the best setup's wrong guesses go (43.5% top 1: the logistic regression on the
OpenAI and DFN-2B ViT-L-14 photo numbers joined, with the zero-shot sentences blended in at
weight 0.2). It takes the 15 most common mix-ups of the OpenAI ViT-L-14 on its own (40.4%) and
shows how many times the best setup made the same mistake. It also lists the best setup's own 15
most common mix-ups and where the photos of the low scoring countries went. Each country only has
100 test photos, so a count of 3 or 4 is weak evidence and a count of 20 is not. Uses the saved
photo numbers, so nothing gets extracted again.

Run it with:
    python3 src/clip_confusions_best.py
"""

import numpy as np
import open_clip
from sklearn.linear_model import LogisticRegression

from clip_confusions import CHECK_CODES, top_wrong_guesses
from clip_joined_l14 import join_features
from clip_per_country import BEST_C
from clip_predict_best import blended_log_probs
from clip_probe import COUNTRY211_DIR
from clip_probe_l14 import CLIP_MODEL_L14, FEATURES_DIR_L14
from clip_probe_l14_dfn2b import FEATURES_DIR_L14_DFN2B
from clip_zero_shot import build_prompts, readable_name, text_features
from confusion_matrix import build_confusion_matrix, find_most_confused_pairs

if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    names = [readable_name(code) for code in codes]
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

    matrix_openai = build_confusion_matrix(y_test.tolist(), guessed_openai.tolist(), len(codes))
    matrix_best = build_confusion_matrix(y_test.tolist(), guessed_best.tolist(), len(codes))

    print("\nThe 15 most common OpenAI ViT-L-14 mix-ups, and how many times the best setup made the same one:")
    fewer = 0
    for actual, guessed, count in find_most_confused_pairs(matrix_openai, names, top_n=15):
        new = matrix_best[names.index(actual)][names.index(guessed)]
        fewer += int(new < count)
        print(f"  Real {actual}, guessed {guessed}: {count} -> {new}")
    print(f"The best setup made fewer of these mistakes in {fewer} of the 15 pairs.")

    print("\nMost common best setup mix-ups across all countries:")
    for actual, guessed, count in find_most_confused_pairs(matrix_best, names, top_n=15):
        print(f"  Real {actual}, guessed {guessed}: {count} times")

    print("\nWhere the photos of the low scoring countries went with the best setup:")
    for code in CHECK_CODES:
        index = codes.index(code)
        right = matrix_best[index][index]
        wrong = top_wrong_guesses(matrix_best, index, names)
        guesses = ", ".join(f"{name} ({count})" for name, count in wrong)
        print(f"  {names[index]}: {right} of {sum(matrix_best[index])} right, wrong ones mostly went to {guesses}")
