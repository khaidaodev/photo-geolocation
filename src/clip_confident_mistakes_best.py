"""Lists the test photos the best setup got wrong while being most sure of itself (the logistic
regression on the OpenAI and DFN-2B ViT-L-14 photo numbers joined, with the zero-shot sentences
blended in at weight 0.2). For each of the 20 most confident wrong guesses it prints where the
photo file is, the real country, the guessed country and the confidence, so the photos can be
opened and looked at. It also counts the wrong guesses at a few confidence levels, and the most
common real and guessed pairs among the 100 most confident mistakes. The photo files are matched
to the saved photo numbers by the same folder order the extraction used, and the labels are
checked against each other first. Nothing is chosen from these numbers, so only the test split is
used. Uses the saved photo numbers, so nothing gets extracted again.

Run it with:
    python3 src/clip_confident_mistakes_best.py
"""

from collections import Counter
from pathlib import Path

import numpy as np
import open_clip
import torchvision
from sklearn.linear_model import LogisticRegression

from clip_joined_l14 import join_features
from clip_per_country import BEST_C
from clip_predict_best import blended_log_probs
from clip_probe import COUNTRY211_DIR
from clip_probe_l14 import CLIP_MODEL_L14, FEATURES_DIR_L14
from clip_probe_l14_dfn2b import FEATURES_DIR_L14_DFN2B
from clip_zero_shot import build_prompts, readable_name, text_features

N_LISTED = 20
N_PAIRS_FROM = 100
LEVELS = [0.9, 0.7, 0.5]


def most_confident_wrong(confidence, correct, n):
    """The positions of the n wrong guesses with the highest confidence, most confident first."""
    confidence = np.asarray(confidence)
    wrong_positions = np.where(~np.asarray(correct))[0]
    order = np.argsort(-confidence[wrong_positions], kind="stable")
    return wrong_positions[order][:n]


def wrong_at_least(confidence, correct, level):
    """Of the photos whose top guess had a confidence of at least this level, how many were
    wrong. Gives back (wrong, total)."""
    chosen = np.asarray(confidence) >= level
    return int((~np.asarray(correct)[chosen]).sum()), int(chosen.sum())


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

    photos = torchvision.datasets.ImageFolder(str(COUNTRY211_DIR / "test")).samples
    assert np.array_equal([label for _, label in photos], y_test), "Photo files and saved photo numbers don't match"

    print(f"Training the joined classifier (C={BEST_C}) and blending in the sentences...")
    clf = LogisticRegression(C=BEST_C, max_iter=1000)
    clf.fit(X_train, y_train)
    probabilities = np.exp(blended_log_probs(clf, X_test, sentences))
    guessed = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = guessed == y_test
    print(f"  Test top-1: {correct.mean():.1%} (should be 43.5%)")

    print(f"\nThe {N_LISTED} wrong guesses the best setup was most sure about:")
    for rank, position in enumerate(most_confident_wrong(confidence, correct, N_LISTED), start=1):
        path = Path(photos[position][0]).relative_to(COUNTRY211_DIR)
        print(f"  {rank}. {path}: real {readable_name(codes[y_test[position]])}, "
              f"guessed {readable_name(codes[guessed[position]])}, confidence {confidence[position]:.1%}")

    print("\nWrong guesses at each confidence level:")
    for level in LEVELS:
        wrong, total = wrong_at_least(confidence, correct, level)
        print(f"  Confidence at least {level:.0%}: {wrong} wrong out of {total} photos")

    pairs = Counter((y_test[p], guessed[p]) for p in most_confident_wrong(confidence, correct, N_PAIRS_FROM))
    print(f"\nMost common real and guessed pairs among the {N_PAIRS_FROM} most confident mistakes:")
    for (real, guess), count in pairs.most_common(5):
        print(f"  Real {readable_name(codes[real])}, guessed {readable_name(codes[guess])}: {count}")
