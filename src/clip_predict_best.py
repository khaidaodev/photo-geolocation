"""
Guesses which country a photo was taken in using the best setup so far (43.5% on all 211
countries). It runs each photo through two ViT-L-14 CLIP models (OpenAI weights and DFN-2B
weights), sticks the two sets of numbers together, scores them with the logistic regression
trained on that, and blends in the zero-shot sentences ("a photo taken in {}") at weight 0.2, the
same as clip_blend_joined_l14.py. It reuses the saving, loading and printing functions from
clip_predict.py, and saves its classifier under its own name so the others are left alone.

Two steps, run in this order:
    python3 src/clip_predict_best.py train                  (once, makes and saves the classifier)
    python3 src/clip_predict_best.py path/to/photo.jpg      (guesses a photo)

The train step needs the photo numbers that clip_probe_l14.py and clip_probe_l14_dfn2b.py save,
so run those first. The saved classifier goes in models/, which git ignores. The percentages are
the blended scores turned back into probabilities, and I only checked how honest percentages are
for the single OpenAI classifier, so I wouldn't treat these as exact odds.
"""

import sys
from pathlib import Path

import numpy as np
import open_clip
from sklearn.linear_model import LogisticRegression

from clip_blend_l14 import LOGIT_SCALE, blend, log_softmax
from clip_joined_l14 import join_features
from clip_per_country import BEST_C
from clip_predict import ROOT, format_guess, load_classifier, photo_features, save_classifier, top_guesses
from clip_probe import COUNTRY211_DIR, top_k_scores
from clip_probe_l14 import CLIP_MODEL_L14, FEATURES_DIR_L14
from clip_probe_l14_dfn2b import CLIP_MODEL_L14_DFN2B, FEATURES_DIR_L14_DFN2B
from clip_zero_shot import build_prompts, text_features

CLASSIFIER_PATH_BEST = ROOT / "models" / "clip_classifier_best.joblib"
BLEND_WEIGHT = 0.2
OPENAI_COLUMNS = slice(0, 768)  # the OpenAI half of the joined photo numbers


def blended_log_probs(clf, joined, sentences, weight=BLEND_WEIGHT):
    """Scores each photo (one row of joined numbers) for every country: the trained classifier's
    log-probabilities mixed with the zero-shot sentence scores, then renormalised so each row adds
    up to 1 as probabilities. The sentence scores use only the OpenAI half of the numbers."""
    classifier_part = np.log(clf.predict_proba(joined) + 1e-12)
    zero_shot_part = log_softmax(LOGIT_SCALE * (joined[:, OPENAI_COLUMNS] @ sentences.T))
    return log_softmax(blend(classifier_part, zero_shot_part, weight))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 src/clip_predict_best.py train   or   python3 src/clip_predict_best.py path/to/photo.jpg")
        sys.exit(1)

    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    print(f"Loading CLIP ({CLIP_MODEL_L14}, OpenAI weights, frozen) for the country sentences...")
    openai_model, _, openai_preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_L14, pretrained="openai")
    openai_model.eval()
    sentences = text_features(openai_model, open_clip.get_tokenizer(CLIP_MODEL_L14), build_prompts(codes))

    if sys.argv[1] == "train":
        folder_name = f"{len(codes)}_countries"
        data = {}
        for split in ("train", "test"):
            first = np.load(FEATURES_DIR_L14 / folder_name / f"{split}.npz")
            second = np.load(FEATURES_DIR_L14_DFN2B / folder_name / f"{split}.npz")
            data[split] = (join_features(first, second), first["labels"])
        X_train, y_train = data["train"]
        X_test, y_test = data["test"]
        assert y_test.max() + 1 == len(codes), "Saved photo numbers and country folders don't match"
        print(f"Training the classifier on {len(y_train)} saved photo numbers (C={BEST_C})...")
        clf = LogisticRegression(C=BEST_C, max_iter=1000)
        clf.fit(X_train, y_train)
        assert list(clf.classes_) == list(range(len(codes)))
        save_classifier(clf, codes, CLASSIFIER_PATH_BEST)
        print(f"Saved to {CLASSIFIER_PATH_BEST}")
        scores = top_k_scores(np.exp(blended_log_probs(clf, X_test, sentences)), y_test)
        print("Test accuracy of the saved classifier with the sentences blended in "
              "(clip_blend_joined_l14.py got 43.5%, 61.9% and 69.4%, so these should match):")
        for k, accuracy in scores.items():
            print(f"Right country in the top {k} guesses: {accuracy:.1%}")
    else:
        image_path = Path(sys.argv[1])
        clf, saved_codes = load_classifier(CLASSIFIER_PATH_BEST)
        assert saved_codes == codes, "The saved classifier and the country folders don't match"
        print(f"Loading CLIP ({CLIP_MODEL_L14_DFN2B}, DFN-2B weights, frozen)...")
        dfn_model, _, dfn_preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_L14_DFN2B, pretrained="dfn2b")
        dfn_model.eval()
        joined = np.concatenate([photo_features(openai_model, openai_preprocess, image_path),
                                 photo_features(dfn_model, dfn_preprocess, image_path)], axis=1)
        probabilities = np.exp(blended_log_probs(clf, joined, sentences))[0]
        print(f"Top guesses for {image_path.name}:")
        for code, probability in top_guesses(probabilities, codes):
            print(f"  {format_guess(code, probability)}")
