"""
Guesses which country a photo was taken in using CLIP plus a trained classifier, the best setup
so far (25.9% on all 211 countries). It's the CLIP version of predict.py.

Two steps, run in this order:
    python src/clip_predict.py train                  (once, makes and saves the classifier)
    python src/clip_predict.py path/to/photo.jpg      (guesses a photo)

The train step needs the CLIP photo numbers that clip_probe.py saves, so run that first. The
saved classifier goes in models/, which git ignores, like the other model files.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

from clip_per_country import BEST_C
from clip_probe import CLIP_MODEL, COUNTRY211_DIR, FEATURES_DIR
from clip_zero_shot import readable_name

ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_PATH = ROOT / "models" / "clip_classifier.joblib"


def save_classifier(clf, codes, path):
    """Saves the trained classifier together with the country codes in the order it knows
    them, so a guess can be turned back into a country later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf, "codes": codes}, path)


def load_classifier(path):
    """Loads a saved classifier and its country codes. Refuses if the number of countries
    in the classifier doesn't match the codes, since the guesses would be mislabelled."""
    saved = joblib.load(path)
    clf, codes = saved["classifier"], saved["codes"]
    if len(clf.classes_) != len(codes):
        raise ValueError(f"Classifier knows {len(clf.classes_)} countries but {len(codes)} codes were saved")
    return clf, codes


def top_guesses(probabilities, codes, k=3):
    """Returns the k most likely countries as (code, probability) pairs, most likely first."""
    order = np.argsort(probabilities)[::-1][:k]
    return [(codes[i], float(probabilities[i])) for i in order]


def format_guess(code, probability):
    """Turns one guess into a readable line like 'Japan (JP): 25.6%'."""
    return f"{readable_name(code)} ({code}): {probability:.1%}"


def photo_features(model, preprocess, image_path):
    """Runs one photo through frozen CLIP and returns its 512 numbers, scaled to length 1 the
    same way clip_probe.py scaled them, as a grid with one row."""
    image = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        encoded = model.encode_image(preprocess(image).unsqueeze(0))
        encoded = encoded / encoded.norm(dim=-1, keepdim=True)
    return encoded.numpy()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/clip_predict.py train   or   python src/clip_predict.py path/to/photo.jpg")
        sys.exit(1)

    if sys.argv[1] == "train":
        codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
        folder = FEATURES_DIR / f"{len(codes)}_countries"
        train, test = np.load(folder / "train.npz"), np.load(folder / "test.npz")
        print(f"Training the classifier on {len(train['labels'])} saved photo numbers (C={BEST_C})...")
        clf = LogisticRegression(C=BEST_C, max_iter=1000)
        clf.fit(train["features"], train["labels"])
        save_classifier(clf, codes, CLASSIFIER_PATH)
        print(f"Saved to {CLASSIFIER_PATH}")
        print(f"Test accuracy of the saved classifier: {clf.score(test['features'], test['labels']):.1%} "
              f"(clip_probe.py got 25.9%, so these should match)")
    else:
        image_path = Path(sys.argv[1])
        clf, codes = load_classifier(CLASSIFIER_PATH)
        model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained="openai")
        model.eval()
        probabilities = clf.predict_proba(photo_features(model, preprocess, image_path))[0]
        print(f"Top guesses for {image_path.name}:")
        for code, probability in top_guesses(probabilities, codes):
            print(f"  {format_guess(code, probability)}")
