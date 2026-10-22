"""
Guesses which country a photo was taken in using the bigger ViT-L-14 CLIP model plus a trained
classifier. It's the same as clip_predict.py, but with the setup that scored 40.4% on all 211
countries instead of 25.9%. It reuses the saving, loading and printing functions from
clip_predict.py, and saves its classifier under a different name so the ViT-B-32 one is left alone.

Two steps, run in this order:
    python3 src/clip_predict_l14.py train                  (once, makes and saves the classifier)
    python3 src/clip_predict_l14.py path/to/photo.jpg      (guesses a photo)

The train step needs the photo numbers that clip_probe_l14.py saves, so run that first. The
saved classifier goes in models/, which git ignores, like the other model files.
"""

import sys
from pathlib import Path

import numpy as np
import open_clip
from sklearn.linear_model import LogisticRegression

from clip_per_country import BEST_C
from clip_predict import ROOT, format_guess, load_classifier, photo_features, save_classifier, top_guesses
from clip_probe import COUNTRY211_DIR
from clip_probe_l14 import CLIP_MODEL_L14, FEATURES_DIR_L14

CLASSIFIER_PATH_L14 = ROOT / "models" / "clip_classifier_l14.joblib"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 src/clip_predict_l14.py train   or   python3 src/clip_predict_l14.py path/to/photo.jpg")
        sys.exit(1)

    if sys.argv[1] == "train":
        codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
        folder = FEATURES_DIR_L14 / f"{len(codes)}_countries"
        train, test = np.load(folder / "train.npz"), np.load(folder / "test.npz")
        print(f"Training the classifier on {len(train['labels'])} saved photo numbers (C={BEST_C})...")
        clf = LogisticRegression(C=BEST_C, max_iter=1000)
        clf.fit(train["features"], train["labels"])
        save_classifier(clf, codes, CLASSIFIER_PATH_L14)
        print(f"Saved to {CLASSIFIER_PATH_L14}")
        print(f"Test accuracy of the saved classifier: {clf.score(test['features'], test['labels']):.1%} "
              f"(clip_probe_l14.py got 40.4%, so these should match)")
    else:
        image_path = Path(sys.argv[1])
        clf, codes = load_classifier(CLASSIFIER_PATH_L14)
        model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_L14, pretrained="openai")
        model.eval()
        probabilities = clf.predict_proba(photo_features(model, preprocess, image_path))[0]
        print(f"Top guesses for {image_path.name}:")
        for code, probability in top_guesses(probabilities, codes):
            print(f"  {format_guess(code, probability)}")
