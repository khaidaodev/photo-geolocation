"""
Tries CLIP as a different starting point from the ResNets. CLIP was trained on hundreds of
millions of photos paired with captions, and captions often say where a photo was taken, so its
features (the list of numbers it turns each photo into) might carry more about geography than
the ImageNet features the ResNets start from.

CLIP stays frozen here. Every photo goes through it once, the 512 numbers per photo are saved
to disk, and a plain logistic regression is trained on top to guess the country. This is called
a linear probe (training only a simple layer on top of frozen features). The regularisation
strength C (how hard the classifier is stopped from over-fitting) is picked using the valid
split, and the test split is only used once, at the end.

Run it with:
    python src/clip_probe.py
"""

from pathlib import Path

import numpy as np
import open_clip
import torch
import torchvision
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from top_k_accuracy import top_k_correct

ROOT = Path(__file__).resolve().parent.parent
COUNTRY211_DIR = ROOT / "data" / "raw" / "country211"
FEATURES_DIR = ROOT / "results" / "clip_features"

CLIP_MODEL = "ViT-B-32-quickgelu"
C_VALUES = [1.0, 10.0, 100.0]


def extract_clip_features(model, loader):
    """Runs every photo through frozen CLIP and returns its features, scaled so each photo's
    list of numbers has length 1 (the usual way to use CLIP features), plus the true labels."""
    features, labels = [], []
    with torch.no_grad():
        for batch_number, (images, targets) in enumerate(loader, start=1):
            encoded = model.encode_image(images)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            features.append(encoded.numpy())
            labels.append(targets.numpy())
            if batch_number % 50 == 0:
                print(f"  {batch_number * loader.batch_size} of {len(loader.dataset)} photos done")
    return np.concatenate(features), np.concatenate(labels)


def cached_features(path, extract):
    """Loads saved features from path if they exist, otherwise calls extract() to make them,
    saves them, and returns them. Means the slow part only ever runs once."""
    if path.exists():
        saved = np.load(path)
        return saved["features"], saved["labels"]
    features, labels = extract()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=features, labels=labels)
    return features, labels


def top_k_scores(probabilities, labels, ks=(1, 3, 5)):
    """For each k, the fraction of photos where the real country is in the classifier's k most
    likely countries. Reuses top_k_correct from top_k_accuracy.py."""
    scores = torch.from_numpy(probabilities)
    targets = torch.from_numpy(labels)
    return {k: top_k_correct(scores, targets, k) / len(labels) for k in ks}


if __name__ == "__main__":
    print(f"Loading CLIP ({CLIP_MODEL}, frozen)...")
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained="openai")
    model.eval()

    data = {}
    class_lists = []
    for split in ("train", "valid", "test"):
        dataset = torchvision.datasets.ImageFolder(str(COUNTRY211_DIR / split), transform=preprocess)
        class_lists.append(dataset.classes)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        path = FEATURES_DIR / f"{len(dataset.classes)}_countries" / f"{split}.npz"
        print(f"Getting CLIP features for the {split} split ({len(dataset)} photos)...")
        data[split] = cached_features(path, lambda: extract_clip_features(model, loader))
    assert class_lists[0] == class_lists[1] == class_lists[2], "Splits have different countries"

    X_train, y_train = data["train"]
    X_valid, y_valid = data["valid"]
    X_test, y_test = data["test"]

    best_c, best_valid, best_clf = None, -1.0, None
    for c in C_VALUES:
        print(f"Training the classifier with C={c}...")
        clf = LogisticRegression(C=c, max_iter=1000)
        clf.fit(X_train, y_train)
        valid_acc = clf.score(X_valid, y_valid)
        print(f"  C={c}: valid acc {valid_acc:.1%}")
        if valid_acc > best_valid:
            best_c, best_valid, best_clf = c, valid_acc, clf

    assert list(best_clf.classes_) == list(range(len(class_lists[0])))
    results = top_k_scores(best_clf.predict_proba(X_test), y_test)
    print(f"Best C was {best_c} (valid acc {best_valid:.1%}). Test split, never used until now:")
    for k, accuracy in results.items():
        print(f"Right country in the top {k} guesses: {accuracy:.1%}")
    n = len(class_lists[0])
    print(f"Random guessing across {n} countries would get {1 / n:.1%} for top 1 and {3 / n:.1%} for top 3.")
