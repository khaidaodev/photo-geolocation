"""
First baseline for the photo-geolocation model, using the 20 starter countries.

Training a CNN from scratch isn't realistic with only 150 photos per country. This uses a
pretrained, frozen ResNet18 to turn each photo into a 512-number summary, then trains a plain
logistic regression on top to guess the country. Basically borrowing a model's existing "eyes"
instead of training a new pair from nothing (transfer learning).

Run it with:
    python src/baseline_model.py
"""

import ssl
import certifi

from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_loading import STARTER_COUNTRIES

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

ROOT = Path(__file__).resolve().parent.parent
COUNTRY211_DIR = ROOT / "data" / "raw" / "country211"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_split(split: str) -> Subset:
    """Loads one split, filtered to just the 20 starter countries."""
    dataset = torchvision.datasets.ImageFolder(str(COUNTRY211_DIR / split), transform=TRANSFORM)
    keep = {dataset.class_to_idx[c] for c in STARTER_COUNTRIES if c in dataset.class_to_idx}
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label in keep]
    return Subset(dataset, indices)


def build_feature_extractor() -> torch.nn.Module:
    """Pretrained ResNet18 with its last layer stripped off, so it outputs a 512-number
    summary instead of a class prediction."""
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    return model


def extract_features(dataset: Subset, model: torch.nn.Module, batch_size: int = 32):
    """Runs every photo through the frozen model, returns the feature vectors and labels."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            features.append(model(images))
            labels.append(targets)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


if __name__ == "__main__":
    print("Loading pretrained ResNet18 (frozen feature extractor)...")
    feature_extractor = build_feature_extractor()

    print("Loading train/valid splits (20 starter countries only)...")
    train_set = load_split("train")
    valid_set = load_split("valid")

    print(f"Extracting features for {len(train_set)} training photos...")
    X_train, y_train = extract_features(train_set, feature_extractor)

    print(f"Extracting features for {len(valid_set)} validation photos...")
    X_valid, y_valid = extract_features(valid_set, feature_extractor)

    print("Training logistic regression classifier on top of the features...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    accuracy = accuracy_score(y_valid, clf.predict(X_valid))
    print(f"Validation accuracy: {accuracy:.1%}")
    print(f"(random guessing across 20 countries would be {1 / 20:.1%})")
