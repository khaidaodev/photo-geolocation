"""
Runs the best saved model against the test set (the one split never used anywhere else in
this project, not for training, not for picking the best epoch) to get an honest final
accuracy, and builds a confusion matrix showing exactly which countries it mixes up with
which, not just how often it's wrong overall.

Run it with:
    python src/confusion_matrix.py
"""

from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
COUNTRY211_DIR = ROOT / "data" / "raw" / "country211"
MODEL_PATH = ROOT / "models" / "best_model.pt"

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_trained_model(path: Path):
    """Loads a saved checkpoint and rebuilds the exact same ResNet50 architecture used during
    training, ready to make predictions rather than be trained any further."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]

    model = torchvision.models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


def get_predictions(model, loader):
    """Runs every photo in a dataset through the model and collects what it actually guessed
    alongside the real, correct answer for each one."""
    true_labels, predicted_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.tolist())
    return true_labels, predicted_labels


def build_confusion_matrix(true_labels, predicted_labels, num_classes):
    """Builds a num_classes x num_classes grid where matrix[actual][guessed] counts how many
    times the model guessed `guessed` when the real answer was `actual`. The diagonal (where
    actual equals guessed) is every correct prediction."""
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for actual, guessed in zip(true_labels, predicted_labels):
        matrix[actual][guessed] += 1
    return matrix


def find_most_confused_pairs(matrix, class_names, top_n=5):
    """Looks at every off-diagonal cell (every case where the model guessed wrong) and
    returns the top_n most common (actual_country, guessed_country, count) mix-ups, sorted
    most frequent first."""
    mix_ups = []
    for actual_index, row in enumerate(matrix):
        for guessed_index, count in enumerate(row):
            if actual_index != guessed_index and count > 0:
                mix_ups.append((class_names[actual_index], class_names[guessed_index], count))
    mix_ups.sort(key=lambda pair: pair[2], reverse=True)
    return mix_ups[:top_n]


if __name__ == "__main__":
    print("Loading test split (never used in training or picking the best epoch)...")
    test_data = torchvision.datasets.ImageFolder(str(COUNTRY211_DIR / "test"), transform=EVAL_TRANSFORM)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    print("Loading best saved model...")
    model, class_names = load_trained_model(MODEL_PATH)

    print(f"Running {len(test_data)} test photos through the model...")
    true_labels, predicted_labels = get_predictions(model, test_loader)

    correct = sum(t == p for t, p in zip(true_labels, predicted_labels))
    accuracy = correct / len(true_labels)
    print(f"Test accuracy (never seen during training): {accuracy:.1%}")

    matrix = build_confusion_matrix(true_labels, predicted_labels, len(class_names))
    top_mix_ups = find_most_confused_pairs(matrix, class_names, top_n=5)

    print("\nMost commonly confused country pairs:")
    for actual, guessed, count in top_mix_ups:
        print(f"  Real {actual}, guessed {guessed}: {count} times")
