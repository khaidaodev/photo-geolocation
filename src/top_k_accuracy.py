"""
Checks how often the right country shows up anywhere in the model's top 1, top 3 and top 5
guesses, not just its single best guess. Since predict.py shows three guesses per photo, this
measures how well those three guesses actually do.

Uses the test split, which was never used in training or for picking the best epoch.

Run it with:
    python src/top_k_accuracy.py
"""

import torch
import torchvision
from torch.utils.data import DataLoader

from confusion_matrix import COUNTRY211_DIR, EVAL_TRANSFORM, MODEL_PATH, load_trained_model


def top_k_correct(scores, labels, k):
    """Counts how many photos in a batch had the real country somewhere in the model's k
    highest scoring guesses. scores has one row per photo and one column per country."""
    top_k_guesses = scores.topk(k, dim=1).indices
    return (top_k_guesses == labels.unsqueeze(1)).any(dim=1).sum().item()


def top_k_accuracy(model, loader, ks=(1, 3, 5)):
    """Runs every photo through the model once and returns, for each k, the fraction of photos
    where the real country was in the top k guesses."""
    correct = {k: 0 for k in ks}
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            scores = model(images)
            for k in ks:
                correct[k] += top_k_correct(scores, labels, k)
            total += labels.size(0)
    return {k: correct[k] / total for k in ks}


if __name__ == "__main__":
    print("Loading test split (never used in training or picking the best epoch)...")
    test_data = torchvision.datasets.ImageFolder(str(COUNTRY211_DIR / "test"), transform=EVAL_TRANSFORM)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    print("Loading best saved model...")
    model, class_names = load_trained_model(MODEL_PATH)

    print(f"Running {len(test_data)} test photos through the model...")
    results = top_k_accuracy(model, test_loader, ks=(1, 3, 5))

    for k, accuracy in results.items():
        print(f"Right country in the top {k} guesses: {accuracy:.1%}")
    print(f"(random guessing would get {1 / len(class_names):.1%} for top 1 and "
          f"{3 / len(class_names):.1%} for top 3)")
