"""
Fine-tuning pass on top of the frozen baseline, still just the 20 starter countries.

Attempt 1 overfit badly (99.3% train, 13.0% valid). Attempt 2 added augmentation, fixing the
overfitting gap short-term but valid accuracy stayed flat. Attempt 3 trained for longer (15
epochs) and found the model eventually overfits anyway by epoch 10+, plateauing at 13.6% valid.
This attempt lowers the learning rate 10x (from 1e-4 to 1e-5), so the unfrozen layer adjusts in
smaller, more careful steps instead of big ones that let it overshoot into memorising too fast.

Run it with:
    python src/finetune_model.py
"""

import ssl
import certifi

from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

ROOT = Path(__file__).resolve().parent.parent
COUNTRY211_DIR = ROOT / "data" / "raw" / "country211"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LEARNING_RATE = 1e-5  # was 1e-4 in the previous attempt

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_model(num_classes: int) -> nn.Module:
    """Pretrained ResNet18 with everything frozen except the last block (layer4) and a fresh
    final layer sized for our number of countries."""
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer=None):
    """Runs one pass over a dataset. Trains if given an optimizer, otherwise just evaluates.
    Returns the average loss and accuracy for that pass."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def best_epoch(valid_accuracies: list[float]) -> tuple[int, float]:
    """Given a list of validation accuracies (one per epoch, in order), returns the 1-indexed
    epoch number and accuracy of whichever epoch did best."""
    best_index = max(range(len(valid_accuracies)), key=lambda i: valid_accuracies[i])
    return best_index + 1, valid_accuracies[best_index]


if __name__ == "__main__":
    print("Loading train/valid splits (20 starter countries)...")
    train_data = torchvision.datasets.ImageFolder(str(COUNTRY211_DIR / "train"), transform=TRAIN_TRANSFORM)
    valid_data = torchvision.datasets.ImageFolder(str(COUNTRY211_DIR / "valid"), transform=EVAL_TRANSFORM)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

    print(f"Building model (ResNet18, last block unfrozen, lr={LEARNING_RATE})...")
    model = build_model(num_classes=len(train_data.classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    epochs = 15
    valid_accuracies = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        valid_loss, valid_acc = run_epoch(model, valid_loader, criterion)
        valid_accuracies.append(valid_acc)
        print(f"Epoch {epoch}/{epochs}: train acc {train_acc:.1%}, valid acc {valid_acc:.1%}")

    epoch_num, best_acc = best_epoch(valid_accuracies)
    print(f"Best epoch: {epoch_num}/{epochs}, valid acc {best_acc:.1%}")
    print("Previous attempt (lr=1e-4, 15 epochs): 13.6% valid. Baseline: 10.6% valid.")
