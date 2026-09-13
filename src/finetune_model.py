"""
Fine-tuning pass on top of the frozen baseline, still just the 20 starter countries.

The baseline kept ResNet18 completely frozen and only trained a logistic regression on top.
This time the last block (layer4) plus a fresh final layer get unfrozen and trained directly
on our photos. Most of the network stays general purpose, but the deeper end gets to adjust to
our specific photos instead of staying fully generic.

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

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


if __name__ == "__main__":
    print("Loading train/valid splits (20 starter countries)...")
    train_data = torchvision.datasets.ImageFolder(str(COUNTRY211_DIR / "train"), transform=TRANSFORM)
    valid_data = torchvision.datasets.ImageFolder(str(COUNTRY211_DIR / "valid"), transform=TRANSFORM)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

    print("Building model (ResNet18, last block unfrozen)...")
    model = build_model(num_classes=len(train_data.classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    epochs = 5
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        valid_loss, valid_acc = run_epoch(model, valid_loader, criterion)
        print(f"Epoch {epoch}/{epochs}: train acc {train_acc:.1%}, valid acc {valid_acc:.1%}")

    print("Baseline (frozen, logistic regression) was 10.6% for comparison.")
