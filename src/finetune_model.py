"""
Fine-tuning pass on top of the frozen baseline, still just the 20 starter countries.

Five earlier attempts (documented in the README) tracked down an overfitting problem: given
enough epochs, the model eventually starts memorising the training photos regardless of
augmentation or a lower learning rate, it just takes longer to happen. Rather than guessing a
fixed epoch count each time, this version adds early stopping: training stops itself once
validation accuracy hasn't improved for a set number of epochs (patience), instead of running
blindly for a fixed length and hoping it lands on a good spot.

Patience is set to 8, based on attempt 5's real data, the best epoch there was 13, and it never
beat that again in the following 22 epochs, so 8 epochs without improvement is a reasonable
signal that training has plateaued.

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

LEARNING_RATE = 1e-5
MAX_EPOCHS = 35
PATIENCE = 8

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


def should_stop_early(valid_accuracies: list[float], patience: int) -> bool:
    """Returns True once it's been more than `patience` epochs since the best validation
    accuracy seen so far, meaning training has plateaued and should stop rather than keep
    running pointlessly (and risking more overfitting the longer it goes on)."""
    if len(valid_accuracies) <= patience:
        return False
    epoch_num, _ = best_epoch(valid_accuracies)
    epochs_since_best = len(valid_accuracies) - epoch_num
    return epochs_since_best >= patience


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

    valid_accuracies = []
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        valid_loss, valid_acc = run_epoch(model, valid_loader, criterion)
        valid_accuracies.append(valid_acc)
        print(f"Epoch {epoch}/{MAX_EPOCHS}: train acc {train_acc:.1%}, valid acc {valid_acc:.1%}")

        if should_stop_early(valid_accuracies, PATIENCE):
            print(f"No improvement for {PATIENCE} epochs, stopping early at epoch {epoch}.")
            break

    epoch_num, best_acc = best_epoch(valid_accuracies)
    print(f"Best epoch: {epoch_num}/{len(valid_accuracies)}, valid acc {best_acc:.1%}")
    print("Previous attempt (35 fixed epochs, same lr): best was epoch 13, 13.7% valid.")
