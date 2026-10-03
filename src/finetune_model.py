"""
Fine-tuning pass on top of the frozen baseline, now with autosave.

Seventeen earlier attempts are documented in the README. This version adds a second save file,
written after every single epoch (not just the best one), holding the model, the optimizer's
internal state, and the full accuracy history so far. If the run gets interrupted (a crash, the
Mac sleeping, the project folder moving mid-run, which just happened), restarting the script
picks training back up from the last completed epoch instead of starting over from epoch 1.

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
MODEL_PATH = ROOT / "models" / "best_model.pt"
RESUME_PATH = ROOT / "models" / "resume_checkpoint.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LEARNING_RATE = 1e-5
MAX_EPOCHS = 60
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
    """Pretrained ResNet101 with everything frozen except the last block (layer4) and a fresh
    final layer sized for our number of countries. Testing whether a deeper model than
    ResNet50 can break past the ~11.5% plateau found at 75 and 100 countries."""
    model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
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
    accuracy seen so far, meaning training has plateaued and should stop."""
    if len(valid_accuracies) <= patience:
        return False
    epoch_num, _ = best_epoch(valid_accuracies)
    epochs_since_best = len(valid_accuracies) - epoch_num
    return epochs_since_best >= patience


def save_checkpoint(model: nn.Module, class_names: list[str], valid_acc: float, path: Path):
    """Saves the model's weights alongside the country codes it was trained on and how well it
    did, so a prediction script can later load it and know both how to use it and how much to
    trust it. This is the best-so-far save, only overwritten when a new best epoch happens."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "valid_accuracy": valid_acc,
    }, path)


def save_resume_checkpoint(model, optimizer, epoch: int, valid_accuracies: list[float],
                            class_names: list[str], path: Path):
    """Saves everything needed to pick training back up from exactly where it left off: which
    epoch just finished, the model's current weights, the optimizer's own internal state, and
    every validation accuracy seen so far. Written after every single epoch, an autosave
    rather than only saving at the very end."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "valid_accuracies": valid_accuracies,
        "class_names": class_names,
    }, path)


def load_resume_checkpoint(path: Path):
    """Loads a resume checkpoint if one exists, returning everything needed to continue
    training. Returns None if there's nothing to resume from, meaning this is a fresh start."""
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    print("Loading train/valid splits (every extracted country)...")
    train_data = torchvision.datasets.ImageFolder(str(COUNTRY211_DIR / "train"), transform=TRAIN_TRANSFORM)
    valid_data = torchvision.datasets.ImageFolder(str(COUNTRY211_DIR / "valid"), transform=EVAL_TRANSFORM)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

    print(f"Building model (ResNet101, last block unfrozen, lr={LEARNING_RATE}, {len(train_data.classes)} classes)...")
    model = build_model(num_classes=len(train_data.classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    start_epoch = 1
    valid_accuracies = []

    resume_data = load_resume_checkpoint(RESUME_PATH)
    if resume_data is not None:
        model.load_state_dict(resume_data["model_state_dict"])
        optimizer.load_state_dict(resume_data["optimizer_state_dict"])
        valid_accuracies = resume_data["valid_accuracies"]
        start_epoch = resume_data["epoch"] + 1
        print(f"Found a resume checkpoint, continuing from epoch {start_epoch} "
              f"({len(valid_accuracies)} epochs already done).")

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        valid_loss, valid_acc = run_epoch(model, valid_loader, criterion)
        valid_accuracies.append(valid_acc)
        print(f"Epoch {epoch}/{MAX_EPOCHS}: train acc {train_acc:.1%}, valid acc {valid_acc:.1%}")

        save_resume_checkpoint(model, optimizer, epoch, valid_accuracies, train_data.classes, RESUME_PATH)

        if valid_acc == max(valid_accuracies):
            save_checkpoint(model, train_data.classes, valid_acc, MODEL_PATH)

        if should_stop_early(valid_accuracies, PATIENCE):
            print(f"No improvement for {PATIENCE} epochs, stopping early at epoch {epoch}.")
            break

    epoch_num, best_acc = best_epoch(valid_accuracies)
    print(f"Best epoch: {epoch_num}/{len(valid_accuracies)}, valid acc {best_acc:.1%}")
    print("ResNet50 ceiling for comparison (75 and 100 countries): ~11.5% valid.")

    if RESUME_PATH.exists():
        RESUME_PATH.unlink()
        print("Training finished cleanly, resume checkpoint removed.")
