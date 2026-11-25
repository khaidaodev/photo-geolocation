"""
Checks the saved CLIP features by making some of them again. It takes every 200th photo of the
test split, runs those photos through frozen OpenAI ViT-L-14 a second time, and compares the
new numbers with the saved ones for the same photos. This is the half of the pipeline that
check_repeatable.py doesn't touch. Only the OpenAI features are checked here, not the DFN-2B
ones.

Run it with:
    python3 src/check_features_again.py
"""

import numpy as np
import open_clip
import torchvision
from torch.utils.data import DataLoader, Subset

from clip_probe import COUNTRY211_DIR, extract_clip_features
from clip_probe_l14 import CLIP_MODEL_L14, FEATURES_DIR_L14

EVERY = 200


def compare_features(new, saved):
    """Gives back (biggest difference between any two numbers, lowest cosine similarity between
    a photo's new and saved features). A cosine similarity of 1 means the same direction."""
    new = np.asarray(new, dtype=np.float64)
    saved = np.asarray(saved, dtype=np.float64)
    cosine = (new * saved).sum(axis=1) / (np.linalg.norm(new, axis=1) * np.linalg.norm(saved, axis=1))
    return float(np.abs(new - saved).max()), float(cosine.min())


if __name__ == "__main__":
    print(f"Loading CLIP ({CLIP_MODEL_L14}, OpenAI weights, frozen)...")
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_L14, pretrained="openai")
    model.eval()

    dataset = torchvision.datasets.ImageFolder(str(COUNTRY211_DIR / "test"), transform=preprocess)
    saved = np.load(FEATURES_DIR_L14 / f"{len(dataset.classes)}_countries" / "test.npz")
    indices = list(range(0, len(dataset), EVERY))
    print(f"Making the features again for {len(indices)} test photos...")
    loader = DataLoader(Subset(dataset, indices), batch_size=64, shuffle=False)
    new_features, new_labels = extract_clip_features(model, loader)

    labels_match = bool((new_labels == saved["labels"][indices]).all())
    biggest, lowest_cosine = compare_features(new_features, saved["features"][indices])
    print(f"\n  Labels match the saved ones: {labels_match}")
    print(f"  Biggest difference between any two numbers: {biggest:.2e}")
    print(f"  Lowest cosine similarity: {lowest_cosine:.6f}")
