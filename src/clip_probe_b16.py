"""Same CLIP probe as clip_probe.py, but with the bigger ViT-B-16 model instead of ViT-B-32.
Everything else is kept the same, so any change in accuracy comes from the model size.
Features go in their own folder so the ViT-B-32 ones are left alone."""

import open_clip
import torchvision
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from clip_probe import C_VALUES, COUNTRY211_DIR, ROOT, cached_features, extract_clip_features, top_k_scores

FEATURES_DIR_B16 = ROOT / "results" / "clip_features_vit_b16"
CLIP_MODEL_B16 = "ViT-B-16-quickgelu"

if __name__ == "__main__":
    print(f"Loading CLIP ({CLIP_MODEL_B16}, frozen)...")
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_B16, pretrained="openai")
    model.eval()

    data = {}
    class_lists = []
    for split in ("train", "valid", "test"):
        dataset = torchvision.datasets.ImageFolder(str(COUNTRY211_DIR / split), transform=preprocess)
        class_lists.append(dataset.classes)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        path = FEATURES_DIR_B16 / f"{len(dataset.classes)}_countries" / f"{split}.npz"
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
