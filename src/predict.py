"""
Loads the best saved fine-tuned model and predicts which country a single photo was probably
taken in, with a confidence score. This is where all the training work actually gets used on a
real photo, rather than just producing an accuracy number.

Run it with:
    python src/predict.py path/to/your/photo.jpg
"""

import sys
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pt"

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_trained_model(path: Path):
    """Loads a saved checkpoint (model weights plus the country codes it was trained on) and
    rebuilds the exact same ResNet50 architecture used during training, ready to make
    predictions rather than be trained any further."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]

    model = torchvision.models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


def predict_top_k(model, class_names, image_path: Path, k: int = 3):
    """Runs one photo through the model and returns its top k guesses as (country_code,
    confidence) pairs, sorted highest confidence first. Confidence is a genuine 0-100%
    probability from softmax, not just a raw, harder-to-read model score."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = EVAL_TRANSFORM(image).unsqueeze(0)  # add a "batch of 1" dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, k)
    return [(class_names[i], top_probs[j].item()) for j, i in enumerate(top_indices)]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/photo.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    model, class_names = load_trained_model(MODEL_PATH)
    predictions = predict_top_k(model, class_names, image_path)

    print(f"Top guesses for {image_path.name}:")
    for country_code, confidence in predictions:
        print(f"  {country_code}: {confidence:.1%}")
