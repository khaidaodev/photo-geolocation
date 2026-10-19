"""Zero-shot CLIP with the bigger ViT-B-16 model, using the same one sentence per country as
clip_zero_shot.py. Nothing gets trained. The photo numbers are the ones clip_probe_b16.py already
saved, so no photos get processed again. Only the model changes, so any difference from the
ViT-B-32 zero-shot score comes from the model size.

Run it with:
    python3 src/clip_zero_shot_b16.py
"""

import numpy as np
import open_clip

from clip_probe import COUNTRY211_DIR, top_k_scores
from clip_probe_b16 import CLIP_MODEL_B16, FEATURES_DIR_B16
from clip_zero_shot import build_prompts, text_features

if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    prompts = build_prompts(codes)
    print(f"Writing one sentence for each of {len(codes)} countries, for example: {prompts[0]!r}")

    print(f"Loading CLIP ({CLIP_MODEL_B16}, frozen)...")
    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL_B16, pretrained="openai")
    model.eval()
    sentences = text_features(model, open_clip.get_tokenizer(CLIP_MODEL_B16), prompts)

    saved = np.load(FEATURES_DIR_B16 / f"{len(codes)}_countries" / "test.npz")
    photos, labels = saved["features"], saved["labels"]
    assert labels.max() + 1 == len(codes), "Saved photo numbers and country folders don't match"

    results = top_k_scores(photos @ sentences.T, labels)
    print(f"Zero-shot on the {len(labels)} test photos, nothing trained:")
    for k, accuracy in results.items():
        print(f"Right country in the top {k} guesses: {accuracy:.1%}")
    n = len(codes)
    print(f"Random guessing across {n} countries would get {1 / n:.1%} for top 1 and {3 / n:.1%} for top 3.")
