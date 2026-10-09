"""
Zero-shot CLIP: guesses each photo's country without training anything on the photos.

CLIP has two halves that were trained together, one that turns a photo into 512 numbers and one
that turns a sentence into 512 numbers, so that a photo and a caption that fits it end up with
similar numbers. Here I write one sentence per country ("a photo taken in Japan"), turn each
sentence into numbers with the text half, and for every photo pick the sentence whose numbers
sit closest to the photo's. The photo numbers are the ones clip_probe.py already saved, so no
photos get processed again. This is called zero-shot (using a model on a task without training
it on any examples of that task).

The sentence wording is fixed up front and not tuned, so a better wording could score higher
and this is closer to a floor than a ceiling.

Run it with:
    python src/clip_zero_shot.py
"""

import numpy as np
import open_clip
import pycountry
import torch

from clip_probe import CLIP_MODEL, COUNTRY211_DIR, FEATURES_DIR, top_k_scores

PROMPT = "a photo taken in {}"


def readable_name(code):
    """Turns a two-letter code like JP into a plain name like Japan. Uses pycountry's short
    common name when it has one, otherwise its normal name, otherwise the code itself."""
    country = pycountry.countries.get(alpha_2=code)
    if country is None:
        return code
    return getattr(country, "common_name", None) or country.name


def build_prompts(codes):
    """Writes one sentence per country code, in the same order as the codes."""
    return [PROMPT.format(readable_name(code)) for code in codes]


def text_features(model, tokenizer, sentences):
    """Turns each sentence into CLIP's 512 numbers, scaled so each list has length 1 (the same
    scaling the photo numbers got), keeping the sentences in the order given."""
    with torch.no_grad():
        encoded = model.encode_text(tokenizer(sentences))
        encoded = encoded / encoded.norm(dim=-1, keepdim=True)
    return encoded.numpy()


if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    prompts = build_prompts(codes)
    print(f"Writing one sentence for each of {len(codes)} countries, for example: {prompts[0]!r}")

    print(f"Loading CLIP ({CLIP_MODEL}, frozen)...")
    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained="openai")
    model.eval()
    sentences = text_features(model, open_clip.get_tokenizer(CLIP_MODEL), prompts)

    saved = np.load(FEATURES_DIR / f"{len(codes)}_countries" / "test.npz")
    photos, labels = saved["features"], saved["labels"]
    assert labels.max() + 1 == len(codes), "Saved photo numbers and country folders don't match"

    results = top_k_scores(photos @ sentences.T, labels)
    print(f"Zero-shot on the {len(labels)} test photos, nothing trained:")
    for k, accuracy in results.items():
        print(f"Right country in the top {k} guesses: {accuracy:.1%}")
    n = len(codes)
    print(f"Random guessing across {n} countries would get {1 / n:.1%} for top 1 and {3 / n:.1%} for top 3.")
