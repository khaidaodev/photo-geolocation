"""
Tries a few different sentence wordings for zero-shot CLIP, to see whether the wording matters.

clip_zero_shot.py used one sentence, "a photo taken in {country}", fixed before seeing any score.
Here five wordings are fixed up front (listed in TEMPLATES), plus one that averages all five
(called an ensemble: combining several versions into one). The wording is picked using the valid
split only. The test split is then used once, to score the winner and the original wording side
by side, so the test photos never influence which wording wins.

Like clip_zero_shot.py, nothing is trained and no photos get processed again: the photo numbers
come from the files clip_probe.py saved.

Run it with:
    python3 src/clip_zero_shot_wordings.py
"""

import numpy as np
import open_clip

from clip_probe import CLIP_MODEL, COUNTRY211_DIR, FEATURES_DIR, top_k_scores
from clip_zero_shot import PROMPT, readable_name, text_features

# The original wording stays first, it's the one the earlier result came from.
TEMPLATES = [
    PROMPT,
    "a photo of {}",
    "a photo from {}",
    "a photo of a place in {}",
    "{}",
]
ENSEMBLE_NAME = "all five wordings averaged"


def sentences_for(template, codes):
    """Fills one wording in for every country code, keeping the order of the codes."""
    return [template.format(readable_name(code)) for code in codes]


def ensemble(feature_sets):
    """Averages several wordings' text numbers country by country, then rescales each country's
    row to length 1 so it can be compared with the photo numbers in the same way."""
    average = np.mean(feature_sets, axis=0)
    return average / np.linalg.norm(average, axis=1, keepdims=True)


def pick_best(scores):
    """Given a dictionary of wording name to valid score, returns the name with the highest score.
    If two are tied, the one listed first wins."""
    return max(scores, key=scores.get)


if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    folder = FEATURES_DIR / f"{len(codes)}_countries"
    valid, test = np.load(folder / "valid.npz"), np.load(folder / "test.npz")
    valid_x, valid_y = valid["features"], valid["labels"]
    test_x, test_y = test["features"], test["labels"]
    assert valid_y.max() + 1 == len(codes) and test_y.max() + 1 == len(codes), \
        "Saved photo numbers and country folders don't match"

    print(f"Loading CLIP ({CLIP_MODEL}, frozen)...")
    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained="openai")
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

    candidates = {t: text_features(model, tokenizer, sentences_for(t, codes)) for t in TEMPLATES}
    candidates[ENSEMBLE_NAME] = ensemble(list(candidates.values()))

    print(f"\nWording picked on the {len(valid_y)} valid photos (top 1 accuracy):")
    valid_scores = {}
    for name, sentences in candidates.items():
        valid_scores[name] = top_k_scores(valid_x @ sentences.T, valid_y, ks=(1,))[1]
        print(f"  {name!r}: {valid_scores[name]:.1%}")

    best = pick_best(valid_scores)
    print(f"\nBest on the valid photos: {best!r}")
    print(f"Test split ({len(test_y)} photos), used once, nothing trained:")
    for label, name in (("best wording", best), ("original wording", TEMPLATES[0])):
        results = top_k_scores(test_x @ candidates[name].T, test_y)
        scores = ", ".join(f"top {k}: {accuracy:.1%}" for k, accuracy in results.items())
        print(f"  {label} ({name!r}): {scores}")
    n = len(codes)
    print(f"Random guessing across {n} countries would get {1 / n:.1%} for top 1 and {3 / n:.1%} for top 3.")
