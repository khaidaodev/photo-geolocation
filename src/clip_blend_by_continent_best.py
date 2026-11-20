"""
Checks whether the sentence blend helps the same everywhere. It scores the test photos three
ways (trained classifier only, sentences only, and the 0.2 blend) and prints the top-1 country
accuracy for each true continent, so any continent where the sentences help a lot, or hurt,
shows up.

Run it with:
    python3 src/clip_blend_by_continent_best.py
"""

import numpy as np
import open_clip
from sklearn.linear_model import LogisticRegression

from clip_joined_l14 import join_features
from clip_per_country import BEST_C
from clip_predict_best import BLEND_WEIGHT, blended_log_probs
from clip_probe import COUNTRY211_DIR
from clip_probe_l14 import CLIP_MODEL_L14, FEATURES_DIR_L14
from clip_probe_l14_dfn2b import FEATURES_DIR_L14_DFN2B
from clip_zero_shot import build_prompts, text_features
from continents import continent_of


def accuracy_by_continent(true_codes, predictions_by_setup):
    """Takes the true codes and a dict of setup name -> predicted codes. Gives back a dict of
    continent -> (photos, {setup name: top-1 accuracy among that continent's photos})."""
    true_codes = list(true_codes)
    result = {}
    for continent in sorted({continent_of(code) for code in true_codes}):
        rows = [i for i, code in enumerate(true_codes) if continent_of(code) == continent]
        scores = {
            name: sum(true_codes[i] == predicted[i] for i in rows) / len(rows)
            for name, predicted in predictions_by_setup.items()
        }
        result[continent] = (len(rows), scores)
    return result


if __name__ == "__main__":
    codes = sorted(p.name for p in (COUNTRY211_DIR / "test").iterdir() if p.is_dir())
    folder_name = f"{len(codes)}_countries"
    print(f"Loading CLIP ({CLIP_MODEL_L14}, frozen) for the country sentences...")
    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL_L14, pretrained="openai")
    model.eval()
    sentences = text_features(model, open_clip.get_tokenizer(CLIP_MODEL_L14), build_prompts(codes))

    data = {}
    for split in ("train", "test"):
        first = np.load(FEATURES_DIR_L14 / folder_name / f"{split}.npz")
        second = np.load(FEATURES_DIR_L14_DFN2B / folder_name / f"{split}.npz")
        data[split] = (join_features(first, second), first["labels"])
    X_train, y_train = data["train"]
    X_test, y_test = data["test"]
    assert y_test.max() + 1 == len(codes), "Saved photo numbers and country folders don't match"

    print(f"Training the joined classifier (C={BEST_C})...")
    clf = LogisticRegression(C=BEST_C, max_iter=1000)
    clf.fit(X_train, y_train)

    setups = {"classifier": 0.0, "sentences": 1.0, "blend": BLEND_WEIGHT}
    predictions = {
        name: [codes[i] for i in blended_log_probs(clf, X_test, sentences, weight).argmax(axis=1)]
        for name, weight in setups.items()
    }
    true_codes = [codes[i] for i in y_test]
    for name, predicted in predictions.items():
        overall = np.mean([t == p for t, p in zip(true_codes, predicted)])
        print(f"  {name}: {overall:.1%} overall")

    print("\nTop-1 by true continent (photos, classifier, sentences, blend):")
    table = accuracy_by_continent(true_codes, predictions)
    for continent, (count, scores) in sorted(table.items(), key=lambda item: -item[1][0]):
        print(
            f"  {continent}: {count} photos, {scores['classifier']:.1%}, "
            f"{scores['sentences']:.1%}, {scores['blend']:.1%}"
        )
