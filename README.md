# Photo Geolocation

Guesses which country a photo was taken in, just from what's in the picture. No GPS metadata,
no filename hints, just pixels.

## Why this problem

Basically what GeoGuessr is built on, road markings, plants, architecture, which side of the
road cars drive on, even power line poles, all give away roughly where a photo's from. It's a
genuinely hard visual problem, way more subtle than "spot the object in the frame."

## Where it's at right now

Data pipeline's built, baseline model's done, and two fine-tuning attempts so far, one that
overfit badly, one that fixed the overfitting but didn't yet improve real accuracy. See
"Results so far" below for the honest breakdown.

Dataset is Country211, built by OpenAI to test CLIP: 63,000 geotagged Flickr photos, balanced
across 211 countries (150 train / 50 valid / 100 test each), about 11GB total.

211-way classification is brutal, even CLIP struggles with it. First pass sticks to 20 well
known, visually distinct countries (see `STARTER_COUNTRIES` in `src/data_loading.py`), then
scales up once that's actually working.

## Results so far

**Baseline:** a frozen pretrained ResNet18 turns each photo into a 512-number summary, then a
logistic regression trained on top guesses the country. 10.6% validation accuracy on the 20
starter countries, vs 5% for random guessing.

**Fine-tuning, attempt 1:** unfroze the last block of the ResNet and trained it directly for 5
passes. Training accuracy hit 99.3%, but validation accuracy barely moved, 13.0%. Classic
overfitting, the model memorised the exact training photos instead of learning anything general.

**Fine-tuning, attempt 2 (with augmentation):** added random crops, flips, and colour jitter to
the training photos only, so the model can't just memorise them. Training accuracy dropped to
73.3% (harder training data, expected), and the train/valid gap shrank a lot. But validation
accuracy still only reached 13.9%, barely above attempt 1. So this fixed the overfitting
symptom, but hasn't yet turned into better real-world accuracy. Most likely needs more training
epochs now that the data's genuinely harder to learn from, or a less aggressive augmentation.

## How to run this yourself

```bash
pip3 install -r requirements.txt
python3 src/data_loading.py
python3 src/baseline_model.py
python3 src/finetune_model.py
```

First script downloads Country211 (~11GB, one-time). Worth only extracting the 20 starter
countries at first, the archive stays on disk so extracting more later doesn't mean
re-downloading.

Second and third scripts train and evaluate the baseline and fine-tuned models on whatever
countries have been extracted.

## Testing and git

`tests/` covers the bits that don't need the full dataset, country code lookups, folder
scanning, and the fine-tuning model's layer-freezing and augmentation setup. Will add more as
the modelling side grows.

## Tools used

Python, PyTorch/torchvision, scikit-learn, pycountry.

## Where the data's from

[Country211](https://github.com/openai/CLIP/blob/main/data/country211.md), built by OpenAI from
YFCC100M geotagged Flickr photos. Used here under the same terms, the photos themselves are
whatever Creative Commons licence their original uploaders picked.
