# Photo Geolocation

Guesses which country a photo was taken in, just from what's in the picture. No GPS metadata,
no filename hints, just pixels.

## Why this problem

Basically what GeoGuessr is built on, road markings, plants, architecture, which side of the
road cars drive on, even power line poles, all give away roughly where a photo's from. It's a
genuinely hard visual problem, way more subtle than "spot the object in the frame."

## Where it's at right now

Data pipeline's built, and there's a first baseline model with real numbers below. Next step is
beating that baseline, then scaling past 20 countries.

Dataset is Country211, built by OpenAI to test CLIP: 63,000 geotagged Flickr photos, balanced
across 211 countries (150 train / 50 valid / 100 test each), about 11GB total.

211-way classification is brutal, even CLIP struggles with it. First pass sticks to 20 well
known, visually distinct countries (see `STARTER_COUNTRIES` in `src/data_loading.py`), then
scales up once that's actually working.

## Results so far

First baseline: a frozen pretrained ResNet18 turns each photo into a 512-number summary, then a
logistic regression trained on top guesses the country. Transfer learning, borrowing a model
that already knows how to "see" instead of training one from scratch on a few thousand photos.

**10.6% validation accuracy** on the 20 starter countries, vs 5% for random guessing. Not
amazing, but clearly picking up real signal, which is expected for a frozen model with zero
fine-tuning. Next step is probably fine-tuning the network itself, or a stronger pretrained model.

## How to run this yourself

```bash
pip3 install -r requirements.txt
python3 src/data_loading.py
python3 src/baseline_model.py
```

First script downloads Country211 (~11GB, one-time). Worth only extracting the 20 starter
countries at first, the archive stays on disk so extracting more later doesn't mean
re-downloading.

Second script trains and evaluates the baseline on whatever's been extracted.

## Testing and git

`tests/` covers the bits that don't need the full dataset, country code lookups and folder
scanning mostly. Will add more as the modelling side grows.

## Tools used

Python, PyTorch/torchvision, scikit-learn, pycountry.

## Where the data's from

[Country211](https://github.com/openai/CLIP/blob/main/data/country211.md), built by OpenAI from
YFCC100M geotagged Flickr photos. Used here under the same terms, the photos themselves are
whatever Creative Commons licence their original uploaders picked.

