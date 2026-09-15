# Photo Geolocation

Guesses which country a photo was taken in, just from what's in the picture. No GPS metadata,
no filename hints, just pixels.

## Why this problem

Basically what GeoGuessr is built on, road markings, plants, architecture, which side of the
road cars drive on, even power line poles, all give away roughly where a photo's from. It's a
genuinely hard visual problem, way more subtle than "spot the object in the frame."

## Where it's at right now

Data pipeline's built, baseline model's done, and four fine-tuning attempts so far, each one
narrowing down what's actually going wrong. See "Results so far" for the honest breakdown.

Dataset is Country211, built by OpenAI to test CLIP: 63,000 geotagged Flickr photos, balanced
across 211 countries (150 train / 50 valid / 100 test each), about 11GB total.

211-way classification is brutal, even CLIP struggles with it. First pass sticks to 20 well
known, visually distinct countries (see `STARTER_COUNTRIES` in `src/data_loading.py`), then
scales up once that's actually working.

## Results so far

**Baseline:** a frozen pretrained ResNet18 turns each photo into a 512-number summary, then a
logistic regression trained on top guesses the country. 10.6% validation accuracy, vs 5% for
random guessing.

**Fine-tuning, attempt 1 (5 epochs, no augmentation):** unfroze the last ResNet block, trained
directly. 99.3% train accuracy, 13.0% valid, classic overfitting.

**Fine-tuning, attempt 2 (5 epochs, with augmentation):** added random crops, flips, colour
jitter to training photos. Train accuracy dropped to 73.3%, gap shrank, but valid accuracy still
only 13.9%.

**Fine-tuning, attempt 3 (15 epochs, with augmentation):** trained longer to see if it just
needed time. Valid accuracy plateaued around 12-13.6% the whole way, and train accuracy climbed
back past 99% by epoch 10, model eventually overfits regardless.

**Fine-tuning, attempt 4 (15 epochs, augmentation, learning rate lowered 10x to 1e-5):** train
and valid accuracy climbed together the whole run for the first time, no overfitting gap at all.
Best result yet, 13.9% valid at the final epoch, but still rising when training stopped, meaning
15 epochs wasn't enough this time, the model was learning steadily but needed more time to reach
its actual ceiling.

Next real step is running this same setup for more epochs (maybe 30-40) to see where it actually
levels off, now that the training curve itself looks healthy for the first time.

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
scanning, and the fine-tuning model's layer-freezing, augmentation, learning rate, and
best-epoch logic. Will add more as the modelling side grows.

## Tools used

Python, PyTorch/torchvision, scikit-learn, pycountry.

## Where the data's from

[Country211](https://github.com/openai/CLIP/blob/main/data/country211.md), built by OpenAI from
YFCC100M geotagged Flickr photos. Used here under the same terms, the photos themselves are
whatever Creative Commons licence their original uploaders picked.
