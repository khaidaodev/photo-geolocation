# Photo Geolocation

Guesses which country a photo was taken in, just from what's in the picture. No GPS metadata,
no filename hints, just pixels.

## Why this problem

Basically what GeoGuessr is built on, road markings, plants, architecture, which side of the
road cars drive on, even power line poles, all give away roughly where a photo's from. It's a
genuinely hard visual problem, way more subtle than "spot the object in the frame."

## Where it's at right now

Data pipeline's built, baseline model's done, and three fine-tuning attempts so far, each one
narrowing down what's actually going wrong rather than just guessing. See "Results so far" for
the honest breakdown.

Dataset is Country211, built by OpenAI to test CLIP: 63,000 geotagged Flickr photos, balanced
across 211 countries (150 train / 50 valid / 100 test each), about 11GB total.

211-way classification is brutal, even CLIP struggles with it. First pass sticks to 20 well
known, visually distinct countries (see `STARTER_COUNTRIES` in `src/data_loading.py`), then
scales up once that's actually working.

## Results so far

**Baseline:** a frozen pretrained ResNet18 turns each photo into a 512-number summary, then a
logistic regression trained on top guesses the country. 10.6% validation accuracy on the 20
starter countries, vs 5% for random guessing.

**Fine-tuning, attempt 1 (5 epochs, no augmentation):** unfroze the last block of the ResNet and
trained it directly. Training accuracy hit 99.3%, but validation accuracy barely moved, 13.0%.
Classic overfitting, the model memorised the exact training photos.

**Fine-tuning, attempt 2 (5 epochs, with augmentation):** added random crops, flips, and colour
jitter to training photos only. Training accuracy dropped to 73.3% and the train/valid gap
shrank a lot, but validation accuracy still only reached 13.9%.

**Fine-tuning, attempt 3 (15 epochs, with augmentation):** trained for longer to see if the
model just needed more time. Validation accuracy stayed basically flat the whole way through,
12% to 13.6%, best at epoch 9. Training accuracy climbed back up past 99% by epoch 10 anyway,
so given enough epochs the model eventually overfits regardless of augmentation. More training
time alone isn't the fix.

Next real step is probably unfreezing less of the network (just the final layer, or fewer
ResNet blocks) or lowering the learning rate, rather than just training longer.

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
scanning, and the fine-tuning model's layer-freezing, augmentation, and best-epoch logic. Will
add more as the modelling side grows.

## Tools used

Python, PyTorch/torchvision, scikit-learn, pycountry.

## Where the data's from

[Country211](https://github.com/openai/CLIP/blob/main/data/country211.md), built by OpenAI from
YFCC100M geotagged Flickr photos. Used here under the same terms, the photos themselves are
whatever Creative Commons licence their original uploaders picked.
