# Photo Geolocation

Guesses which country a photo was taken in, just from what's in the picture. No GPS metadata,
no filename hints, just pixels.

## Why this problem

Basically what GeoGuessr is built on, road markings, plants, architecture, which side of the
road cars drive on, even power line poles, all give away roughly where a photo's from. It's a
genuinely hard visual problem, way more subtle than "spot the object in the frame."

## Where it's at right now

Data pipeline's built, baseline model's done, and ten fine-tuning attempts so far. ResNet50 has
now properly found its actual plateau, 17.1% valid. See "Results so far" for the full breakdown.

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
jitter to training photos. Train accuracy dropped to 73.3%, gap shrank, valid stayed at 13.9%.

**Fine-tuning, attempt 3 (15 epochs, with augmentation):** trained longer to see if it just
needed time. Valid plateaued around 12-13.6%, train climbed back past 99% by epoch 10, model
overfits regardless of augmentation given enough epochs.

**Fine-tuning, attempt 4 (15 epochs, learning rate lowered 10x to 1e-5):** train and valid
climbed together for the first time, no overfitting gap, but still rising at epoch 15, 13.9%
valid.

**Fine-tuning, attempt 5 (35 epochs, same lower learning rate):** best result was epoch 13,
13.7% valid, then valid drifted down while train kept climbing to 79.9%, confirming the lower
learning rate only delays overfitting rather than fixing it.

**Fine-tuning, attempt 6 (same setup, with early stopping, patience of 8):** training found its
own best point automatically, stopping at epoch 29. Best result yet, 14.2% valid at epoch 21.

**Fine-tuning, attempt 7 (layer3 and layer4 both unfrozen instead of just layer4):** slightly
worse, 14.0% valid at epoch 12, and it overfit sooner. Reverted to layer4-only.

**Fine-tuning, attempt 8 (ResNet50 instead of ResNet18, capped at 15 epochs as a first look):**
new best, 16.1% valid, still climbing steadily when the run hit its cap.

**Fine-tuning, attempt 9 (ResNet50, uncapped up to 35 epochs):** new best, 16.8% valid at epoch
31, but ran the full 35 epochs without early stopping ever triggering, so still not the true
ceiling.

**Fine-tuning, attempt 10 (ResNet50, cap raised to 60 epochs):** early stopping finally
triggered on its own, stopping at epoch 42 after 8 epochs with no improvement. Real best point:
epoch 34, 17.1% valid, the actual plateau for this setup rather than a run cut short. Took
5 hours 17 minutes on CPU, by far the longest run so far, ResNet50 at this many epochs is
genuinely slow without a GPU.

ResNet50 with layer4 unfrozen, this augmentation, and this learning rate genuinely plateaus
around 17%, a real, found ceiling rather than a guess. That's a solid improvement over the
ResNet18 ceiling of 14.2%, and a long way past the 5% random-guessing floor. Next real step is
either scaling up to more of the 211 countries now that this setup's properly understood, or
trying unfreezing layer3 as well on ResNet50 specifically, since it wasn't tested for this
bigger model yet, only on ResNet18 where it made things worse.

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
countries have been extracted. The fine-tuning script stops itself automatically once
validation accuracy plateaus, no need to guess how long to train for, though with ResNet50 and
a high epoch cap, expect it to genuinely take several hours on a CPU.

## Testing and git

`tests/` covers the bits that don't need the full dataset, country code lookups, folder
scanning, and both models' feature extraction, layer-freezing, augmentation, learning rate,
best-epoch, and early stopping logic. Will add more as the modelling side grows.

## Tools used

Python, PyTorch/torchvision, scikit-learn, pycountry.

## Where the data's from

[Country211](https://github.com/openai/CLIP/blob/main/data/country211.md), built by OpenAI from
YFCC100M geotagged Flickr photos. Used here under the same terms, the photos themselves are
whatever Creative Commons licence their original uploaders picked.
