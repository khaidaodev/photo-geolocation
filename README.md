# Photo Geolocation

Guesses which country a photo was taken in, just from what's in the picture. No GPS metadata,
no filename hints, just pixels.

## Why this problem

Basically what GeoGuessr is built on, road markings, plants, architecture, which side of the
road cars drive on, even power line poles, all give away roughly where a photo's from. It's a
genuinely hard visual problem, way more subtle than "spot the object in the frame."

## Where it's at right now

Data pipeline's built, baseline model's done, twelve fine-tuning attempts so far, a working
prediction script, and now a proper honest test-set result plus a confusion matrix showing
which countries it actually mixes up. See "Results so far" for the full breakdown.

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

**Fine-tuning, attempt 7 (layer3 and layer4 both unfrozen instead of just layer4, ResNet18):**
slightly worse, 14.0% valid at epoch 12, and it overfit sooner. Reverted to layer4-only.

**Fine-tuning, attempt 8 (ResNet50 instead of ResNet18, capped at 15 epochs as a first look):**
new best, 16.1% valid, still climbing steadily when the run hit its cap.

**Fine-tuning, attempt 9 (ResNet50, uncapped up to 35 epochs):** new best, 16.8% valid at epoch
31, but ran the full 35 epochs without early stopping ever triggering, so still not the true
ceiling.

**Fine-tuning, attempt 10 (ResNet50, cap raised to 60 epochs):** early stopping finally
triggered on its own, stopping at epoch 42 after 8 epochs with no improvement. Real best point:
epoch 34, 17.1% valid, the confirmed plateau for this setup. Took 5 hours 17 minutes on CPU.

**Fine-tuning, attempt 11 (ResNet50, layer3+layer4 both unfrozen, capped at 15 epochs as a
quick first look):** 16.5% valid at epoch 14, still climbing when the run hit its cap, roughly
in line with layer4-only at the same point. Genuinely inconclusive, still an open question.

**Fine-tuning, attempt 12 (ResNet50, layer4-only, 20 epochs, with model saving added):** 16.7%
valid at epoch 17. Broadly consistent with the confirmed 17.1% ceiling from attempt 10, small
run-to-run variation is expected. Took 2 hours 40 minutes. Best model saved to
`models/best_model.pt`.

**Final honest test-set result:** ran the saved model against the test split, the one part of
the dataset never touched anywhere else in this whole project, not for training, not for
picking the best epoch. 16.0% accuracy, close to the 16.7% validation number from the same run,
a good sign the model isn't quietly overfit to validation either. A confusion matrix (a table
showing exactly which countries get mixed up with which) surfaced the most common mistakes: Thailand guessed as South Korea, Japan guessed as South Korea, Australia guessed as South
Africa, India and the US both sometimes guessed as South Africa. These aren't random errors,
they line up with genuine regional visual overlap (similar architecture, signage, or scenery),
suggesting the model's picking up on real patterns rather than guessing blind, even when wrong.

## Try it on your own photo

`src/predict.py` loads the best saved model and predicts the top 3 most likely countries for
any photo you give it, with a genuine confidence percentage for each (not just a raw guess).

```bash
python3 src/predict.py path/to/your/photo.jpg
```

Only works on JPG or PNG, not HEIC (the default iPhone format), convert first if needed:
```bash
sips -s format jpeg photo.HEIC --out photo.jpg
```

Since the model's only trained on the 20 starter countries so far, and only really learns
outdoor, geographic clues (landscapes, architecture, road signs), a photo with nothing
geographic in it will come back with low, scattered confidence rather than a strong guess,
that's expected behaviour, not a bug.

## How to run this yourself

```bash
pip3 install -r requirements.txt
python3 src/data_loading.py
python3 src/baseline_model.py
python3 src/finetune_model.py
python3 src/predict.py path/to/your/photo.jpg
python3 src/confusion_matrix.py
```

First script downloads Country211 (~11GB, one-time). Worth only extracting the 20 starter
countries at first, the archive stays on disk so extracting more later doesn't mean
re-downloading.

Third script trains the model and saves the best version to `models/best_model.pt` as it
trains. Stops itself automatically once validation accuracy plateaus, no need to guess how
long to train for, though with ResNet50 and a high epoch cap, expect it to genuinely take
hours on a CPU.

Last script runs the saved model against the untouched test split for an honest final accuracy
number, plus prints the most commonly confused country pairs.

## Testing and git

`tests/` covers the bits that don't need the full dataset, country code lookups, folder
scanning, both models' feature extraction, layer-freezing, augmentation, learning rate,
best-epoch, early stopping, and checkpoint saving logic, the prediction script's ranking and
model-loading logic, and the confusion matrix's counting and mix-up-finding logic, all using
fake data and fake models rather than needing a real trained model to test against.

## Tools used

Python, PyTorch/torchvision, scikit-learn, pycountry.

## Where the data's from

[Country211](https://github.com/openai/CLIP/blob/main/data/country211.md), built by OpenAI from
YFCC100M geotagged Flickr photos. Used here under the same terms, the photos themselves are
whatever Creative Commons licence their original uploaders picked.
