# Photo Geolocation

Guesses which country a photo was taken in, just from what's in the picture. No GPS metadata,
no filename hints, just pixels.

## Why this problem

Basically what GeoGuessr is built on, road markings, plants, architecture, which side of the
road cars drive on, even power line poles, all give away roughly where a photo's from. It's a
genuinely hard visual problem, way more subtle than "spot the object in the frame."

## Where it's at right now

Data pipeline's built, the 20-country problem is fully solved and understood (17.1% ceiling),
and the 50-country version is settled too (13.0% valid, 12.1% on the untouched test split). The
question of how much of the network to unfreeze gave the same answer at both sizes. See
"Results so far" for the full breakdown.

Dataset is Country211, built by OpenAI to test CLIP: 63,000 geotagged Flickr photos, balanced
across 211 countries (150 train / 50 valid / 100 test each), about 11GB total.

211-way classification is brutal, even CLIP struggles with it. First pass stuck to 20 well
known, visually distinct countries (see `STARTER_COUNTRIES` in `src/data_loading.py`), now
scaling up towards the full 211.

## Results so far

**Baseline:** a frozen pretrained ResNet18 turns each photo into a 512-number summary, then a
logistic regression trained on top guesses the country. 10.6% validation accuracy, vs 5% for
random guessing (20 countries).

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
quick first look):** 16.5% valid at epoch 14, still climbing when the run hit its cap. Left
genuinely inconclusive at the time.

**Fine-tuning, attempt 12 (ResNet50, layer4-only, 20 epochs, with model saving added):** 16.7%
valid at epoch 17. Broadly consistent with the confirmed 17.1% ceiling from attempt 10. Best
model saved to `models/best_model.pt`.

**Final honest test-set result (20 countries):** ran the saved model against the untouched test
split, never used anywhere else in this project. 16.0% accuracy, close to the 16.7% validation
number from the same run. A confusion matrix surfaced the most common mistakes: Thailand
guessed as South Korea, Japan guessed as South Korea, Australia guessed as South Africa, India
and the US both sometimes guessed as South Africa, all genuine regional visual overlap rather
than random error.

**Fine-tuning, attempt 13 (ResNet50, layer3+layer4 unfrozen, properly uncapped, up to 60
epochs):** settles attempt 11 for good. Early stopping triggered at epoch 32, best epoch 24,
17.0% valid, essentially tied with the confirmed 17.1% layer4-only ceiling. Left running
overnight since 5+ hours was too long to sit and wait for. Layer4-only stays the simpler choice
since it trains faster for the same result, no reason to unfreeze more of the network here.

The 20-country problem is fully solved and understood at this point: ResNet50, layer4-only,
around 17% valid, 16% on genuinely unseen test data.

**Fine-tuning, attempt 14 (scaling up to 50 countries, same confirmed 20-country setup, no
new hyperparameters guessed on top of a bigger dataset at the same time):** early stopping
triggered at epoch 31, best epoch 23, 13.0% valid. Down from the 17.1% ceiling on 20 countries,
but still 6.5x better than random guessing across 50 countries (2.0%). This is the expected,
sensible trade-off of a genuinely harder problem, not a broken setup, more countries to tell
apart means more ways to be wrong. Took 9 hours 30 minutes, nearly double the 20-country run,
matching the roughly 2.5x more training photos overall.

**Fine-tuning, attempt 15 (ResNet50, layer3+layer4 unfrozen, 50 countries):** re-tested the
layer3 question at the harder 50-country size, since a tie on an easier problem doesn't
automatically carry over to a bigger one. It did. Early stopping triggered at epoch 32, best
epoch 24, 13.0% valid, exactly the same as layer4-only. Took 12 hours 19 minutes, the longest
run so far, for no gain at all. Layer4-only stays the setup to use at both sizes.

**Final honest test-set result (50 countries):** ran the saved model (the one from attempt 15,
which tied layer4-only at 13.0%) against the untouched test split of 5,000 photos. 12.1%
accuracy, close to the 13.0% validation number, so the model isn't quietly overfit to the
validation photos. About 6x better than random guessing (2.0%). Took about 7 minutes. The most
common mistakes were Kenya guessed as South Africa (19 times) and South Africa guessed as Kenya
(12 times), Kenya guessed as Nigeria (10), China guessed as Vietnam (10) and Malaysia guessed as
Singapore (10). These are countries that look alike, neighbours or the same part of the world,
so the model is getting confused the way a person would rather than guessing blindly.

Same answer at 20 and at 50 countries: unfreezing more of the network doesn't help here, it
just makes training slower. The ceiling for this approach on 50 countries is about 13% valid,
12% on unseen test photos.

**Fine-tuning, attempt 16 (scaling up further to 75 countries, same confirmed setup):** early
stopping triggered at epoch 38, best epoch 30, 11.5% valid. Continues the same honest trend,
17.1% at 20 countries, 13.0% at 50, 11.5% at 75, each one a real, expected trade-off for a
genuinely harder problem. Still about 8.8x better than random guessing across 75 countries
(1.3%). Took 20 hours 53 minutes, the longest run yet.

**Final honest test-set result (75 countries):** ran the saved model against the untouched test
split of 7,500 photos. 11.7% accuracy, matching the 11.5% validation number closely, so the
model isn't quietly overfit to validation. Top-3 accuracy (the right country somewhere in the
3 guesses predict.py shows) was 21.8%, top-5 was 29.1%, both around 5-7x better than random
guessing (1.3% top-1, 4.0% top-3). The confusion matrix showed South Africa and Kenya still
confusing each other in both directions, plus Jordan mistaken for Saudi Arabia and Bangladesh
mistaken for India, both look-alike neighbouring pairs. China mistaken for Qatar was the one
pairing without an obvious explanation, reported honestly rather than guessed at.

**Fine-tuning, attempt 17 (scaling up to 100 countries, same confirmed setup):** early stopping
triggered at epoch 37, best epoch 29, 11.5% valid, essentially flat against the 11.5% at 75
countries rather than continuing the earlier drop (17.1% at 20, 13.0% at 50, 11.5% at 75). Still
11.5x better than random guessing at this size (1.0%). Took 25 hours 51 minutes, the longest run
yet. Whether this is a genuine plateau for this exact setup, or just how these particular 25
added countries happened to land, isn't clear from one run, reported honestly rather than
explained away.

**Final honest test-set result (100 countries):** ran the saved model against the untouched test
split of 10,000 photos. 11.4% accuracy, close to the 11.5% validation number, so the plateau
seen at validation holds up on genuinely unseen photos too, not just a validation quirk. Top-3
accuracy was 20.8%, top-5 was 27.6%, both around 7x better than random guessing (1.0% top-1,
3.0% top-3). The confusion matrix showed Jordan still mistaken for Saudi Arabia (a repeat from
75 countries), plus new pairs: Ecuador and Costa Rica, South Africa and Tanzania, Croatia and
Greece, all neighbouring or visually similar regions. Nepal mistaken for Peru was the one
pairing without an obvious shared explanation, reported honestly rather than guessed at.

Next real step is either trying a different pretrained model, or continuing to scale further
towards all 211 countries.

## Try it on your own photo

`src/predict.py` loads the best saved model and predicts the top 3 most likely countries for
any photo you give it, with a genuine confidence percentage for each (not just a raw guess).
Prints the full country name next to its code (Japan (JP)), not just the code on its own.

**How good are those top 3 guesses, really?** Measured properly against the untouched test
split (50 countries): the right country was the single best guess 12.1% of the time, somewhere
in the top 3 guesses 23.4% of the time, and in the top 5 guesses 31.2% of the time. Random
guessing across 50 countries would get 2.0% for the top guess and 6.0% for the top 3, so the
model's three guesses do genuinely capture the right answer far more often than one guess
alone, even when its single best pick is wrong. Fits the earlier confusion matrix, the right
answer is often sitting just behind a look-alike country the model picked first. Took 9 minutes
28 seconds on the test split's 5,000 photos.

```bash
python3 src/predict.py path/to/your/photo.jpg
```

Only works on JPG or PNG, not HEIC (the default iPhone format), convert first if needed:
```bash
sips -s format jpeg photo.HEIC --out photo.jpg
```

Since the model only learns outdoor, geographic clues (landscapes, architecture, road signs), a
photo with nothing geographic in it will come back with low, scattered confidence rather than a
strong guess, that's expected behaviour, not a bug.

## How to run this yourself

```bash
pip3 install -r requirements.txt
python3 src/data_loading.py
python3 src/baseline_model.py
python3 src/finetune_model.py
python3 src/predict.py path/to/your/photo.jpg
python3 src/confusion_matrix.py
```

First script downloads Country211 (~11GB, one-time). The archive stays on disk after
extraction, so pulling out more countries later doesn't mean re-downloading.

Third script trains the model on whatever countries have actually been extracted, and saves
the best version to `models/best_model.pt` as it trains. Stops itself automatically once
validation accuracy plateaus. With ResNet50 and more countries, expect real training runs to
take many hours, sometimes overnight, on a CPU.

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
