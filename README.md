# Photo Geolocation

Guesses which country a photo was taken in, just from what's in the picture. No GPS metadata,
no filename hints, just pixels.

## Why this problem

Basically what GeoGuessr is built on, road markings, plants, architecture, which side of the
road cars drive on, even power line poles, all give away roughly where a photo's from. It's a
really hard visual problem, way more subtle than "spot the object in the frame."

## Where it's at right now

The best I have is 43.5% top 1, 61.9% top 3 and 69.4% top 5 on the 21,100 untouched test photos,
across all 211 countries. Random guessing gets 0.5% for top 1 and 1.4% for top 3. It comes from
CLIP, a model OpenAI trained on photos paired with captions, kept frozen with no fine-tuning, and a
plain logistic regression on top. The setup joins the photo numbers from two versions of ViT-L-14
(OpenAI's and DFN-2B's) and mixes in the zero-shot country sentence scores at weight 0.2, which I
picked on the valid photos only.

The ResNets I fine-tuned first stopped at about 11.4% on 100 countries, however I changed the setup.
Swapping in frozen CLIP took the same 100 countries to 26.1% in 6 minutes 25 seconds, and all 211
countries to 25.9%. After that it was bigger CLIP models and joining them. Not every later
experiment helped (the LAION-2B weights scored lower and a third joined set added nothing), and I've
written those up in "Results so far" the same as the ones that worked.

| Setup | Countries | Top 1 | Top 3 | Top 5 |
|---|---|---|---|---|
| ResNet50, fine-tuned | 100 | 11.4% | 20.8% | 27.6% |
| ResNet101, fine-tuned | 100 | 11.3% | 20.9% | 27.7% |
| CLIP ViT-B-32, trained probe | 211 | 25.9% | 41.9% | 50.1% |
| CLIP ViT-B-16, trained probe | 211 | 31.4% | 48.5% | 57.0% |
| CLIP ViT-L-14 (OpenAI), no training | 211 | 29.5% | 48.3% | 57.2% |
| CLIP ViT-L-14 (OpenAI), trained probe | 211 | 40.4% | 58.6% | 66.9% |
| OpenAI and DFN-2B ViT-L-14 joined | 211 | 42.3% | 60.0% | 67.8% |
| Joined, plus sentence scores (best) | 211 | 43.5% | 61.9% | 69.4% |

The ResNets were only tested on 100 countries, so they aren't a like for like comparison with the
211-country CLIP rows. Everything is one run each. When I resampled the test photos, the best setup
beat OpenAI's ViT-L-14 alone by 2.6 to 3.5 points every time (see "Checking how much luck moves the
score" in "Results so far"). I haven't run that check on the smaller steps, joining on its own (1.9
points) or adding the sentences (1.2 points). Country211 was built by OpenAI to test CLIP, and the
CLIP weights I used were trained on web data, so I haven't ruled out some of the test photos being
in their training data.

Dataset is Country211, built by OpenAI to test CLIP: 63,300 geotagged Flickr photos, balanced across
211 countries (150 train / 50 valid / 100 test each), about 11GB total. The ResNet work started on
20 well known, visually distinct countries (see `STARTER_COUNTRIES` in `src/data_loading.py`) and
scaled up to 100. The CLIP work uses all 211.

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
inconclusive at the time.

**Fine-tuning, attempt 12 (ResNet50, layer4-only, 20 epochs, with model saving added):** 16.7%
valid at epoch 17. Broadly consistent with the confirmed 17.1% ceiling from attempt 10. Best
model saved to `models/best_model.pt`.

**Final test-set result (20 countries):** ran the saved model against the untouched test
split, never used anywhere else in this project. 16.0% accuracy, close to the 16.7% validation
number from the same run. A confusion matrix surfaced the most common mistakes: Thailand
guessed as South Korea, Japan guessed as South Korea, Australia guessed as South Africa, India
and the US both sometimes guessed as South Africa, all regional visual overlap rather
than random error.

**Fine-tuning, attempt 13 (ResNet50, layer3+layer4 unfrozen, properly uncapped, up to 60
epochs):** settles attempt 11 for good. Early stopping triggered at epoch 32, best epoch 24,
17.0% valid, essentially tied with the confirmed 17.1% layer4-only ceiling. Left running
overnight since 5+ hours was too long to sit and wait for. Layer4-only stays the simpler choice
since it trains faster for the same result, no reason to unfreeze more of the network here.

The 20-country problem is fully solved and understood at this point: ResNet50, layer4-only,
around 17% valid, 16% on unseen test data.

**Fine-tuning, attempt 14 (scaling up to 50 countries, same confirmed 20-country setup, no
new hyperparameters guessed on top of a bigger dataset at the same time):** early stopping
triggered at epoch 31, best epoch 23, 13.0% valid. Down from the 17.1% ceiling on 20 countries,
but still 6.5x better than random guessing across 50 countries (2.0%). This is the expected,
sensible trade-off of a harder problem, not a broken setup, more countries to tell
apart means more ways to be wrong. Took 9 hours 30 minutes, nearly double the 20-country run,
matching the roughly 2.5x more training photos overall.

**Fine-tuning, attempt 15 (ResNet50, layer3+layer4 unfrozen, 50 countries):** re-tested the
layer3 question at the harder 50-country size, since a tie on an easier problem doesn't
automatically carry over to a bigger one. It did. Early stopping triggered at epoch 32, best
epoch 24, 13.0% valid, exactly the same as layer4-only. Took 12 hours 19 minutes, the longest
run so far, for no gain at all. Layer4-only stays the setup to use at both sizes.

**Final test-set result (50 countries):** ran the saved model (the one from attempt 15,
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
stopping triggered at epoch 38, best epoch 30, 11.5% valid. Same trend again,
17.1% at 20 countries, 13.0% at 50, 11.5% at 75, each one an expected trade-off for a
harder problem. Still about 8.8x better than random guessing across 75 countries
(1.3%). Took 20 hours 53 minutes, the longest run yet.

**Final test-set result (75 countries):** ran the saved model against the untouched test
split of 7,500 photos. 11.7% accuracy, matching the 11.5% validation number closely, so the
model isn't quietly overfit to validation. Top-3 accuracy (the right country somewhere in the
3 guesses predict.py shows) was 21.8%, top-5 was 29.1%, both around 5-7x better than random
guessing (1.3% top-1, 4.0% top-3). The confusion matrix showed South Africa and Kenya still
confusing each other in both directions, plus Jordan mistaken for Saudi Arabia and Bangladesh
mistaken for India, both look-alike neighbouring pairs. China mistaken for Qatar was the one
pairing without an obvious explanation, so I've left it unexplained.

**Fine-tuning, attempt 17 (scaling up to 100 countries, same confirmed setup):** early stopping
triggered at epoch 37, best epoch 29, 11.5% valid, essentially flat against the 11.5% at 75
countries rather than continuing the earlier drop (17.1% at 20, 13.0% at 50, 11.5% at 75). Still
11.5x better than random guessing at this size (1.0%). Took 25 hours 51 minutes, the longest run
yet. Whether this is a real plateau for this exact setup, or just how these particular 25
added countries happened to land, isn't clear from one run, so I'm not going to
explain it away.

**Final test-set result (100 countries):** ran the saved model against the untouched test
split of 10,000 photos. 11.4% accuracy, close to the 11.5% validation number, so the plateau
seen at validation holds up on unseen photos too, not just a validation quirk. Top-3
accuracy was 20.8%, top-5 was 27.6%, both around 7x better than random guessing (1.0% top-1,
3.0% top-3). The confusion matrix showed Jordan still mistaken for Saudi Arabia (a repeat from
75 countries), plus new pairs: Ecuador and Costa Rica, South Africa and Tanzania, Croatia and
Greece, all neighbouring or visually similar regions. Nepal mistaken for Peru was the one
pairing without an obvious shared explanation, so I've left it unexplained.

**Project folder moved** from `~/Downloads/photo-geolocation` to
`~/Downloads/github projects/photo-geolocation`, part of organising all the GitHub projects
together. Caused a crash partway through the first ResNet101 attempt (training had saved photo
locations from the old folder, which stopped existing mid-run). Led directly to the autosave
feature below, so a repeat of this never costs hours of training again.

**Fine-tuning, attempt 18 (autosave added):** training now saves a full resume checkpoint after
every single epoch, not just the best one, model weights, optimizer state, and the accuracy
history so far. If a run gets interrupted (a crash, the Mac sleeping, the folder moving), it
picks back up from the last completed epoch instead of starting over from epoch 1. 2 new tests
cover the save and load logic directly, no real training needed to test it.

**Fine-tuning, attempt 19 (ResNet101 instead of ResNet50, 100 countries, same setup):** the
second go at this, after the first one crashed at epoch 15 when the project folder moved. Early
stopping triggered at epoch 36, best epoch 28, 11.8% valid, against 11.5% for ResNet50 at the
same size. That is a gap of 0.3 points from a single run each, small enough to be run-to-run
randomness (the photos get shuffled and augmented differently every time), so this does not
show the deeper model is any better. It did take 48 hours 54 minutes, nearly double the 25 hours
51 minutes ResNet50 needed, so the deeper model cost a lot more time and gave no clear gain.

**Final test-set result (100 countries, ResNet101):** ran the saved ResNet101 model
against the untouched test split of 10,000 photos. 11.3% accuracy, a tie with ResNet50's 11.4%
on the same photos, and 0.5 points under its 11.8% valid score, so it isn't quietly tuned to
the validation photos. Top-3 accuracy was 20.9% and top-5 was 27.7% (ResNet50 got 20.8% and
27.6%), about 7x better than random guessing at top-3 (3.0%). The confusion matrix showed
Jordan guessed as Saudi Arabia 14 times, Ecuador guessed as Costa Rica 11 times, Uganda guessed
as Ghana 10 times, Bahamas guessed as Barbados 9 times and Switzerland guessed as Austria 9
times. Jordan and Saudi Arabia, and Ecuador and Costa Rica, were also ResNet50's top two
mix-ups, so two different networks get tripped up by the same look-alike pairs. The scripts
now work out which ResNet a saved model needs, so they load either one. The confusion matrix
took 46 minutes 1 second and the top-3 numbers 29 minutes 36 seconds, against 18 minutes 41
seconds and 17 minutes 25 seconds for ResNet50.

The picture is the same at 75 and 100 countries, and with ResNet50 and ResNet101: about 11.5%
valid and 11.3 to 11.4% on the test split. A bigger ResNet isn't what's holding it back.

**CLIP as a different starting point (100 countries):** every model so far started from a
ResNet, which learned by sorting about a million photos into 1,000 labelled boxes. CLIP was
trained on hundreds of millions of photos paired with captions, and captions often say where a
photo was taken, so I tried it instead. CLIP stays completely frozen, no fine-tuning at all.
Every photo goes through it once, the 512 numbers it gives back are saved, and a plain logistic
regression on top guesses the country. The regularisation strength C was picked on the valid
split (C=1 gave 24.9%, C=10 gave 26.6%, C=100 gave 23.5%) and the test split was only used once,
at the end. Result on the 10,000 untouched test photos: 26.1% accuracy, against 11.4% for
ResNet50 and 11.3% for ResNet101, and about 26x better than random guessing (1.0%). Top-3 was
42.8% and top-5 was 51.8%, against 20.8% to 20.9% and 27.6% to 27.7% for the two ResNets. The
whole thing took 6 minutes 25 seconds, against 25 hours 51 minutes and 48 hours 54 minutes of
training for the two ResNets.

This is one run of each, but the gap is far bigger than the 0.3 point difference between the
two ResNets. It doesn't tell me why CLIP wins. The captions, the amount of training data, or
CLIP being a different kind of network could each be the reason. I also haven't checked
whether any of these photos were in CLIP's own training data, and Country211 was built by
OpenAI to test CLIP, so that's worth keeping in mind.

**CLIP on all 211 countries:** same setup as the 100-country run and no code changes, I just
unpacked the other 111 country folders from the archive I already had. That's 31,650 train
photos, 10,550 valid and 21,100 test. C=10 won again on the valid split (C=1 gave 22.5%, C=10
gave 26.9%, C=100 gave 24.1%). Result on the 21,100 untouched test photos: 25.9% accuracy,
41.9% top-3 and 50.1% top-5. Random guessing across 211 countries gets 0.5% for top 1 and 1.4%
for top 3, so that's about 55x and 29x better.

Going from 100 countries to 211 barely moved CLIP. Its test accuracy went from 26.1% to 25.9%,
its top-5 from 51.8% to 50.1%, and its valid score from 26.6% to 26.9%. The ResNets' valid
scores fell as countries were added, 17.1% at 20 countries, 13.0% at 50 and 11.5% at 75, then
stayed flat at 11.5% at 100. I only ran CLIP at 100 and 211, so two points aren't a trend. The
ResNets were never run at 211, so it isn't a like for like comparison, but CLIP on the harder
211-country problem still beats both ResNets on the 100-country one (11.4% and 11.3%). The test
score is 1.0 point under the valid score, which is normal because the valid score is the best of
three tries. One run, and it took 14 minutes 32 seconds for all 63,300 photos.

**Zero-shot CLIP, with no training at all (211 countries):** CLIP has two halves that were
trained together, one that turns a photo into 512 numbers and one that turns a sentence into 512
numbers, so a photo and a caption that fits it end up with similar numbers. I wrote one sentence
per country, 'a photo taken in Japan' and so on, turned each one into numbers with the text
half, and for every photo picked the country whose sentence sat closest. The photo numbers are
the ones saved from the 211-country run, so no photos were processed again and nothing was
trained. I chose the wording before running it and didn't change it afterwards. Result on the
21,100 untouched test photos: 16.3% top 1, 30.6% top 3 and 39.0% top 5, against 0.5% and 1.4%
for random guessing, so about 34x and 21x better. The whole run took 7.7 seconds.

Compared with the trained version at 25.9%, my 31,650 training photos added about 9.6 points,
and CLIP already had about 63% of the final score from its own training. It also beats the two
ResNets (11.4% and 11.3%), though those were tested on 100 countries, not 211, so it isn't a
like for like comparison.

A few things to keep in mind. The country names come from pycountry, and a few read oddly as a
sentence, like 'Virgin Islands, British' and 'Holy See (Vatican City State)'. I haven't measured
how much that costs. Country211 was built by OpenAI to test CLIP, and zero-shot is the way they
test it, and I haven't checked whether any of these photos were in CLIP's own training data.

**How the 25.9% is spread across the 211 countries:** an average can hide a lot, so I worked
out CLIP's accuracy for each country on its own. The classifier wasn't saved from the earlier
run, so I trained it again from the saved photo numbers with the same C (10), and the overall
test accuracy came out at 25.9% again, which is the check that nothing was wired wrong. Each
country has 100 test photos, so each score is a fraction out of 100.

The score is spread out, not carried by a few easy countries. The middle country scored 22%, no
country scored exactly 0%, 16 of the 211 scored 50% or more and 18 scored under 10%, which
leaves 177 in between. Even the lowest, at 4%, is about 8x random guessing (0.5%).

Best 10: Panama 79%, Jordan 61%, Greenland 60%, Faroe Islands 59%, Antarctica 57%, North Korea
57%, San Marino 57%, Bhutan 56%, Liechtenstein 56%, South Georgia and the South Sandwich Islands
55%. Worst 10: Austria 4%, Chile 4%, Germany 4%, Colombia 5%, Mozambique 5%, Poland 5%, Romania
5%, Tanzania 6%, Bulgaria 7%, Ghana 7%.

My reading is that the best ones are mostly small or remote places with a distinctive look, and
the worst are mostly bigger countries with look-alike neighbours. That's a guess from the lists,
I haven't looked at any photos. Because each country only has 100 test photos, a score can move
by several points on a different 100, so the pattern is more trustworthy than the exact order
of two close countries (Austria, Chile and Germany at 4% can't be told apart from Colombia at
5%). A few of the 211 are territories, not countries, like Antarctica, Greenland and the Faroe
Islands. It took 9.4 seconds.

**Where CLIP's wrong guesses go (211 countries):** the per-country scores showed which countries
do badly, but not what their photos get guessed as, so I looked at that. Same setup as before:
the classifier trained again from the saved photo numbers (C=10), and the overall test accuracy
came out at 25.9% again. Each country has 100 test photos, so a count of 3 or 4 is weak
evidence and a count of 20 is not.

The 15 most common mix-ups are mostly twins or places in the same area: Jersey guessed as
Guernsey (21 times), Namibia as Botswana (19), South Georgia and the South Sandwich Islands as
the Falkland Islands (17) and the Falklands back as South Georgia (15), Svalbard and Jan Mayen
as Greenland (16), the British Virgin Islands as the US Virgin Islands (16), Burkina Faso as
Mali (13), India as Bangladesh (13), Afghanistan as Tajikistan (12), Bhutan as Nepal (12),
Botswana as Namibia (12), Switzerland as Liechtenstein (12), Italy as the Holy See (12), Kenya
as Botswana (12) and Kyrgyzstan as Tajikistan (12). Kenya as Botswana is the one that doesn't
fit the neighbour idea, those two don't share a border.

For the low scorers, the photos mostly went to small or nearby countries. Austria (4 of 100
right) went to Liechtenstein (8), Andorra (7) and Slovenia (6). Germany (4 right) went to
Liechtenstein (7), Luxembourg (7) and the Netherlands (7). Poland (5 right) went to Czechia (6),
Slovakia (5) and the Åland Islands (4). Romania (5 right) went to Belarus (5), Andorra (4) and
Slovakia (4). Bulgaria (7 right) went to Andorra (6), Albania (5) and Belarus (4). Chile (4
right) went to Mexico (5), Argentina (3) and Poland (3). Colombia (5 right) went to Mexico (4),
Costa Rica (3) and Guatemala (3).

So my guess from the last section held up for most of them, the mistakes are mostly look-alike
neighbours. Some have no obvious reason, like Andorra showing up for Austria, Romania and
Bulgaria, and Mexico showing up for Chile and Colombia, and I haven't looked at the photos to
find out why. It could be that the small countries pick up a lot of loose guesses, but I
haven't tested that. It took 8.1 seconds.

**Guessing a photo with CLIP:** clip_predict.py is the CLIP version of predict.py. I run it
once with `train`, which trains the classifier again from the saved photo numbers (C=10) and
saves it to models/clip_classifier.joblib. That folder is git-ignored, like the other model
files. The test accuracy it prints for the saved classifier is 25.9% on all 211 countries, the
same as clip_probe.py, so the saved one is the right one. After that I give it a photo and it
prints the top 3 guesses.

I tried it on a photo I saved as japan.png: Japan 98.1%, South Korea 1.4%, Taiwan 0.2%. That is
one photo, so it's a demo and not a score. The 25.9% is the real number.

**Does the wording of the zero-shot sentence matter?** I picked "a photo taken in {}" up front
and never tuned it, so I tried five wordings and an average of all five, still with no training
and still on the ViT-B-32 numbers. I picked the best one on the valid photos and only then looked
at the test photos, once. On valid: "a photo taken in {}" 17.1%, "a photo of {}" 15.9%, "a photo
from {}" 17.1%, "a photo of a place in {}" 16.5%, the bare country name 14.9%, and all five
averaged 17.1%. Three tied at the top and the script picked the averaged one.

On the test photos the averaged wordings got 16.5% top 1, 30.8% top 3 and 39.3% top 5. The original
wording got 16.3%, 30.6% and 39.0%. That is 0.2 points better at top 1 and top 3 and 0.3 at top 5. I
hadn't measured how much luck moves a score when I ran this, so I'm not calling this an improvement.
The wording doesn't seem to matter much, and the only one that did clearly worse on valid was the
bare country name. It took 19.0 seconds.

**A bigger CLIP model (ViT-B-16):** every CLIP number up to here used the smallest model, ViT-B-32.
clip_probe_b16.py is the same probe with ViT-B-16 swapped in and nothing else changed: same 31,650
training photos, same C values, C picked on the valid photos only, and the test photos read once at
the end. Its photo numbers go in their own folder, results/clip_features_vit_b16, so the ViT-B-32
ones are left alone.

On valid it got 32.2%. On the test photos it got 31.4% top 1, 48.5% top 3 and 57.0% top 5, against
25.9%, 41.9% and 50.1% for ViT-B-32. That is 5.5 points better at top 1, 6.6 at top 3 and 6.9 at top
5, and the model size is the only thing that changed. It took 1 hour 8 minutes 16 seconds.

**An even bigger model (ViT-L-14):** clip_probe_l14.py is the same probe again with OpenAI's
ViT-L-14 swapped in, so the model is the only thing that changed. Its photo numbers go in
results/clip_features_vit_l14.

On valid it got 41.1%. On the test photos it got 40.4% top 1, 58.6% top 3 and 66.9% top 5. That is
9.0 points better than ViT-B-16 at top 1 (10.1 at top 3 and 9.9 at top 5) and 14.5 points better
than ViT-B-32 (16.7 and 16.8). The gain didn't level off going from ViT-B-16 to ViT-L-14. It took 3
hours 19 minutes 56 seconds.

**The same bigger models with no training (zero-shot):** clip_zero_shot_b16.py and
clip_zero_shot_l14.py are the zero-shot script again with ViT-B-16 and ViT-L-14 swapped in. Same
sentence, "a photo taken in {}", and no training photos used at all.

ViT-B-16 got 20.9% top 1, 36.7% top 3 and 45.6% top 5, and took 8.5 seconds. ViT-L-14 got 29.5%,
48.3% and 57.2%, and took 14.3 seconds. ViT-B-32 got 16.3%, 30.6% and 39.0%, so ViT-L-14 is 13.2
points better at top 1 without seeing a single training photo.

Next to the trained scores, training added 9.6 points at top 1 for ViT-B-32 (16.3% to 25.9%), 10.5
for ViT-B-16 (20.9% to 31.4%) and 10.9 for ViT-L-14 (29.5% to 40.4%). So it adds roughly 10 points
whichever model it sits on. Of the 14.5 points ViT-L-14 gains over ViT-B-32 once trained, 13.2 were
already there with no training.

**Country by country with ViT-L-14:** clip_per_country_l14.py is the same per-country breakdown as
before, done with ViT-L-14's photo numbers. Each country has 100 test photos, so each score is a
fraction out of 100.

The middle country went from 22% with ViT-B-32 to 38%. The number of countries scoring 50% or more
went from 16 to 54, and the number under 10% went from 18 to 0. ViT-L-14 did better on 207 of the
211 countries, the same on 2 and worse on 2. It took 13.0 seconds.

So the gain is spread across almost every country, not carried by a few. A single country can move
by several points on a different 100 photos, but 207 out of 211 is hard to put down to luck.

**Where ViT-L-14's wrong guesses go:** clip_confusions_l14.py takes the 15 most common mix-ups from
the ViT-B-32 check and counts how often ViT-L-14 makes each one. ViT-L-14 made fewer of them in 11
of the 15. In the other 4 it made the same number or more. It took 18.2 seconds.

**Guessing a photo with ViT-L-14:** clip_predict_l14.py is clip_predict.py with ViT-L-14. I run it
once with `train`, which trains the classifier from the saved photo numbers and saves it to
models/clip_classifier_l14.joblib. That file is git-ignored, like the other model files. After that
I give it a photo and it prints the top 3 guesses.

On the same japan.png photo as before it gave Japan 92.5%, South Korea 4.0% and Hong Kong 0.6%,
against Japan 98.1%, South Korea 1.4% and Taiwan 0.2% from ViT-B-32. That is one photo, so it's a
demo and not a score. The 40.4% is the real number.

**Mixing the classifier with the country sentences (ViT-L-14):** clip_blend_l14.py mixes the trained
classifier's guesses with the country-sentence guesses, with a weight for how much the sentences
count. I picked the weight, 0.2, on the valid photos only and read the test photos once.

The blend got 41.2% top 1, 59.9% top 3 and 68.0% top 5, against 40.4%, 58.6% and 66.9% for the
classifier alone. That is 0.8, 1.3 and 1.1 points better. I hadn't measured how much luck moves a
score when I ran this, so I'm treating it as a small gain, not a big one. It took 19.1 seconds.

**Is C=10 the right setting for ViT-L-14?** clip_probe_l14_c_grid.py tries C = 3, 5, 10, 20 and 30
on ViT-L-14's photo numbers. The original probe only tried 1, 10 and 100. On valid, C=3 got 40.5%,
C=5 got 40.9%, C=10 got 41.1%, C=20 got 40.1% and C=30 got 39.7%. So 10 is still the best, and the
score falls away on both sides, but only a little close to 10 (C=5 is 0.2 points lower), so the
setting isn't touchy. On the test photos C=10 gave 40.4% again, the same as before. It took 29.8
seconds.

**Can ViT-L-14's confidence be trusted?** When the classifier says 80%, is it right about 80% of the
time? clip_confidence_l14.py sorts the test photos into 10 bands by how confident ViT-L-14 was in
its top guess, then compares what it said with how often it was right in each band.

It was under-confident, meaning it was right more often than it claimed, not less.

**Fixing ViT-L-14's confidence (temperature scaling):** clip_calibrate_l14.py rescales the
classifier's probabilities with one number, called the temperature, to see if the confidence can be
made to match how often it's right. I picked the temperature, 0.9, on the valid photos only and read
the test photos once.

The average gap between what it said and how often it was right went from 5.3% to 2.3%. Top-1
accuracy didn't change, which is expected because rescaling doesn't change which country is on top.
It took 9.6 seconds.

**Does the number of training photos matter (ViT-L-14)?** clip_training_size_l14.py trains the
classifier on a share of the training photos, the same share from every country, and scores it on
the test photos. With 10% of the training photos it got 26.8% top 1, with 25% it got 33.1%, with 50%
it got 36.9% and with all of them it got 40.4%. Each share is one random pick with a fixed seed, so
the small ones can wobble a little. It took 15.4 seconds.

The curve hasn't flattened: going from half the photos to all of them still added 3.5 points. With
only 10% of the photos, ViT-L-14 already scored slightly higher than ViT-B-32 trained on all of them
(26.8% against 25.9%).

**Is logistic regression the right classifier (ViT-L-14)?** clip_classifiers_l14.py tries two
simpler ways of guessing the country from ViT-L-14's photo numbers. Nearest centroid averages each
country's training photos into one point and picks the country whose point is closest. Nearest
neighbours lets the k closest training photos vote. I picked k on the valid photos only and it ended
up as k=1, so the single closest photo decides. On the test photos, logistic regression got 40.4%
top 1, nearest neighbours 30.2% and nearest centroid 28.8%. It took 32.0 seconds.

So logistic regression is clearly the better choice, by 10 points or more.

**A different ViT-L-14 (DFN-2B weights):** clip_probe_l14_dfn2b.py is the same probe with the DFN-2B
version of ViT-L-14 in place of OpenAI's. It's the same size of model trained on different data, so
only the weights changed. Its photo numbers go in results/clip_features_vit_l14_dfn2b.

On valid it got 37.0%. On the test photos it got 36.7% top 1, 53.7% top 3 and 61.5% top 5. OpenAI's
ViT-L-14 got 40.4%, 58.6% and 66.9%, so DFN-2B was 3.7, 4.9 and 5.4 points worse. It took 3 hours 44
minutes 40 seconds.

DFN-2B was trained on web data, so some of the Country211 test photos could be in its training data.
I haven't checked, the same as for OpenAI's CLIP.

**A third ViT-L-14 (LAION-2B weights):** clip_probe_l14_laion2b.py is the same probe with the
LAION-2B version of ViT-L-14. Again only the weights changed. Its photo numbers go in
results/clip_features_vit_l14_laion2b.

On valid it got 35.5%. On the test photos it got 35.2% top 1, 52.1% top 3 and 60.0% top 5. That is
5.2, 6.5 and 6.9 points worse than OpenAI's ViT-L-14, and 1.5, 1.6 and 1.5 points worse than DFN-2B.
It took 4 hours 32 minutes 13 seconds.

So of the three sets of ViT-L-14 weights, OpenAI's is the best on this task by a clear margin, and
they are not interchangeable. LAION-2B was also trained on web data, so I haven't ruled out overlap
with the Country211 test photos there either.

**Joining two sets of ViT-L-14 photo numbers:** clip_joined_l14.py sticks the OpenAI and DFN-2B
photo numbers for each photo side by side and trains one classifier on the joined list. The idea is
that two different sets of weights might notice different things.

On the test photos it got 42.3% top 1, 60.0% top 3 and 67.8% top 5. OpenAI's alone got 40.4%, 58.6%
and 66.9%, so joining added 1.9, 1.4 and 0.9 points. Against DFN-2B alone (36.7%, 53.7% and 61.5%)
it is 5.6, 6.3 and 6.3 points better. It took 41.2 seconds.

That is more than the 0.8 points the country-sentence blend added at top 1.

**Joining a third set (ViT-B-16 as well):** clip_joined_three.py joins the ViT-B-16 photo numbers
onto the OpenAI and DFN-2B ViT-L-14 ones and trains one classifier on all three.

On the test photos it got 42.0% top 1, 59.6% top 3 and 67.7% top 5. That is 0.3, 0.4 and 0.1 points
lower than the two-set join (42.3%, 60.0% and 67.8%), and 1.6, 1.0 and 0.8 points better than
OpenAI's ViT-L-14 alone. I hadn't measured how much luck moves a score when I ran this, so I'm
treating the third set as neither helping nor clearly hurting. It took 45.6 seconds.

**Joining the LAION-2B numbers too:** clip_joined_l14_all.py tries two more joins of the ViT-L-14
photo numbers. One is OpenAI with LAION-2B, the other is all three ViT-L-14 sets together.

OpenAI with LAION-2B got 41.2% top 1, 59.1% top 3 and 66.7% top 5. That is 0.8 and 0.5 points better
than OpenAI alone at top 1 and top 3, 0.2 worse at top 5, and 1.1, 0.9 and 1.1 points below the
OpenAI and DFN-2B join (42.3%, 60.0% and 67.8%). All three sets got 42.1% top 1, 60.1% top 3 and
67.5% top 5, which is 0.2 below, 0.1 above and 0.3 below that same join. I hadn't measured how much
luck moves a score when I ran this, so I'm treating the LAION-2B numbers as not adding anything to
the OpenAI and DFN-2B join. Both script checks matched the earlier runs (42.3% and 35.2%), and the
whole script took 1 minute 25 seconds.

**Blending the sentences into the join:** clip_blend_joined_l14.py does the same blend as before,
but on top of the classifier trained on the OpenAI and DFN-2B photo numbers joined together. The
sentences are still "a photo taken in {}", scored with the OpenAI ViT-L-14 numbers. The weight on
the sentence scores was picked on the valid photos only.

On the valid photos the weights scored 43.3% at 0.0, 43.9% at 0.1, 44.5% at 0.2 and 44.4% at 0.3, so
0.2 was used. On the test photos that gave 43.5% top 1, 61.9% top 3 and 69.4% top 5. That is 1.2,
1.9 and 1.6 points better than the join alone (42.3%, 60.0% and 67.8%), and 3.1, 3.3 and 2.5 points
better than OpenAI's ViT-L-14 alone. It is the best result so far. It took 23.4 seconds.

**Guessing a photo with the best setup:** clip_predict_best.py runs one photo through both ViT-L-14
models (OpenAI and DFN-2B), joins the two sets of numbers, scores them with the joined classifier
and blends in the sentence scores at weight 0.2. Run it with `train` once to save the classifier,
then give it a photo. It saves under its own name in models/, so the earlier classifiers are left
alone.

On the same japan.png photo as before it gave Japan 95.7%, South Korea 2.1% and China 0.5%, against
Japan 92.5% for ViT-L-14 alone and 98.1% for ViT-B-32. That is one photo, so it says nothing about
which setup is better. The test photos do that, and there it scored 43.5% against 40.4% and 25.9%.
The run took 24.4 seconds. The percentages are the blended scores turned back into probabilities,
and I only checked how honest percentages are for the single OpenAI classifier, so I wouldn't treat
these as exact odds.

**Country by country with the best setup:** clip_per_country_best.py trains ViT-L-14 on its own and
the best setup side by side, and scores each country for both. Each country has 100 test photos, so
each score is a fraction out of 100. Both overall scores matched the earlier runs (40.4% and 43.5%).

The middle country went from 38% with ViT-L-14 alone to 41%. The number of countries scoring 50% or
more went from 54 to 71. The best setup did better on 151 of the 211 countries, the same on 14 and
worse on 46. The best countries were the Faroe Islands at 81% and Panama at 79%. The lowest were
Austria and Colombia at 16%, then Jamaica and Mozambique at 17%, and Ukraine, Venezuela and Kosovo
at 18%. It took 25.5 seconds.

Of the 10 countries ViT-L-14 alone scored lowest on, 9 went up and Austria stayed at 16%, but 7 of
those 10 are still in the best setup's bottom 10. So it helped them without fixing them. I picked
those 10 from one model's results, so some would creep up on a re-run by chance, and the 46
countries that got worse show the gain is not even. With 100 photos a country, small differences
wobble.

**Where the best setup's wrong guesses go:** clip_confusions_best.py takes the 15 most common
mistakes of ViT-L-14 on its own and counts how often the best setup makes each one. Both overall
scores matched the earlier runs (40.4% and 43.5%).

The best setup made fewer of these mistakes in 10 of the 15 pairs, more in 4 and the same in 1.
Israel as Palestine went from 14 to 8, Bhutan as Nepal from 13 to 8, Cyprus as Greece from 13 to 9
and South Georgia as the Falklands from 17 to 11. Kosovo as Bosnia and Herzegovina went up from 16
to 20 and Latvia as Estonia from 13 to 17. It took 24.4 seconds.

The best setup's own most common mistakes include Kosovo as Bosnia and Herzegovina (20), Latvia as
Estonia (17), Finland as Sweden (16), Peru as Bolivia (16) and Tanzania as Kenya (16), all nearby
countries. For the low scoring countries, Austria, Bulgaria, Romania, Chile and Colombia mostly went
to neighbours or near neighbours, but not all of it is tidy. Some of the odd ones are Poland photos
going to Latvia (9) and Colombia photos going to Paraguay (7).

I picked those 15 pairs because they were ViT-L-14's worst, so some would drop by chance alone, and
counts of 4 to 11 out of 100 are weak evidence. I haven't looked at the photos, so I'm leaving the
reasons out.

**How honest the best setup's percentages are:** clip_calibrate_best.py does the same check as
clip_calibrate_l14.py, but on the blended scores. It tries one number that makes every percentage a
bit sharper or flatter, and picks it on the valid photos only by lowest log loss. The test photos
were read once at the end.

The best number was 0.95, which is barely a change. The valid log loss was 2.3338 at 0.90, 2.3310 at
0.95 and 2.3346 at 1.00, so it is flat there and I'd say about 0.95, not exactly 0.95. The average
gap between what it said and how often it was right went from 3.0% to 2.0%. ViT-L-14 on its own
started at 5.3%, so blending in the sentences already made the percentages more honest before any
rescaling. Top 1 stayed at 43.5%, as it has to, since the order of the countries for each photo does
not change. It took 27.0 seconds.

After rescaling, the low bins are a little too sure of themselves (10% to 20% said 15.0% and was
right 13.5%). The 70% to 90% bins are still modest by about 4 points (80% to 90% said 85.0% and was
right 88.8%). The top bin, 90% to 100%, said 95.8% and was right 97.2%, so very confident guesses
can be trusted. So the percentages are mostly honest and slightly modest in the middle, and I
wouldn't call them exact odds.

**Checking how much luck moves the score:** earlier I wrote that luck alone moves a score by about
0.3 points, but that was a guess and I hadn't measured it. clip_bootstrap_best.py measures it. It
picks 21,100 photos at random from the test photos, with repeats allowed, scores a setup on that
pick, and does this 1000 times. The middle 95% of those scores is the range. Both setups are scored
on the same picks each time, so the gap between them is fair.

Both full-test scores came back as before (40.4% for OpenAI's ViT-L-14 alone and 43.5% for the best
setup). Over the 1000 resamples, OpenAI's ViT-L-14 alone ranged from 39.7% to 41.1% top 1, and the
best setup ranged from 42.8% to 44.2%. The gap between them ranged from 2.62 to 3.51 points, and the
best setup scored higher than OpenAI alone in 100.0% of the resamples. So my 0.3 guess was too
small. Luck moves one score by about 0.7 points either way, and the best setup is still clearly
ahead of OpenAI's ViT-L-14 alone. I haven't run this on the smaller steps, so I can't say whether
joining on its own (1.9 points) or adding the sentences (1.2 points) would hold up. The run took 1
minute 36.40 seconds, which includes downloading the 1.71GB OpenAI weights again.

**Next real step:** check that no photo shows up in more than one of the train, valid and test
splits.

## Try it on your own photo

For the best results use the CLIP version. It runs each photo through both ViT-L-14 models, scores
it with the joined classifier and mixes in the sentence scores. Run `train` once to save the
classifier, then give it a photo and it prints the top 3 guesses:

```bash
python3 src/clip_predict_best.py train
python3 src/clip_predict_best.py path/to/your/photo.jpg
```

The `train` step needs the photo numbers that `clip_probe_l14.py` and `clip_probe_l14_dfn2b.py`
save, so run those first. The saved classifier goes in `models/`, which git ignores. On my japan.png
demo photo it gave Japan 95.7%, South Korea 2.1% and China 0.5%. That is one photo, so it's a demo
and not a score. The percentages are only mostly honest, a bit modest in the middle, so I wouldn't
treat them as exact odds.

`clip_predict.py` (ViT-B-32) and `clip_predict_l14.py` (ViT-L-14 alone) are the smaller versions,
and they're run the same way.

**The ResNet version:**

`src/predict.py` loads the best saved model and predicts the top 3 most likely countries for
any photo you give it, with a confidence percentage for each (not just a raw guess).
Prints the full country name next to its code (Japan (JP)), not just the code on its own.

**How good are those top 3 guesses, really?** Measured properly against the untouched test
split (50 countries): the right country was the single best guess 12.1% of the time, somewhere
in the top 3 guesses 23.4% of the time, and in the top 5 guesses 31.2% of the time. Random
guessing across 50 countries would get 2.0% for the top guess and 6.0% for the top 3, so the
model's three guesses do capture the right answer far more often than one guess
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

`build_model` in `src/finetune_model.py` currently builds a ResNet101. For the ResNet50 setup from
earlier, swap `resnet101` and `ResNet101_Weights` in that function for `resnet50` and
`ResNet50_Weights`.

Last script runs the saved model against the untouched test split for a final accuracy
number, plus prints the most commonly confused country pairs.

**The CLIP scripts:** these read photo numbers that an earlier script saved, so they go in order.
Each probe script extracts the photo numbers the first time and saves them to `results/` (git
ignored), and reuses them after that. The first run of each CLIP model downloads its weights through
open_clip, so that run needs internet.

```bash
python3 src/clip_probe.py
python3 src/clip_probe_l14.py
python3 src/clip_probe_l14_dfn2b.py
python3 src/clip_joined_l14.py
python3 src/clip_blend_joined_l14.py
python3 src/clip_predict_best.py train
```

That's the path to the best result. `clip_probe_b16.py`, `clip_probe_l14_laion2b.py`, the
`clip_zero_shot*.py` scripts and the other `clip_` scripts are the side experiments from "Results so
far", and each one says at the top what it needs and how to run it. The ViT-L-14 probes were the
slow ones, between 3 hours 19 minutes and 4 hours 32 minutes each on my Mac, and ViT-B-16 took 1
hour 8 minutes. After the probes, `clip_joined_l14.py` took 41.2 seconds and
`clip_blend_joined_l14.py` took 23.4 seconds.

## Testing and git

`tests/` covers the bits that don't need the full dataset, country code lookups, folder scanning,
both models' feature extraction, layer-freezing, augmentation, learning rate, best-epoch, early
stopping, and checkpoint saving logic, the prediction script's ranking and model-loading logic, and
the confusion matrix's counting and mix-up-finding logic. The CLIP scripts have tests for their
small functions too, like the top-k scoring, the sentence blend, the temperature scaling, the scores
for each country, the nearest centroid, the training-size sampling and the joining and column
picking. All of them use made-up numbers and fake models, so none of them needs the dataset or a
downloaded model.

`data/`, `models/` and `results/` are git-ignored, so the dataset, the saved classifiers and the
saved photo numbers never go in the repo.

## Tools used

Python, PyTorch/torchvision, scikit-learn, open_clip, pycountry, pytest.

## Where the data's from

[Country211](https://github.com/openai/CLIP/blob/main/data/country211.md), built by OpenAI from
YFCC100M geotagged Flickr photos. Used here under the same terms, the photos themselves are
whatever Creative Commons licence their original uploaders picked.
