# How it works

This is the plain version of what the code does, written so I can explain it to myself later.
Every number here came from a run on my own machine.

## The job

Given a photo, guess which of 211 countries it was taken in. The data is Country211: 31,650
photos to learn from (train), 10,550 to choose settings on (valid) and 21,100 to score at the
end (test). The test photos are for reporting scores. I picked C and the sentence weight on
valid, but I also compared the different setups by their test scores, so the best score has had
a small push from that choice, and I haven't measured how much.

## Step 1: turn each photo into numbers

CLIP is a model that was trained on huge numbers of photos paired with captions. I never change
it. I run each photo through it once and save the list of numbers it gives back. That list is
called a feature vector, and it works like a fingerprint of what's in the photo. I use two
different CLIP models, OpenAI ViT-L-14 and DFN-2B ViT-L-14, and stick their numbers together
for each photo, 768 from each.

## Step 2: learn which fingerprints go with which country

On top of the saved numbers I train a logistic regression. That's a simple classifier, a bit
like drawing lines that separate the countries. The setting C controls how tightly it fits the
training photos, and I picked it on the valid split for the earlier CLIP runs, then kept C =
10.0 for the joined setup.

## Step 3: mix in the country names

CLIP can also turn a sentence like "a photo taken in France" into numbers, and photos close to
that sentence are probably from France. That needs no training, so it is called zero-shot. On
its own it gets 29.5% right on the test photos. The classifier on its own gets 42.3%. I mix
the two with a weight of 0.2 for the sentences, chosen on the valid split, and the mix gets
43.5%.

## What the final score means

The best setup picks the right country first time for 43.5% of test photos. The right country
is in its top 3 guesses 61.9% of the time and in its top 5 guesses 69.4% of the time. With
OpenAI ViT-L-14 alone it was 40.4%, 58.6% and 66.9%.

## Checks I ran so I could trust the score

- No photo appears in more than one split. I compared a fingerprint of every file (a SHA-256
  hash) and found no overlap between train, valid and test.
- Right continent: the best setup puts the photo on the right continent 77.0% of the time, and
  59.4% of its wrong-country answers still land on the right continent.
- The sentence mix helped on 6 of the 7 continents. On Antarctica, which has only 200 test
  photos, it was 1.0 points lower.

## What it still gets wrong

South America (55.7% right continent) and Oceania (63.3%) are the weakest. Photos from both
are guessed as North America more than any other wrong continent (15.7% and 16.9%). I haven't
tested why, so I'm not claiming a reason.
