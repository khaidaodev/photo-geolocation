# Photo Geolocation

Guesses which country a photo was taken in, just from what's in the picture. No GPS metadata,
no filename hints, just pixels.

## Why this problem

Humans who are good at this (there's a whole hobby/game built around it called GeoGuessr) pick
up on things like road markings, plant life, architecture style, which side of the road cars
drive on, even the shape of power line poles, to work out roughly where in the world a photo was
taken. It's a genuinely hard visual reasoning problem, and a good test of whether a model can
pick up on the same kind of subtle, spread-out clues rather than just recognising one obvious
object sitting in the middle of the frame.

## Where it's at right now

Data pipeline's built, and there's a first working baseline model with real numbers, see
"Results so far" below. Next step is trying to actually improve on that baseline, then scaling
up from 20 countries towards the full 211.

Using Country211, a dataset OpenAI built (originally to test their CLIP model) by filtering
millions of geotagged Flickr photos down to a clean, balanced set: 150 training photos, 50
validation photos, and 100 test photos, for each of 211 countries. Total dataset is about 63,000
photos, roughly 11GB.

211-way classification (is this Peru or Chile or literally any of the other 209 countries) is an
extremely hard first problem, even OpenAI's own CLIP model finds this a genuinely tough
benchmark. So rather than jump straight into all 211 countries, the first pass uses a smaller,
easier-to-tell-apart starting set of well known, visually distinctive countries (US, UK, France,
Japan, Brazil, and so on, see `STARTER_COUNTRIES` in `src/data_loading.py`), then scales up to
more countries once that's working properly.

## Results so far

First baseline: a frozen, pretrained ResNet18 turns each photo into a 512-number summary, and a
logistic regression classifier is trained on top of those numbers to guess the country. This is
called transfer learning, using a model that already knows how to "see" from millions of general
photos, rather than trying to train a whole network from scratch on a few thousand photos, which
just isn't enough data to learn from.

On the 20 starter countries: **10.6% validation accuracy**, against 5% for random guessing.
Not a great number by itself, but it's genuinely learning something rather than guessing blind,
and this is expected for a first pass with no fine-tuning. The realistic next steps are actually
fine-tuning the network instead of keeping it fully frozen, or trying a stronger pretrained model
as the feature extractor.

## How to run this yourself

```bash
pip3 install -r requirements.txt
python3 src/data_loading.py
python3 src/baseline_model.py
```

The first script downloads the full Country211 dataset the first time you run it. It's about
11GB, so it'll take a while depending on your internet connection, and only needs to happen once.
Worth extracting just the 20 starter countries first rather than all 211, since the archive stays
on disk afterwards and more countries can be extracted from it later without downloading again.

The second script trains and evaluates the baseline model on whichever countries have actually
been extracted.

## Testing and git

There's a `tests/` folder with pytest tests covering the parts of the pipeline that don't need
the actual 11GB dataset downloaded, country code lookups and folder-scanning logic mostly. Will
keep adding tests as the actual model-building continues.

## Tools used

Python, PyTorch/torchvision (has this exact dataset built in already), scikit-learn for the
baseline classifier, pycountry for turning country codes into readable names.

## Where the data's from

[Country211](https://github.com/openai/CLIP/blob/main/data/country211.md), built by OpenAI from
the YFCC100M dataset of geotagged Flickr photos, for testing their CLIP model's geolocation
ability. Used here under the same terms, the underlying photos are subject to whatever Creative
Commons licence their original uploaders chose.
