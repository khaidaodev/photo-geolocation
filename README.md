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

Just getting started, the data pipeline is set up, nothing trained yet.

Using Country211, a dataset OpenAI built (originally to test their CLIP model) by filtering
millions of geotagged Flickr photos down to a clean, balanced set: 150 training photos, 50
validation photos, and 100 test photos, for each of 211 countries. Total dataset is about 63,000
photos, roughly 11GB.

211-way classification (is this Peru or Chile or literally any of the other 209 countries) is an
extremely hard first problem, even OpenAI's own CLIP model finds this a genuinely tough
benchmark. So rather than jump straight into all 211 countries, the first pass is going to be a
smaller, easier-to-tell-apart starting set of well known, visually distinctive countries (US, UK,
France, Japan, Brazil, and so on, see `STARTER_COUNTRIES` in `src/data_loading.py`), then scale
up to more countries once that's actually working.

## How to run this yourself

```bash
pip3 install -r requirements.txt
python3 src/data_loading.py
```

Downloads the full Country211 dataset the first time you run it. It's about 11GB, so it'll take
a while depending on your internet connection, only needs to happen once.

## Testing and git

There's a `tests/` folder with pytest tests covering the parts of the pipeline that don't need
the actual 11GB dataset downloaded, country code lookups and folder-scanning logic mostly. Will
keep adding tests as the actual model-building starts.

## Tools used

Python, PyTorch/torchvision (has this exact dataset built in already), pycountry for turning
country codes into readable names.

## Where the data's from

[Country211](https://github.com/openai/CLIP/blob/main/data/country211.md), built by OpenAI from
the YFCC100M dataset of geotagged Flickr photos, for testing their CLIP model's geolocation
ability. Used here under the same terms, the underlying photos are subject to whatever Creative
Commons licence their original uploaders chose.
