"""Checks that no photo turns up in more than one of the train, valid and test splits, since a
test photo that is also in the training photos would make the test scores look better than they
are. It reads every photo file, makes a short fingerprint (a SHA-256 hash) of its bytes, and looks
for fingerprints that appear in more than one split, or twice in the same split. This only catches
byte for byte identical files. A resized or re-saved copy of the same picture would get a
different fingerprint and slip through.

Run it with:
    python3 src/check_split_overlap.py
"""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COUNTRY211_DIR = ROOT / "data" / "raw" / "country211"
SPLITS = ("train", "valid", "test")


def file_hash(path, chunk_size=1024 * 1024):
    """A short fingerprint of a file's bytes. Two files with the same bytes get the same one."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shared_between_splits(hashes_by_split):
    """The fingerprints that show up in more than one split. A fingerprint repeated inside a
    single split does not count here."""
    seen_in = {}
    for split, hashes in hashes_by_split.items():
        for h in set(hashes):
            seen_in.setdefault(h, set()).add(split)
    return {h for h, splits in seen_in.items() if len(splits) > 1}


def repeats_within_split(hashes):
    """How many photos in one split have the same fingerprint as a photo already seen there."""
    return len(hashes) - len(set(hashes))


if __name__ == "__main__":
    hashes_by_split = {}
    for split in SPLITS:
        files = sorted(p for p in (COUNTRY211_DIR / split).rglob("*") if p.is_file() and not p.name.startswith("."))
        print(f"Reading the {len(files)} {split} photos...")
        hashes_by_split[split] = [file_hash(p) for p in files]

    shared = shared_between_splits(hashes_by_split)
    print("\nPhotos per split:", {split: len(h) for split, h in hashes_by_split.items()})
    print(f"Fingerprints that appear in more than one split: {len(shared)}")
    for split, hashes in hashes_by_split.items():
        print(f"Photos in {split} that repeat another {split} photo: {repeats_within_split(hashes)}")
