"""
Tests for check_split_overlap.py, using tiny made-up files and fingerprint lists instead of the
real Country211 photos.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import check_split_overlap


def test_file_hash_is_the_same_for_files_with_the_same_bytes_and_different_otherwise(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"same bytes")
    (tmp_path / "b.jpg").write_bytes(b"same bytes")
    (tmp_path / "c.jpg").write_bytes(b"other bytes")

    assert check_split_overlap.file_hash(tmp_path / "a.jpg") == check_split_overlap.file_hash(tmp_path / "b.jpg")
    assert check_split_overlap.file_hash(tmp_path / "a.jpg") != check_split_overlap.file_hash(tmp_path / "c.jpg")


def test_shared_between_splits_finds_a_fingerprint_that_is_in_two_splits():
    hashes = {"train": ["a", "b"], "valid": ["c"], "test": ["d", "a"]}

    assert check_split_overlap.shared_between_splits(hashes) == {"a"}


def test_shared_between_splits_is_empty_when_the_splits_have_nothing_in_common():
    hashes = {"train": ["a", "b"], "valid": ["c"], "test": ["d"]}

    assert check_split_overlap.shared_between_splits(hashes) == set()


def test_shared_between_splits_does_not_count_a_fingerprint_repeated_inside_one_split():
    hashes = {"train": ["a", "a"], "valid": ["b"], "test": ["c"]}

    assert check_split_overlap.shared_between_splits(hashes) == set()


def test_repeats_within_split_counts_the_extra_copies():
    assert check_split_overlap.repeats_within_split(["a", "b", "a", "a", "c"]) == 2
    assert check_split_overlap.repeats_within_split(["a", "b", "c"]) == 0
