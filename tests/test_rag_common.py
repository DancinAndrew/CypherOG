"""rag_common: path-safe error messages and the data/processed corpus pin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import rag_common
from rag_common import (
    corpus_digest,
    corpus_mismatches,
    describe_exception,
    display_path,
    load_corpus_pin,
    processed_corpus_hashes,
)

# --- error messages ---------------------------------------------------------


def test_display_path_is_relative_inside_the_repo_and_short_outside(tmp_path):
    assert display_path(rag_common.ROOT / "experiments" / "gold.json") == "experiments/gold.json"
    assert display_path(tmp_path / "deep" / "file.txt") == "deep/file.txt"


def test_os_errors_show_only_a_short_path(tmp_path):
    exc = FileNotFoundError(2, "No such file or directory", str(tmp_path / "r1" / "q3.md"))
    text = describe_exception(exc)
    assert text == "FileNotFoundError: No such file or directory r1/q3.md"
    assert str(tmp_path) not in text


def test_other_messages_lose_repo_and_home_prefixes():
    exc = RuntimeError(f"bad index at {rag_common.ROOT / 'chroma_db'} and {Path.home() / 'x'}")
    text = describe_exception(exc)
    assert text.startswith("RuntimeError: bad index at chroma_db and ~")
    assert str(rag_common.ROOT) not in text
    assert describe_exception(ValueError()) == "ValueError"


# --- corpus pin -------------------------------------------------------------


def test_pin_file_matches_the_committed_processed_text():
    pinned = load_corpus_pin()
    assert corpus_mismatches(processed_corpus_hashes(), pinned) == []
    data = json.loads(rag_common.CORPUS_PIN_PATH.read_text(encoding="utf-8"))
    assert data["digest"] == corpus_digest(pinned)
    assert data["directory"] == "data/processed"


def test_digest_depends_on_names_and_contents_not_order():
    a = {"x.txt": "1" * 64, "y.txt": "2" * 64}
    assert corpus_digest(a) == corpus_digest(dict(reversed(list(a.items()))))
    assert corpus_digest(a) != corpus_digest({**a, "y.txt": "3" * 64})
    assert corpus_digest(a) != corpus_digest({"x.txt": "1" * 64, "z.txt": "2" * 64})


def test_mismatches_list_changed_missing_and_extra_files():
    pinned = {"a.txt": "1", "b.txt": "2", "c.txt": "3"}
    actual = {"a.txt": "1", "b.txt": "changed", "d.txt": "4"}
    assert corpus_mismatches(actual, pinned) == ["b.txt", "c.txt", "d.txt"]


def test_text_hash_equals_file_hash_for_processed_text(tmp_path):
    (tmp_path / "doc.txt").write_text("震感舞 Popping\n第二行", encoding="utf-8")
    assert processed_corpus_hashes(tmp_path) == {"doc.txt": rag_common.text_sha256("震感舞 Popping\n第二行")}


@pytest.mark.parametrize(
    "payload",
    [
        {"files": {}},
        {"files": {"a.txt": 1}, "digest": "x"},
        {"files": {"a.txt": "1"}, "digest": "not-the-digest"},
        ["not", "an", "object"],
    ],
)
def test_malformed_pin_files_are_rejected(tmp_path, payload):
    path = tmp_path / "corpus_pin.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_corpus_pin(path)
