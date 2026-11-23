"""rag_common: path-safe error messages and the data/processed corpus pin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import rag_common
from rag_common import (
    absent_withheld,
    corpus_digest,
    corpus_mismatches,
    describe_exception,
    display_path,
    load_corpus_pin,
    load_withheld,
    pin_status,
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
    withheld = load_withheld(pinned=pinned)
    actual = processed_corpus_hashes()
    # withheld 文件不在時不算不一致；在時（讀者自備副本）必須完全相同
    assert corpus_mismatches(actual, pinned, withheld) == []
    if not absent_withheld(actual, withheld):
        assert corpus_mismatches(actual, pinned) == []
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


# --- withheld documents -----------------------------------------------------


def test_the_withheld_chapter_is_listed_with_its_pinned_hash():
    pinned = load_corpus_pin()
    withheld = load_withheld(pinned=pinned)
    name = "a-historical-overview-of-hip-hop-dances-johnson-imani-kai.txt"
    assert list(withheld) == [name]
    assert withheld[name]["raw"] == "a-historical-overview-of-hip-hop-dances-johnson-imani-kai.pdf"
    assert withheld[name]["sha256"] == pinned[name]
    assert "data/SOURCES.md" in withheld[name]["reason"]
    assert rag_common.WITHHELD_PATH == rag_common.ROOT / "experiments" / "withheld.json"


def test_an_absent_withheld_file_is_not_a_mismatch_but_an_altered_copy_is():
    pinned = {"a.txt": "1", "w.txt": "2"}
    assert corpus_mismatches({"a.txt": "1"}, pinned, ["w.txt"]) == []
    assert absent_withheld({"a.txt": "1"}, ["w.txt"]) == ["w.txt"]
    assert corpus_mismatches({"a.txt": "1", "w.txt": "2"}, pinned, ["w.txt"]) == []
    assert corpus_mismatches({"a.txt": "1", "w.txt": "own copy"}, pinned, ["w.txt"]) == ["w.txt"]
    # 其他檔案的嚴格度不變：非 withheld 的缺檔仍是不一致
    assert corpus_mismatches({"w.txt": "2"}, pinned, ["w.txt"]) == ["a.txt"]
    assert corpus_mismatches({"a.txt": "1"}, pinned) == ["w.txt"]


def test_pin_status_never_reports_a_corpus_with_withheld_gaps_as_the_pinned_one():
    assert pin_status([]) is True
    assert pin_status(["w.txt"]) == "withheld:1"
    assert pin_status(["w.txt", "v.txt", "w.txt"]) == "withheld:2"


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"raw": "w.pdf", "processed": "w.txt", "sha256": "1" * 64}, "reason"),
        ({"raw": "w.pdf", "processed": "other.txt", "sha256": "1" * 64, "reason": "x"}, "processed"),
        ({"raw": "../w.pdf", "processed": "w.txt", "sha256": "1" * 64, "reason": "x"}, "processed"),
        ({"raw": "w.pdf", "processed": "w.txt", "sha256": "0" * 64, "reason": "x"}, "corpus_pin"),
    ],
)
def test_malformed_withheld_files_are_rejected(tmp_path, entry, match):
    path = tmp_path / "withheld.json"
    path.write_text(json.dumps({"documents": [entry]}), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_withheld(path, pinned={"w.txt": "1" * 64})
