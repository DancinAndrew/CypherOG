"""experiments/corpus_round1.json：round-1 索引實際收錄的 39 份文件，以及語料定義檔的共用工具。"""

from __future__ import annotations

import json

import pytest

import rag_common
from data_update import SUPPORTED_SUFFIXES
from rag_common import (
    CORPUS_ROUND1_PATH,
    DATA_RAW,
    corpus_digest,
    document_list_digest,
    load_corpus_definition,
    load_corpus_pin,
    pinned_digest_for,
    processed_name_for,
)

# 三份結構化筆記在 round 1 之後才加入索引（隔離重現：排除後 25/25 完全一致，全部 42 份只有 11/25）
STRUCTURED_NOTES = ("popping_glossary.md", "popping_history_timeline.md", "popping_style_taxonomy.md")


def supported_raw_files() -> set[str]:
    return {p.name for p in DATA_RAW.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES}


# --- the committed definition -----------------------------------------------


def test_round1_corpus_is_every_raw_file_minus_the_three_structured_notes():
    definition = load_corpus_definition(CORPUS_ROUND1_PATH)
    raw = supported_raw_files()

    assert definition["name"] == "round1"
    assert set(STRUCTURED_NOTES) <= raw
    assert set(definition["documents"]) == raw - set(STRUCTURED_NOTES)
    assert len(definition["documents"]) == 39
    assert definition["excluded"] == sorted(STRUCTURED_NOTES)


def test_round1_documents_are_listed_sorted_and_with_evidence():
    data = json.loads(CORPUS_ROUND1_PATH.read_text(encoding="utf-8"))
    assert data["documents"] == sorted(data["documents"])
    assert set(data) == {"name", "description", "evidence", "documents", "excluded"}
    assert "25" in data["evidence"] and "11/25" in data["evidence"]


def test_every_round1_document_is_covered_by_the_corpus_pin():
    pinned = load_corpus_pin()
    documents = load_corpus_definition(CORPUS_ROUND1_PATH)["documents"]
    assert [d for d in documents if processed_name_for(d) not in pinned] == []
    # 鎖定檔另外只多出三份結構化筆記
    extra = set(pinned) - {processed_name_for(d) for d in documents}
    assert extra == {processed_name_for(n) for n in STRUCTURED_NOTES}


# --- helpers ----------------------------------------------------------------


def test_processed_name_follows_data_update():
    assert processed_name_for("Popping.md") == "Popping.txt"
    assert processed_name_for("Guarato.Dance.Articulated.pdf") == "Guarato.Dance.Articulated.txt"
    assert processed_name_for("popping-terminology-snippet.txt") == "popping-terminology-snippet.txt"


def test_document_list_digest_ignores_order_and_duplicates_but_not_names():
    assert document_list_digest(["b.md", "a.md"]) == document_list_digest(["a.md", "b.md", "a.md"])
    assert document_list_digest(["a.md"]) != document_list_digest(["a.md", "b.md"])
    assert document_list_digest(["a.md"]) != document_list_digest(["a.txt"])


def test_pinned_digest_covers_only_the_selected_documents():
    pinned = {"a.txt": "1" * 64, "b.txt": "2" * 64, "c.txt": "3" * 64}
    assert pinned_digest_for(["a.md", "b.pdf"], pinned) == corpus_digest({"a.txt": "1" * 64, "b.txt": "2" * 64})
    assert pinned_digest_for(["a.md", "z.md"], pinned) is None  # z 未被鎖定
    assert pinned_digest_for([], pinned) is None


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {"name": "Round 1", "documents": ["a.md"]},  # 名稱會進 collection 名稱，只允許小寫英數與底線
        {"name": "round1", "documents": []},
        {"name": "round1", "documents": ["a.md", "a.md"]},
        {"name": "round1", "documents": ["../secret.md"]},
        {"name": "round1", "documents": ["a.md"], "excluded": ["a.md"]},
        {"name": "round1", "documents": ["a.md", 3]},
    ],
)
def test_malformed_corpus_definitions_are_rejected(tmp_path, payload):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_corpus_definition(path)


def test_loaded_definition_is_a_new_sorted_copy(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"name": "mini", "documents": ["b.md", "a.md"]}), encoding="utf-8")
    first = load_corpus_definition(path)
    first["documents"].append("c.md")
    assert load_corpus_definition(path) == {"name": "mini", "documents": ["a.md", "b.md"], "excluded": []}
    assert rag_common.CORPUS_ROUND1_PATH == rag_common.ROOT / "experiments" / "corpus_round1.json"
