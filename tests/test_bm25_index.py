"""Per-collection BM25 index: shared tokenizer, JSON persistence and index building."""

from __future__ import annotations

import json

import pytest

import data_update
import rag_common
from rag_common import (
    bm25_index_path,
    load_bm25_index,
    refresh_bm25_index,
    save_bm25_index,
    tokenize_for_bm25,
)


@pytest.fixture
def persist_dir(tmp_path, monkeypatch):
    target = tmp_path / "chroma"
    monkeypatch.setattr(rag_common, "CHROMA_PERSIST_DIR", str(target))
    monkeypatch.setattr(rag_common, "CHROMA_COLLECTION", "cypher_og")
    return target


class FakeWritableCollection:
    """The subset of chromadb.Collection used by data_update and refresh_bm25_index."""

    def __init__(self, rows: dict[str, tuple[str, dict]] | None = None):
        self.rows = dict(rows or {})

    def get(self, ids=None, where=None, include=None):
        keys = [
            k
            for k, (_, meta) in self.rows.items()
            if (ids is None or k in ids)
            and (where is None or all(meta.get(f) == v for f, v in where.items()))
        ]
        out = {"ids": keys}
        if include and "documents" in include:
            out["documents"] = [self.rows[k][0] for k in keys]
        if include and "metadatas" in include:
            out["metadatas"] = [self.rows[k][1] for k in keys]
        return out

    def delete(self, ids):
        self.rows = {k: v for k, v in self.rows.items() if k not in set(ids)}

    def add(self, ids, documents, metadatas, embeddings):
        assert len(ids) == len(documents) == len(metadatas) == len(embeddings)
        self.rows = {**self.rows, **{i: (d, m) for i, d, m in zip(ids, documents, metadatas)}}

    def count(self):
        return len(self.rows)


# --- tokenizer --------------------------------------------------------------


def test_tokenizer_keeps_english_words_digits_and_single_cjk_characters():
    assert tokenize_for_bm25("Popin' Pete 的成員 2002!") == [
        "popin'",
        "pete",
        "的",
        "成",
        "員",
        "2002",
    ]


def test_tokenizer_never_returns_an_empty_list():
    assert tokenize_for_bm25("!!! ... ？") == [""]
    assert tokenize_for_bm25("") == [""]


# --- persistence ------------------------------------------------------------


def test_each_collection_gets_its_own_json_file(persist_dir):
    alpha = save_bm25_index(["Boogaloo Sam"], ["a.txt::0"], "alpha_coll")
    beta = save_bm25_index(["Juste Debout", "Popping"], ["b.txt::0", "b.txt::1"], "beta_coll")

    assert alpha == persist_dir / "bm25_alpha_coll.json"
    assert beta == persist_dir / "bm25_beta_coll.json"
    payload = json.loads(beta.read_text(encoding="utf-8"))
    assert payload == {
        "collection": "beta_coll",
        "ids": ["b.txt::0", "b.txt::1"],
        "docs": ["Juste Debout", "Popping"],
        "tokenized": [["juste", "debout"], ["popping"]],
    }
    assert load_bm25_index("alpha_coll")[1] == ["a.txt::0"]
    assert load_bm25_index("beta_coll")[1] == ["b.txt::0", "b.txt::1"]


def test_default_collection_and_persist_dir_are_read_at_call_time(persist_dir, monkeypatch):
    assert bm25_index_path() == persist_dir / "bm25_cypher_og.json"
    monkeypatch.setattr(rag_common, "CHROMA_COLLECTION", "other_coll")
    assert bm25_index_path() == persist_dir / "bm25_other_coll.json"


def test_index_file_is_utf8_json_with_readable_cjk(persist_dir):
    path = save_bm25_index(["震感舞"], ["x.txt::0"], "cjk_coll")
    assert "震感舞" in path.read_text(encoding="utf-8")


def test_unsafe_collection_names_are_rejected(persist_dir):
    with pytest.raises(ValueError):
        bm25_index_path("../escape")
    with pytest.raises(ValueError):
        bm25_index_path("a/b")


def test_mismatched_docs_and_ids_are_rejected(persist_dir):
    with pytest.raises(ValueError):
        save_bm25_index(["one", "two"], ["only-one"], "bad_coll")


def test_missing_index_is_silent(persist_dir, capsys):
    assert load_bm25_index("never_built") == (None, [], [])
    assert capsys.readouterr().err == ""


def test_empty_corpus_loads_as_no_index(persist_dir, capsys):
    save_bm25_index([], [], "empty_coll")
    assert load_bm25_index("empty_coll") == (None, [], [])
    assert capsys.readouterr().err == ""


def test_corrupted_index_warns_on_stderr_and_returns_none(persist_dir, capsys):
    persist_dir.mkdir(parents=True)
    (persist_dir / "bm25_broken_coll.json").write_text("{not json", encoding="utf-8")
    assert load_bm25_index("broken_coll") == (None, [], [])
    err = capsys.readouterr().err
    assert "broken_coll" in err
    assert "JSONDecodeError" in err


def test_structurally_invalid_index_warns_on_stderr(persist_dir, capsys):
    persist_dir.mkdir(parents=True)
    bad = {"collection": "odd_coll", "ids": ["a", "b"], "docs": ["x"], "tokenized": [["x"]]}
    (persist_dir / "bm25_odd_coll.json").write_text(json.dumps(bad), encoding="utf-8")
    assert load_bm25_index("odd_coll") == (None, [], [])
    err = capsys.readouterr().err
    assert "odd_coll" in err
    assert "ValueError" in err


# --- ranking ----------------------------------------------------------------


def test_exact_names_rank_their_document_first(persist_dir):
    docs = [
        "Popping 比賽通常由評審依照音樂性與技巧判定勝負。",
        "Juste Debout 是每年在巴黎舉辦的街舞比賽，採 2 對 2 battle 形式。",
        "Boogaloo Sam 在 Fresno 創立 Electric Boogaloos。",
    ]
    ids = ["popping.txt::0", "juste.txt::0", "boogaloo.txt::0"]
    save_bm25_index(docs, ids, "rank_coll")
    bm25, loaded_ids, loaded_docs = load_bm25_index("rank_coll")

    scores = bm25.get_scores(tokenize_for_bm25("Juste Debout 比賽"))
    best = max(range(len(loaded_ids)), key=lambda i: scores[i])
    assert loaded_ids[best] == "juste.txt::0"
    assert loaded_docs == docs


# --- keeping the index in step with Chroma ----------------------------------


def test_refresh_rebuilds_from_the_collection_documents(persist_dir, monkeypatch):
    save_bm25_index(["stale"], ["old.txt::0"], "fresh_coll")
    coll = FakeWritableCollection(
        {"new.txt::0": ("Electric Boogaloos", {}), "new.txt::1": ("Popping", {})}
    )
    monkeypatch.setattr(rag_common, "get_chroma_collection", lambda name=None: coll)

    path = refresh_bm25_index("fresh_coll")

    assert path == persist_dir / "bm25_fresh_coll.json"
    _, ids, docs = load_bm25_index("fresh_coll")
    assert ids == ["new.txt::0", "new.txt::1"]
    assert docs == ["Electric Boogaloos", "Popping"]


@pytest.fixture
def data_dirs(tmp_path, monkeypatch, persist_dir):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    manifest = tmp_path / ".ingest_manifest.json"
    raw.mkdir()
    processed.mkdir()
    monkeypatch.setattr(data_update, "DATA_RAW", raw)
    monkeypatch.setattr(data_update, "DATA_PROCESSED", processed)
    monkeypatch.setattr(data_update, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(data_update, "embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])
    coll = FakeWritableCollection()
    monkeypatch.setattr(data_update, "get_chroma_collection", lambda name=None: coll)
    monkeypatch.setattr(rag_common, "get_chroma_collection", lambda name=None: coll)
    return raw, processed, manifest, coll


def test_rebuild_reusing_processed_text_leaves_data_untouched(data_dirs):
    raw, processed, manifest, coll = data_dirs
    (raw / "one.md").write_text("raw text that must not be used", encoding="utf-8")
    (processed / "one.txt").write_text("processed text for one", encoding="utf-8")
    (raw / "two.md").write_text("raw text two", encoding="utf-8")
    manifest.write_text('{"keep": "me"}', encoding="utf-8")

    data_update.rebuild_processed_and_index(
        chunk_fn=lambda t: [t], collection_name="reuse_coll", reuse_processed=True
    )

    assert coll.get(include=["documents"])["documents"] == [
        "processed text for one",
        "raw text two",
    ]
    assert manifest.read_text(encoding="utf-8") == '{"keep": "me"}'
    assert sorted(p.name for p in processed.iterdir()) == ["one.txt"]
    _, ids, docs = load_bm25_index("reuse_coll")
    assert ids == ["one.txt::0", "two.txt::0"]
    assert docs == ["processed text for one", "raw text two"]


def test_rebuild_without_reuse_still_resets_processed_and_manifest(data_dirs):
    raw, processed, manifest, coll = data_dirs
    (raw / "one.md").write_text("fresh raw text", encoding="utf-8")
    (processed / "stale.txt").write_text("left over", encoding="utf-8")
    manifest.write_text('{"stale.md": "0"}', encoding="utf-8")

    data_update.rebuild_processed_and_index(chunk_fn=lambda t: [t], collection_name="plain_coll")

    assert sorted(p.name for p in processed.iterdir()) == ["one.txt"]
    assert (processed / "one.txt").read_text(encoding="utf-8") == "fresh raw text"
    assert list(json.loads(manifest.read_text(encoding="utf-8"))) == ["one.md"]
    assert load_bm25_index("plain_coll")[1] == ["one.txt::0"]


def test_rebuild_with_no_documents_still_replaces_the_index(data_dirs):
    save_bm25_index(["stale"], ["old.txt::0"], "empty_after_coll")
    data_update.rebuild_processed_and_index(
        chunk_fn=lambda t: [t], collection_name="empty_after_coll", reuse_processed=True
    )
    assert load_bm25_index("empty_after_coll") == (None, [], [])


def test_incremental_update_refreshes_the_lexical_index(data_dirs):
    raw, _, _, coll = data_dirs
    save_bm25_index(["stale"], ["old.txt::0"], "incr_coll")
    (raw / "new.md").write_text("Electric Boogaloos history", encoding="utf-8")

    data_update.incremental_update(chunk_fn=lambda t: [t], collection_name="incr_coll")

    _, ids, docs = load_bm25_index("incr_coll")
    assert ids == ["new.txt::0"]
    assert docs == ["Electric Boogaloos history"]


def test_incremental_update_keeps_unchanged_documents_in_the_lexical_index(data_dirs):
    """全量刷新：未變動文件的 chunk 必須留在 BM25 索引中；縮短的文件不留舊 id。"""
    raw, _, manifest, coll = data_dirs
    (raw / "keep.md").write_text("Boogaloo Sam history", encoding="utf-8")
    (raw / "shrink.md").write_text("Popping now fits in one chunk", encoding="utf-8")
    coll.rows = {
        "keep.txt::0": ("Boogaloo Sam history", {"source_file": "keep.txt"}),
        "shrink.txt::0": ("old first half", {"source_file": "shrink.txt"}),
        "shrink.txt::1": ("old second half", {"source_file": "shrink.txt"}),
    }
    manifest.write_text(
        json.dumps({"keep.md": data_update._sha256_file(raw / "keep.md"), "shrink.md": "outdated"}),
        encoding="utf-8",
    )
    (raw / "new.md").write_text("Electric Boogaloos", encoding="utf-8")

    data_update.incremental_update(chunk_fn=lambda t: [t], collection_name="keep_coll")

    _, ids, docs = load_bm25_index("keep_coll")
    assert ids == ["keep.txt::0", "new.txt::0", "shrink.txt::0"]
    assert "Boogaloo Sam history" in docs
    assert "old second half" not in docs


def test_incremental_update_ignores_the_manifest_for_an_empty_collection(data_dirs, capsys):
    """manifest 是全域的；換 collection 或 persist 目錄後不可因 hash 相同而全部略過。"""
    raw, _, manifest, coll = data_dirs
    (raw / "one.md").write_text("Juste Debout", encoding="utf-8")
    manifest.write_text(json.dumps({"one.md": data_update._sha256_file(raw / "one.md")}), encoding="utf-8")

    data_update.incremental_update(chunk_fn=lambda t: [t], collection_name="fresh_empty_coll")

    assert coll.count() == 1
    assert load_bm25_index("fresh_empty_coll")[1] == ["one.txt::0"]
    assert "fresh_empty_coll" in capsys.readouterr().err


class FailingDeleteCollection(FakeWritableCollection):
    def delete(self, ids):
        raise ConnectionError("chroma unavailable")


def test_rebuild_stops_when_the_collection_cannot_be_cleared(data_dirs, monkeypatch):
    raw, processed, manifest, _ = data_dirs
    (raw / "one.md").write_text("fresh raw text", encoding="utf-8")
    (processed / "one.txt").write_text("committed text", encoding="utf-8")
    manifest.write_text('{"one.md": "0"}', encoding="utf-8")
    broken = FailingDeleteCollection({"one.txt::0": ("committed text", {"source_file": "one.txt"})})
    monkeypatch.setattr(data_update, "get_chroma_collection", lambda name=None: broken)

    with pytest.raises(RuntimeError, match="ConnectionError"):
        data_update.rebuild_processed_and_index(chunk_fn=lambda t: [t], collection_name="broken_coll")

    # 清空失敗時尚未動到 data/processed 與 manifest
    assert (processed / "one.txt").read_text(encoding="utf-8") == "committed text"
    assert manifest.read_text(encoding="utf-8") == '{"one.md": "0"}'
    assert broken.rows["one.txt::0"][0] == "committed text"


def test_incremental_update_stops_when_old_chunks_cannot_be_deleted(data_dirs, monkeypatch):
    raw, _, _, _ = data_dirs
    (raw / "one.md").write_text("changed text", encoding="utf-8")
    broken = FailingDeleteCollection({"one.txt::0": ("old text", {"source_file": "one.txt"})})
    monkeypatch.setattr(data_update, "get_chroma_collection", lambda name=None: broken)

    with pytest.raises(RuntimeError, match="one.txt"):
        data_update.incremental_update(chunk_fn=lambda t: [t], collection_name="broken_incr")
    assert broken.rows == {"one.txt::0": ("old text", {"source_file": "one.txt"})}


def test_re_extracting_in_memory_leaves_data_untouched(data_dirs):
    raw, processed, manifest, coll = data_dirs
    (raw / "one.md").write_text("fresh raw text", encoding="utf-8")
    (processed / "one.txt").write_text("committed text", encoding="utf-8")
    manifest.write_text('{"keep": "me"}', encoding="utf-8")

    hashes = data_update.rebuild_processed_and_index(
        chunk_fn=lambda t: [t], collection_name="mem_coll", reuse_processed=False, write_processed=False
    )

    assert coll.get(include=["documents"])["documents"] == ["fresh raw text"]
    assert (processed / "one.txt").read_text(encoding="utf-8") == "committed text"
    assert manifest.read_text(encoding="utf-8") == '{"keep": "me"}'
    assert hashes == {"one.txt": rag_common.text_sha256("fresh raw text")}


def test_rebuild_returns_hashes_of_the_indexed_text(data_dirs):
    raw, processed, _, _ = data_dirs
    (raw / "one.md").write_text("raw", encoding="utf-8")
    (processed / "one.txt").write_text("committed text", encoding="utf-8")

    hashes = data_update.rebuild_processed_and_index(
        chunk_fn=lambda t: [t], collection_name="hash_coll", reuse_processed=True
    )

    assert hashes == rag_common.processed_corpus_hashes(processed)


def test_cli_rebuild_can_reuse_the_committed_processed_text(data_dirs):
    raw, processed, manifest, coll = data_dirs
    (raw / "one.md").write_text("fresh raw text", encoding="utf-8")
    (processed / "one.txt").write_text("committed text", encoding="utf-8")

    assert data_update.main(["--rebuild", "--reuse-processed"]) == 0

    assert coll.get(include=["documents"])["documents"] == ["committed text"]
    assert (processed / "one.txt").read_text(encoding="utf-8") == "committed text"
    assert not manifest.exists()


def test_cli_reuse_processed_requires_rebuild(capsys):
    with pytest.raises(SystemExit) as exc:
        data_update.parse_args(["--reuse-processed"])
    assert exc.value.code == 2


def test_cli_errors_do_not_leak_absolute_paths(data_dirs, monkeypatch, capsys, tmp_path):
    def failing(**_kwargs):
        raise FileNotFoundError(2, "No such file or directory", str(tmp_path / "raw" / "gone.pdf"))

    monkeypatch.setattr(data_update, "rebuild_processed_and_index", failing)
    assert data_update.main(["--rebuild"]) == 1
    err = capsys.readouterr().err
    assert "raw/gone.pdf" in err
    assert str(tmp_path) not in err


# --- include: index only selected raw documents ------------------------------


def test_rebuild_can_index_only_the_included_documents(data_dirs):
    raw, processed, _, coll = data_dirs
    for name, text in (("one.md", "first"), ("two.md", "second"), ("three.md", "third")):
        (raw / name).write_text(f"raw {text}", encoding="utf-8")
        (processed / f"{name[:-3]}.txt").write_text(f"processed {text}", encoding="utf-8")

    hashes = data_update.rebuild_processed_and_index(
        chunk_fn=lambda t: [t], collection_name="subset_coll", reuse_processed=True, include={"two.md", "one.md"}
    )

    # 摘要（語料指紋的輸入）只含所選文件；插入順序仍依 data/raw 排序
    assert hashes == {
        "one.txt": rag_common.text_sha256("processed first"),
        "two.txt": rag_common.text_sha256("processed second"),
    }
    assert coll.get(include=["documents"])["documents"] == ["processed first", "processed second"]
    assert load_bm25_index("subset_coll")[1] == ["one.txt::0", "two.txt::0"]
    assert sorted(p.name for p in processed.iterdir()) == ["one.txt", "three.txt", "two.txt"]


def test_include_none_still_indexes_every_document(data_dirs):
    raw, _, _, coll = data_dirs
    (raw / "one.md").write_text("first", encoding="utf-8")
    (raw / "two.txt").write_text("second", encoding="utf-8")
    hashes = data_update.rebuild_processed_and_index(
        chunk_fn=lambda t: [t], collection_name="all_coll", reuse_processed=True, include=None
    )
    assert sorted(hashes) == ["one.txt", "two.txt"]
    assert coll.count() == 2


@pytest.mark.parametrize(
    ("include", "error", "match"),
    [
        ({"one.md", "missing.md"}, ValueError, "missing.md"),
        (set(), ValueError, "include"),
        ("one.md", TypeError, "include"),
    ],
)
def test_invalid_include_fails_before_the_collection_is_cleared(data_dirs, include, error, match):
    raw, _, _, coll = data_dirs
    (raw / "one.md").write_text("first", encoding="utf-8")
    coll.rows = {"old.txt::0": ("old text", {"source_file": "old.txt"})}

    with pytest.raises(error, match=match):
        data_update.rebuild_processed_and_index(
            chunk_fn=lambda t: [t], collection_name="bad_include", reuse_processed=True, include=include
        )
    assert coll.rows == {"old.txt::0": ("old text", {"source_file": "old.txt"})}


def test_include_is_refused_when_data_processed_would_be_rewritten(data_dirs):
    """寫回 data/processed 的 --rebuild 會先清空該目錄；只寫回部分文件會讓其他文件的已提交文字消失。"""
    raw, processed, manifest, coll = data_dirs
    (raw / "one.md").write_text("first", encoding="utf-8")
    (raw / "two.md").write_text("second", encoding="utf-8")
    (processed / "two.txt").write_text("committed two", encoding="utf-8")
    manifest.write_text('{"keep": "me"}', encoding="utf-8")

    with pytest.raises(ValueError, match="include"):
        data_update.rebuild_processed_and_index(chunk_fn=lambda t: [t], collection_name="persist_coll", include={"one.md"})

    assert (processed / "two.txt").read_text(encoding="utf-8") == "committed two"
    assert manifest.read_text(encoding="utf-8") == '{"keep": "me"}'
    assert coll.count() == 0


def test_include_works_when_re_extracting_in_memory(data_dirs):
    raw, processed, _, coll = data_dirs
    (raw / "one.md").write_text("fresh one", encoding="utf-8")
    (raw / "two.md").write_text("fresh two", encoding="utf-8")
    hashes = data_update.rebuild_processed_and_index(
        chunk_fn=lambda t: [t],
        collection_name="mem_subset",
        reuse_processed=False,
        write_processed=False,
        include=["two.md"],
    )
    assert hashes == {"two.txt": rag_common.text_sha256("fresh two")}
    assert coll.get(include=["documents"])["documents"] == ["fresh two"]
    assert list(processed.iterdir()) == []
