"""Dense and hybrid retrieval with Chroma, the embedding model and BM25 replaced by fakes.

The fakes mimic chromadb's return shapes: ``query()`` returns one list per query
embedding (lists of lists); ``get()`` returns flat lists and numpy arrays for
embeddings, like chromadb 1.x does.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import rag_query
from rag_common import tokenize_for_bm25


def unit(*xs: float) -> list[float]:
    v = np.asarray(xs, dtype=float)
    return (v / np.linalg.norm(v)).tolist()


def cosine_distance(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(1.0 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def chunk(source: str, index: int, text: str, embedding: list[float], section: str = "") -> dict:
    return {
        "id": f"{source}::{index}",
        "text": text,
        "embedding": embedding,
        "meta": {"source_file": source, "chunk_index": index, "section_title": section},
    }


class FakeCollection:
    """Cosine-space collection with chromadb-shaped query()/get() results."""

    def __init__(self, rows: list[dict]):
        self._rows = list(rows)
        self.query_calls = 0

    def query(self, query_embeddings, n_results, include):
        self.query_calls += 1
        q = query_embeddings[0]
        scored = sorted(
            ((cosine_distance(q, r["embedding"]), r) for r in self._rows),
            key=lambda pair: pair[0],
        )[:n_results]
        return {
            "ids": [[r["id"] for _, r in scored]],
            "documents": [[r["text"] for _, r in scored]],
            "metadatas": [[r["meta"] for _, r in scored]],
            "distances": [[d for d, _ in scored]],
            "embeddings": None,
        }

    def get(self, ids=None, include=None, where=None):
        wanted = None if ids is None else set(ids)
        rows = [r for r in self._rows if wanted is None or r["id"] in wanted]  # storage order
        include = include or []
        return {
            "ids": [r["id"] for r in rows],
            "documents": [r["text"] for r in rows] if "documents" in include else None,
            "metadatas": [r["meta"] for r in rows] if "metadatas" in include else None,
            "embeddings": (
                np.array([r["embedding"] for r in rows], dtype=np.float32)
                if "embeddings" in include
                else None
            ),
        }


class FakeBM25:
    def __init__(self, ids: list[str], scores: dict[str, float]):
        self._scores = np.array([scores.get(i, 0.0) for i in ids], dtype=float)
        self.seen_tokens: list[list[str]] = []

    def get_scores(self, tokens):
        self.seen_tokens.append(list(tokens))
        return self._scores.copy()


@pytest.fixture
def install(monkeypatch):
    def _install(rows, query_vectors, bm25_index=None):
        coll = FakeCollection(rows)
        monkeypatch.setattr(rag_query, "get_chroma_collection", lambda name=None: coll)
        monkeypatch.setattr(
            rag_query, "embed_texts", lambda texts: [list(query_vectors[t]) for t in texts]
        )
        index = bm25_index if bm25_index is not None else (None, [], [])
        monkeypatch.setattr(rag_query, "load_bm25_index", lambda name=None: index)
        return coll

    return _install


QUERY = "Juste Debout 比賽"
Q = unit(1, 0, 0)
EMB_C = unit(0.3, 0, 1)  # far from the query: cosine distance ~0.71 > 0.65
ROWS = [
    chunk("a.txt", 0, "街舞 battle 的基本介紹", unit(0.8, 0.3, 0)),
    chunk("b.txt", 0, "Juste Debout 是巴黎的街舞比賽", unit(0.9, 0.1, 0), "Juste Debout"),
    chunk("c.txt", 0, "Juste Debout 2002 冠軍名單", EMB_C, "Winners"),
    chunk("d.txt", 0, "Locking 的歷史", unit(0.7, 0.5, 0)),
    chunk("e.txt", 0, "無關內容", unit(0, 1, 0)),
]
BM25_IDS = [r["id"] for r in ROWS]


def bm25_index(scores):
    return FakeBM25(BM25_IDS, scores), BM25_IDS, [r["text"] for r in ROWS]


# --- hybrid -----------------------------------------------------------------


def test_lexical_only_chunk_enters_top_k_with_its_true_distance(install):
    install(ROWS, {QUERY: Q}, bm25_index({"c.txt::0": 5.0, "b.txt::0": 3.0}))

    context, cites = rag_query.hybrid_retrieve(QUERY, top_k=3, max_distance=0.65)

    assert [c["chunk_id"] for c in cites] == ["b.txt::0", "c.txt::0", "a.txt::0"]
    lexical = cites[1]
    assert lexical["retrieved_by"] == "bm25"
    assert lexical["distance"] == pytest.approx(cosine_distance(Q, EMB_C), abs=1e-6)
    assert lexical["distance"] != pytest.approx(0.9, abs=1e-3)
    assert lexical["source_file"] == "c.txt"
    assert lexical["chunk_index"] == 0
    assert lexical["section_title"] == "Winners"
    assert lexical["text"] == "Juste Debout 2002 冠軍名單"
    assert "[來源 2] 檔案: c.txt | chunk: 0 | 章節: Winners" in context


def test_chunk_in_both_lists_is_marked_both_and_ranks_first(install):
    install(ROWS, {QUERY: Q}, bm25_index({"c.txt::0": 5.0, "b.txt::0": 3.0}))

    _, cites = rag_query.hybrid_retrieve(QUERY, top_k=3, max_distance=0.65)

    assert cites[0]["chunk_id"] == "b.txt::0"
    assert cites[0]["retrieved_by"] == "both"
    assert cites[2]["retrieved_by"] == "dense"


def test_lexical_ranking_uses_the_shared_tokenizer_on_the_original_query(install):
    index = bm25_index({"c.txt::0": 5.0})
    install(ROWS, {QUERY: Q, "sub query": unit(0, 1, 0)}, index)

    rag_query.hybrid_retrieve(QUERY, top_k=3, max_distance=0.65, extra_queries=["sub query"])

    assert index[0].seen_tokens == [tokenize_for_bm25(QUERY)]


def test_missing_bm25_index_falls_back_to_dense_with_a_warning(install, capsys):
    install(ROWS, {QUERY: Q})

    _, cites = rag_query.hybrid_retrieve(
        QUERY, top_k=2, max_distance=0.65, collection_name="cypher_og_combo_f"
    )
    _, dense = rag_query.retrieve_context(
        QUERY, top_k=2, max_distance=0.65, collection_name="cypher_og_combo_f"
    )

    assert cites == dense
    assert all(c["retrieved_by"] == "dense" for c in cites)
    err = capsys.readouterr().err
    assert "BM25" in err
    assert "cypher_og_combo_f" in err
    assert "data_update.py --rebuild" in err


def test_strict_mode_raises_instead_of_falling_back_to_dense(install):
    coll = install(ROWS, {QUERY: Q})

    with pytest.raises(RuntimeError, match="cypher_og_combo_f"):
        rag_query.hybrid_retrieve(
            QUERY, top_k=2, max_distance=0.65, collection_name="cypher_og_combo_f", strict=True
        )
    assert coll.query_calls == 0  # 不先做一次用不到的向量檢索


def test_strict_mode_with_an_index_behaves_like_default(install):
    install(ROWS, {QUERY: Q}, bm25_index({"c.txt::0": 5.0, "b.txt::0": 3.0}))
    default = rag_query.hybrid_retrieve(QUERY, top_k=3, max_distance=0.65)
    strict = rag_query.hybrid_retrieve(QUERY, top_k=3, max_distance=0.65, strict=True)
    assert strict == default


def test_bm25_ids_missing_from_chroma_are_skipped_with_a_warning(install, capsys):
    ids = BM25_IDS + ["gone.txt::0"]
    index = (FakeBM25(ids, {"gone.txt::0": 9.0}), ids, [r["text"] for r in ROWS] + ["gone"])
    install(ROWS, {QUERY: Q}, index)

    _, cites = rag_query.hybrid_retrieve(QUERY, top_k=3, max_distance=0.65)

    assert "gone.txt::0" not in [c["chunk_id"] for c in cites]
    assert "gone.txt::0" in capsys.readouterr().err
    # 過期 id 被略過後由下一名遞補，仍回傳 top_k 筆
    assert [c["chunk_id"] for c in cites] == ["b.txt::0", "a.txt::0", "d.txt::0"]


def test_hybrid_returns_min_of_top_k_and_available_chunks(install):
    ids = BM25_IDS + ["gone.txt::0", "gone.txt::1"]
    index = (FakeBM25(ids, {"gone.txt::0": 9.0, "gone.txt::1": 8.0, "c.txt::0": 1.0}), ids, [""] * len(ids))
    install(ROWS, {QUERY: Q}, index)

    _, five = rag_query.hybrid_retrieve(QUERY, top_k=5, max_distance=0.65)
    _, many = rag_query.hybrid_retrieve(QUERY, top_k=10, max_distance=0.65)

    assert len(five) == 4  # b, a, d（dense）+ c（BM25）；兩個過期 id 不算
    assert {c["chunk_id"] for c in five} == {"a.txt::0", "b.txt::0", "c.txt::0", "d.txt::0"}
    assert len(many) == 4


def test_zero_bm25_scores_add_nothing(install):
    install(ROWS, {QUERY: Q}, bm25_index({}))

    _, cites = rag_query.hybrid_retrieve(QUERY, top_k=3, max_distance=0.65)

    assert [c["chunk_id"] for c in cites] == ["b.txt::0", "a.txt::0", "d.txt::0"]
    assert all(c["retrieved_by"] == "dense" for c in cites)


# --- dense ------------------------------------------------------------------


def test_dense_cites_carry_chunk_id_and_retriever(install):
    install(ROWS, {QUERY: Q})

    _, cites = rag_query.retrieve_context(QUERY, top_k=2, max_distance=0.65)

    assert cites[0] == {
        "chunk_id": "b.txt::0",
        "source_file": "b.txt",
        "chunk_index": 0,
        "distance": pytest.approx(cosine_distance(Q, ROWS[1]["embedding"])),
        "section_title": "Juste Debout",
        "text": "Juste Debout 是巴黎的街舞比賽",
        "retrieved_by": "dense",
    }


def test_truncate_to_reproduces_plain_top_k(install):
    rows = [chunk("s.txt", i, f"chunk {i}", unit(1, 0.1 * i, 0)) for i in range(12)]
    install(rows, {QUERY: Q})

    truncated = rag_query.retrieve_context(QUERY, top_k=10, max_distance=1.0, truncate_to=5)
    plain = rag_query.retrieve_context(QUERY, top_k=5, max_distance=1.0)

    assert truncated == plain
    assert len(plain[1]) == 5


def test_invalid_truncate_to_is_rejected(install):
    install(ROWS, {QUERY: Q})
    with pytest.raises(ValueError):
        rag_query.retrieve_context(QUERY, top_k=5, truncate_to=0)


def test_multi_query_merge_keeps_the_minimum_distance(install):
    q0, q1 = unit(1, 0, 0), unit(0, 1, 0)
    emb_x, emb_y = unit(0.6, 0.8, 0), unit(0.75, 0, 0.66)
    rows = [
        chunk("x.txt", 0, "X", emb_x),
        chunk("y.txt", 0, "Y", emb_y),
        chunk("z.txt", 0, "Z", unit(0, 0, 1)),
    ]
    install(rows, {"q0": q0, "q1": q1})

    _, cites = rag_query.retrieve_context("q0", top_k=2, max_distance=0.65, extra_queries=["q1"])

    assert [c["chunk_id"] for c in cites] == ["x.txt::0", "y.txt::0"]
    assert cites[0]["distance"] == pytest.approx(cosine_distance(q1, emb_x))
    assert cites[1]["distance"] == pytest.approx(cosine_distance(q0, emb_y))


def test_cosine_distances_to_query_reads_numpy_embeddings(install):
    install(ROWS, {QUERY: Q})

    got = rag_query.cosine_distances_to_query(QUERY, ["c.txt::0", "a.txt::0", "nope::0"])

    assert set(got) == {"c.txt::0", "a.txt::0"}
    assert got["c.txt::0"] == pytest.approx(cosine_distance(Q, EMB_C), abs=1e-6)
    assert got["a.txt::0"] == pytest.approx(cosine_distance(Q, ROWS[0]["embedding"]), abs=1e-6)
    assert all(isinstance(v, float) for v in got.values())


# --- query expansion --------------------------------------------------------


def completion(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_expand_query_returns_the_original_and_up_to_three_sub_queries(monkeypatch):
    monkeypatch.setattr(
        rag_query, "litellm_chat_completion", lambda **k: completion("one\n\ntwo\n原始\nthree\nfour")
    )
    assert rag_query.expand_query("原始", None, strict=True) == ["原始", "one", "two"]


def test_expand_query_falls_back_by_default_but_raises_when_strict(monkeypatch, capsys):
    def down(**_kwargs):
        raise TimeoutError("provider timeout")

    monkeypatch.setattr(rag_query, "litellm_chat_completion", down)

    assert rag_query.expand_query("Q?", None) == ["Q?"]
    assert "TimeoutError" in capsys.readouterr().err
    with pytest.raises(TimeoutError):
        rag_query.expand_query("Q?", None, strict=True)


def test_cosine_distances_to_query_with_no_ids_does_no_work(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("should not be called")

    monkeypatch.setattr(rag_query, "embed_texts", boom)
    monkeypatch.setattr(rag_query, "get_chroma_collection", boom)
    assert rag_query.cosine_distances_to_query("anything", []) == {}
