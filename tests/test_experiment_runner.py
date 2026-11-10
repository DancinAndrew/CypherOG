"""experiment_runner: output-directory guard, strict runs, results contract and summary building."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

import experiment_runner as er
import rag_eval
import rag_query
from rag_common import document_list_digest, load_corpus_definition, load_corpus_pin, pinned_digest_for

ROUND1_DOCS = load_corpus_definition()["documents"]

CITE_KEYS = {
    "chunk_id",
    "source_file",
    "chunk_index",
    "section_title",
    "distance",
    "distance_to_question",
    "retrieved_by",
    "text",
}
COMBO_KEYS = {
    "id",
    "name",
    "chunk_desc",
    "top_k",
    "max_distance",
    "truncate_to",
    "prompt_style",
    "use_hybrid",
    "use_expand",
    "collection",
}


# --- --out guard ------------------------------------------------------------


def test_out_pointing_at_experiments_is_refused():
    with pytest.raises(ValueError):
        er.resolve_out_dir(er.EXPERIMENTS_DIR)
    with pytest.raises(ValueError):
        er.resolve_out_dir(str(er.EXPERIMENTS_DIR / "round2" / ".."))


# 大小寫不同的拼法：macOS 的 APFS 預設不分大小寫，這些路徑指向同一個 experiments/
CASE_VARIANTS_OF_EXPERIMENTS = ("EXPERIMENTS", "Experiments", "./Experiments/", "eXperiments/.")
CASE_VARIANTS_OF_ROUND1 = (
    "experiments/COMBO_A",
    "Experiments/combo_e",
    "EXPERIMENTS/combo_e",
    "experiments/Combo_E",
    "experiments/Combo_a/sub",
)


@pytest.mark.parametrize("spelling", ("experiments", "./experiments/", "experiments/.", *CASE_VARIANTS_OF_EXPERIMENTS))
def test_relative_spellings_of_experiments_are_refused(monkeypatch, spelling):
    monkeypatch.chdir(er.ROOT)
    with pytest.raises(ValueError, match="experiments/ 本身"):
        er.resolve_out_dir(spelling)


@pytest.mark.parametrize(
    "spelling", ("experiments/combo_a", "experiments/combo_e/nested", *CASE_VARIANTS_OF_ROUND1)
)
def test_round1_combo_directories_are_refused(monkeypatch, spelling):
    monkeypatch.chdir(er.ROOT)
    with pytest.raises(ValueError, match="第一輪結果目錄"):
        er.resolve_out_dir(spelling)


@pytest.mark.parametrize("spelling", (".", "data", "data/raw", "data/processed/new", "tests", "chroma_db"))
def test_repo_paths_outside_experiments_are_refused(monkeypatch, spelling):
    monkeypatch.chdir(er.ROOT)
    with pytest.raises(ValueError, match="experiments/ 之下"):
        er.resolve_out_dir(spelling)


def test_paths_that_do_not_exist_are_compared_case_insensitively(tmp_path):
    """samefile 只適用於已存在的路徑；不存在時（或大小寫敏感的檔案系統上）改用 casefold 比對。"""
    assert er._same_dir(tmp_path / "missing" / "EXPERIMENTS", tmp_path / "missing" / "experiments")
    assert not er._same_dir(tmp_path / "missing" / "round2", tmp_path / "missing" / "experiments")


def test_an_existing_file_is_refused():
    with pytest.raises(ValueError, match="檔案"):
        er.resolve_out_dir(er.EXPERIMENTS_DIR / "summary.md")


def test_subdirectories_of_experiments_are_allowed(tmp_path, monkeypatch):
    assert er.resolve_out_dir(er.EXPERIMENTS_DIR / "round2") == er.EXPERIMENTS_DIR / "round2"
    assert er.resolve_out_dir(tmp_path / "out") == (tmp_path / "out").resolve()
    monkeypatch.chdir(er.ROOT)
    assert er.resolve_out_dir("experiments/round2/") == er.EXPERIMENTS_DIR / "round2"
    assert er.resolve_out_dir("experiments/latest_combo_x").name == "latest_combo_x"


def test_main_refuses_experiments_before_doing_any_work(monkeypatch, capsys):
    def boom(*_args, **_kwargs):
        raise AssertionError("no combo may run")

    monkeypatch.setattr(er, "run_combo", boom)
    monkeypatch.chdir(er.ROOT)
    for out in (str(er.EXPERIMENTS_DIR), "Experiments", "EXPERIMENTS/combo_e", "data/raw"):
        assert er.main(["--combos", "e", "--out", out]) == 2
    assert "experiments" in capsys.readouterr().err


def test_main_refuses_an_existing_file_without_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(er, "run_combo", lambda *a, **k: pytest.fail("no combo may run"))
    out = str(er.EXPERIMENTS_DIR / "summary.md")
    assert er.main(["--combos", "e", "--out", out, "--retrieval-only"]) == 2
    assert "summary.md" in capsys.readouterr().err


def test_run_combo_and_summary_reapply_the_guard(monkeypatch):
    """第二道防線：即使繞過 main，也不會在第一輪目錄內寫入或搬動任何東西。"""
    q1 = er.EXPERIMENTS_DIR / "combo_e" / "q1.md"
    before = q1.stat().st_mtime_ns
    monkeypatch.setattr(er, "rebuild_processed_and_index", lambda **k: pytest.fail("must not rebuild"))
    with pytest.raises(ValueError):
        er.run_combo(er.COMBO_MAP["e"], er.ROOT / "EXPERIMENTS", model="m")
    with pytest.raises(ValueError):
        er.generate_summary(er.ROOT / "Experiments")
    assert q1.stat().st_mtime_ns == before
    assert not (er.EXPERIMENTS_DIR / ".previous").exists()


# --- combo table ------------------------------------------------------------


def test_combos_use_truncate_to_and_current_corpus_collections_by_default():
    for combo in er.COMBOS:
        assert "rerank" not in combo
        assert combo["collection"] == f"cypher_og_current_combo_{combo['id']}"
    assert er.COMBO_MAP["d"]["truncate_to"] == 5
    assert er.COMBO_MAP["d"]["top_k"] == 10
    assert er.COMBO_MAP["d"]["name"] == "Combo D — 小 chunk + top-10 距離截斷"
    assert all(c["truncate_to"] is None for c in er.COMBOS if c["id"] != "d")


# --- one question -----------------------------------------------------------


def dense_cite(chunk_id: str, distance: float, retrieved_by: str = "dense") -> dict:
    source, index = chunk_id.split("::")
    return {
        "chunk_id": chunk_id,
        "source_file": source,
        "chunk_index": int(index),
        "distance": distance,
        "section_title": "",
        "text": f"text of {chunk_id}",
        "retrieved_by": retrieved_by,
    }


@pytest.fixture
def no_llm(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("the LLM must not be called")

    monkeypatch.setattr(er, "litellm_chat_completion", boom)


def test_retrieval_only_records_distances_to_the_original_question(monkeypatch, no_llm):
    seen = {}

    def fake_retrieve(question, **kwargs):
        seen["retrieve"] = kwargs
        return "ctx", [dense_cite("a.txt::0", 0.2), dense_cite("b.txt::3", 0.3)]

    def fake_distances(question, chunk_ids, collection_name=None):
        seen["distances"] = (question, list(chunk_ids), collection_name)
        return {"a.txt::0": 0.21, "b.txt::3": 0.31}

    monkeypatch.setattr(er, "retrieve_context", fake_retrieve)
    monkeypatch.setattr(er, "cosine_distances_to_query", fake_distances)

    result = er._run_query(er.COMBO_MAP["d"], "Q?", retrieval_only=True)

    assert seen["retrieve"]["truncate_to"] == 5
    assert seen["distances"] == ("Q?", ["a.txt::0", "b.txt::3"], "cypher_og_current_combo_d")
    assert result["answer"] == ""
    assert result["error"] is None
    assert result["expanded_queries"] is None
    assert [c["distance_to_question"] for c in result["cites"]] == [0.21, 0.31]
    assert all(set(c) == CITE_KEYS for c in result["cites"])


def test_query_expansion_is_recorded_and_distances_use_the_original_question(monkeypatch, no_llm):
    seen = {}

    def fake_expand(q, model=None, *, strict=False):
        seen["strict"] = strict
        return [q, "sub one", "sub two"]

    monkeypatch.setattr(er, "expand_query", fake_expand)

    def fake_retrieve(question, **kwargs):
        seen["extra"] = kwargs["extra_queries"]
        return "ctx", [dense_cite("a.txt::0", 0.1)]

    def fake_distances(question, chunk_ids, collection_name=None):
        seen["question"] = question
        return {"a.txt::0": 0.4}

    monkeypatch.setattr(er, "retrieve_context", fake_retrieve)
    monkeypatch.setattr(er, "cosine_distances_to_query", fake_distances)

    result = er._run_query(er.COMBO_MAP["g"], "原始問題", retrieval_only=True)

    assert seen == {"strict": True, "extra": ["sub one", "sub two"], "question": "原始問題"}
    assert result["expanded_queries"] == ["sub one", "sub two"]
    assert result["error"] is None
    assert result["cites"][0]["distance"] == 0.1
    assert result["cites"][0]["distance_to_question"] == 0.4


@pytest.mark.parametrize("retrieval_only", (True, False))
def test_failed_expansion_is_recorded_and_never_runs_plain_dense(monkeypatch, capsys, retrieval_only):
    """供應商錯誤（例如 429）時，G 不可默默變成 E：該題記為 expansion_failed、不檢索、不生成。"""

    def provider_down(**_kwargs):
        raise RuntimeError("429 rate limited; key sk-should-not-leak")

    monkeypatch.setattr(rag_query, "litellm_chat_completion", provider_down)
    monkeypatch.setattr(er, "litellm_chat_completion", lambda **k: pytest.fail("no answer call"))
    monkeypatch.setattr(er, "retrieve_context", lambda *a, **k: pytest.fail("must not retrieve"))

    result = er._run_query(er.COMBO_MAP["g"], "Q?", retrieval_only=retrieval_only)

    assert result["error"] == "expansion_failed: RuntimeError"
    assert result["expanded_queries"] is None
    assert result["cites"] == []
    assert "sk-should-not-leak" not in result["answer"]
    err = capsys.readouterr().err
    assert "RuntimeError" in err
    assert "sk-should-not-leak" not in err


def test_expansion_without_new_sub_queries_is_a_failure(monkeypatch, no_llm):
    monkeypatch.setattr(er, "expand_query", lambda q, model=None, *, strict=False: [q])
    monkeypatch.setattr(er, "retrieve_context", lambda *a, **k: pytest.fail("must not retrieve"))

    result = er._run_query(er.COMBO_MAP["g"], "Q?", retrieval_only=True)

    assert result["error"] == "expansion_failed: no_subqueries"
    assert result["cites"] == []


def test_hybrid_combo_retrieves_in_strict_mode(monkeypatch, no_llm):
    seen = {}

    def fake_hybrid(question, **kwargs):
        seen.update(kwargs)
        return "ctx", [dense_cite("a.txt::0", 0.2, "both")]

    monkeypatch.setattr(er, "hybrid_retrieve", fake_hybrid)
    monkeypatch.setattr(er, "cosine_distances_to_query", lambda q, ids, collection_name=None: {"a.txt::0": 0.2})

    er._run_query(er.COMBO_MAP["f"], "Q?", retrieval_only=True)

    assert seen["strict"] is True
    assert seen["collection_name"] == "cypher_og_current_combo_f"


def test_hybrid_combo_without_a_bm25_index_raises_instead_of_falling_back(monkeypatch, no_llm):
    monkeypatch.setattr(rag_query, "load_bm25_index", lambda name=None: (None, [], []))
    monkeypatch.setattr(rag_query, "retrieve_context", lambda *a, **k: pytest.fail("must not fall back"))
    with pytest.raises(RuntimeError, match="cypher_og_current_combo_f"):
        er._run_query(er.COMBO_MAP["f"], "Q?", retrieval_only=True)


def test_a_cite_without_a_question_distance_fails_loudly(monkeypatch, no_llm):
    monkeypatch.setattr(er, "retrieve_context", lambda q, **k: ("ctx", [dense_cite("a.txt::0", 0.1)]))
    monkeypatch.setattr(er, "cosine_distances_to_query", lambda q, ids, collection_name=None: {})
    with pytest.raises(RuntimeError):
        er._run_query(er.COMBO_MAP["e"], "Q?", retrieval_only=True)


def test_llm_failure_is_reported_by_type_only(monkeypatch, capsys):
    monkeypatch.setattr(er, "retrieve_context", lambda q, **k: ("ctx", [dense_cite("a.txt::0", 0.1)]))
    monkeypatch.setattr(er, "cosine_distances_to_query", lambda q, ids, collection_name=None: {"a.txt::0": 0.1})

    def failing(**_kwargs):
        raise RuntimeError("secret-looking detail sk-should-not-leak")

    monkeypatch.setattr(er, "litellm_chat_completion", failing)

    result = er._run_query(er.COMBO_MAP["e"], "Q?")

    assert result["error"] == "llm_error: RuntimeError"
    assert "sk-should-not-leak" not in result["answer"]
    assert "RuntimeError" in capsys.readouterr().err


# --- one combo --------------------------------------------------------------


class FakeCountCollection:
    def __init__(self, n: int = 7):
        self.n = n

    def count(self):
        return self.n


def fake_completion(**_kwargs):
    message = SimpleNamespace(content="生成的回答")
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


TEXT_HASHES = {"a.txt": "1" * 64, "c.txt": "2" * 64}
FAKE_ENV = {"git_commit": "abc1234", "git_dirty": False, "embedding_model": "emb", "text_source": "data/processed"}


@pytest.fixture
def fake_index(monkeypatch):
    """假的重建與 collection：rebuild 回傳文字指紋，collection 有 7 個 chunk，BM25 索引也有 7 個。"""
    rebuilds: list[dict] = []

    def fake_rebuild(**kw):
        rebuilds.append(kw)
        return dict(TEXT_HASHES)

    monkeypatch.setattr(er, "rebuild_processed_and_index", fake_rebuild)
    monkeypatch.setattr(er, "get_chroma_collection", lambda name=None: FakeCountCollection(7))
    monkeypatch.setattr(er, "load_bm25_index", lambda name=None: (object(), [f"x::{i}" for i in range(7)], []))
    return rebuilds


def install_f_retrieval(monkeypatch, answer: str = "生成的回答"):
    monkeypatch.setattr(
        er,
        "hybrid_retrieve",
        lambda q, **k: ("ctx", [dense_cite("a.txt::0", 0.2, "both"), dense_cite("c.txt::1", 0.7, "bm25")]),
    )
    monkeypatch.setattr(
        er,
        "cosine_distances_to_query",
        lambda q, ids, collection_name=None: {cid: 0.5 for cid in ids},
    )
    message = SimpleNamespace(content=answer)
    monkeypatch.setattr(
        er, "litellm_chat_completion", lambda **k: SimpleNamespace(choices=[SimpleNamespace(message=message)])
    )


def test_run_combo_writes_the_results_contract(tmp_path, monkeypatch, fake_index):
    install_f_retrieval(monkeypatch)

    info = er.run_combo(er.COMBO_MAP["f"], tmp_path, model="gpt-test", environment=FAKE_ENV)

    assert info == {
        "chunk_count": 7,
        "corpus_sha256": er.corpus_digest(TEXT_HASHES),
        "status": "ok",
        "failed_questions": [],
    }
    rebuilds = fake_index
    assert rebuilds[0]["collection_name"] == "cypher_og_current_combo_f"
    assert rebuilds[0]["reuse_processed"] is True
    assert rebuilds[0]["write_processed"] is False
    combo_dir = tmp_path / "combo_f"
    assert sorted(p.name for p in combo_dir.iterdir()) == [
        "config.md",
        "q1.md",
        "q2.md",
        "q3.md",
        "q4.md",
        "q5.md",
        "results.json",
    ]
    payload = json.loads((combo_dir / "results.json").read_text(encoding="utf-8"))
    assert set(payload) == {"combo", "model", "retrieval_only", "environment", "generated_at", "questions"}
    assert payload["environment"] == {
        **FAKE_ENV,
        "corpus_sha256": er.corpus_digest(TEXT_HASHES),
        "chunk_count": 7,
        "index_rebuilt": True,
    }
    assert set(payload["combo"]) == COMBO_KEYS
    assert payload["combo"]["id"] == "f"
    assert payload["model"] == "gpt-test"
    assert payload["retrieval_only"] is False
    assert [q["index"] for q in payload["questions"]] == [1, 2, 3, 4, 5]
    assert [q["question"] for q in payload["questions"]] == er.QUESTIONS
    first = payload["questions"][0]
    assert set(first) == {"index", "question", "expanded_queries", "cites", "answer", "error"}
    assert first["answer"] == "生成的回答"
    assert [c["retrieved_by"] for c in first["cites"]] == ["both", "bm25"]
    assert all(set(c) == CITE_KEYS for c in first["cites"])

    q1 = (combo_dir / "q1.md").read_text(encoding="utf-8")
    assert q1.startswith("# Q1: What is Popping and who created it?")
    assert "gpt-test" in q1
    assert "與原問題距離: 0.500" in q1
    assert "bm25" in q1


def test_run_combo_skip_rebuild_and_re_extract_flags(tmp_path, monkeypatch, no_llm, fake_index):
    rebuilds = fake_index
    monkeypatch.setattr(er, "retrieve_context", lambda q, **k: ("", []))
    monkeypatch.setattr(er, "cosine_distances_to_query", lambda q, ids, collection_name=None: {})

    info = er.run_combo(
        er.COMBO_MAP["e"], tmp_path, model="m", skip_rebuild=True, retrieval_only=True, environment=FAKE_ENV
    )
    assert rebuilds == []
    assert info["corpus_sha256"] == er.corpus_digest(er.processed_corpus_hashes())
    skipped = json.loads((tmp_path / "combo_e" / "results.json").read_text(encoding="utf-8"))
    assert skipped["environment"]["index_rebuilt"] is False

    er.run_combo(er.COMBO_MAP["e"], tmp_path, model="m", retrieval_only=True, reuse_processed=False)
    assert rebuilds[0]["reuse_processed"] is False
    assert rebuilds[0]["write_processed"] is False  # --re-extract 只在記憶體中抽取

    payload = json.loads((tmp_path / "combo_e" / "results.json").read_text(encoding="utf-8"))
    assert payload["retrieval_only"] is True
    assert payload["environment"]["text_source"] == "data/raw (re-extracted in memory)"
    assert set(payload["environment"]) >= {"git_commit", "git_dirty", "embedding_model"}
    assert all(q["error"] == "no_context" for q in payload["questions"])  # no_context 不算失敗


def test_empty_collection_fails_the_combo(tmp_path, monkeypatch, no_llm, fake_index):
    monkeypatch.setattr(er, "get_chroma_collection", lambda name=None: FakeCountCollection(0))
    monkeypatch.setattr(er, "retrieve_context", lambda *a, **k: pytest.fail("must not query"))
    with pytest.raises(er.ExperimentError, match="空的"):
        er.run_combo(er.COMBO_MAP["e"], tmp_path, model="m", skip_rebuild=True, environment=FAKE_ENV)
    assert not (tmp_path / "combo_e" / "results.json").exists()


@pytest.mark.parametrize(
    ("index", "match"),
    [((None, [], []), "BM25 索引"), ((object(), ["x::0", "x::1"], []), "2 個 chunk")],
)
def test_hybrid_combo_needs_a_current_bm25_index(tmp_path, monkeypatch, no_llm, fake_index, index, match):
    monkeypatch.setattr(er, "load_bm25_index", lambda name=None: index)
    monkeypatch.setattr(er, "hybrid_retrieve", lambda *a, **k: pytest.fail("must not query"))
    with pytest.raises(er.ExperimentError, match=match):
        er.run_combo(er.COMBO_MAP["f"], tmp_path, model="m", skip_rebuild=True, environment=FAKE_ENV)
    assert not (tmp_path / "combo_f" / "results.json").exists()


def test_llm_failures_mark_the_combo_degraded(tmp_path, monkeypatch, fake_index):
    install_f_retrieval(monkeypatch)
    calls = {"n": 0}

    def flaky(**_kwargs):
        calls["n"] += 1
        if calls["n"] in (2, 4):
            raise RuntimeError("upstream timeout")
        return fake_completion()

    monkeypatch.setattr(er, "litellm_chat_completion", flaky)

    info = er.run_combo(er.COMBO_MAP["f"], tmp_path, model="m", environment=FAKE_ENV)

    assert info["status"] == "degraded"
    assert info["failed_questions"] == [2, 4]
    q2 = (tmp_path / "combo_f" / "q2.md").read_text(encoding="utf-8")
    assert "本題失敗：`llm_error: RuntimeError`" in q2
    summary = er.generate_summary(tmp_path).read_text(encoding="utf-8")
    assert "degraded（Q2、Q4）" in summary


def test_results_round_trip_into_rag_eval(tmp_path, monkeypatch, fake_index):
    """experiment_runner 寫出的 results.json 必須能被 rag_eval 直接讀取與計分（契約不靠手寫 fixture）。"""
    install_f_retrieval(monkeypatch, answer="成員有 Boogaloo Sam 與 Popin' Pete。")
    monkeypatch.setattr(
        er,
        "hybrid_retrieve",
        lambda q, **k: ("ctx", [dense_cite("a.txt::0", 0.2, "both"), {**dense_cite("c.txt::1", 0.7, "bm25"), "text": "Skeeter Rabbit"}]),
    )
    corpus = er.load_corpus_selection("round1")
    environment = er.build_environment(git={"commit": "abc1234", "dirty": False}, reuse_processed=True, corpus=corpus)
    er.run_combo(
        er.combo_for_corpus(er.COMBO_MAP["f"], "round1"), tmp_path, model="gpt-test",
        environment=environment, corpus=corpus,
    )

    run = rag_eval.load_round2_run(tmp_path / "combo_f" / "results.json")
    gold = rag_eval.load_gold(rag_eval.GOLD_PATH)
    (row,) = rag_eval.compute_metrics([run], gold)

    assert row["run"] == "F"
    assert row["model"] == "gpt-test"
    assert row["mean_distance_to_question"] == pytest.approx(0.5)
    assert row["questions_with_cites"] == 5
    assert row["q2_members_named"] == 2
    assert row["q2_members_in_context"] == 1
    assert row["failed_questions"] == []
    assert run["consistency"]["corpus_sha256"] == er.corpus_digest(TEXT_HASHES)
    assert run["corpus"] == "round1"
    assert run["consistency"]["document_count"] == 39
    assert run["consistency"]["documents_digest"] == document_list_digest(ROUND1_DOCS)
    assert "dense channel only" in row["retrieval"]


def test_a_failed_rerun_does_not_leave_the_previous_results_behind(tmp_path, monkeypatch, fake_index):
    install_f_retrieval(monkeypatch, answer="OLD ANSWER")
    monkeypatch.setattr(er, "_git_state", lambda: {"commit": "abc1234", "dirty": False})
    assert er.main(["--combos", "f", "--out", str(tmp_path)]) == 0
    assert "OLD ANSWER" in (tmp_path / "summary.md").read_text(encoding="utf-8")

    def broken_rebuild(**_kw):
        raise RuntimeError("chroma exploded")

    monkeypatch.setattr(er, "rebuild_processed_and_index", broken_rebuild)
    assert er.main(["--combos", "f", "--out", str(tmp_path)]) == 1

    assert not (tmp_path / "combo_f" / "results.json").exists()
    assert rag_eval.load_round2_runs(tmp_path) == []
    assert not (tmp_path / "summary.md").exists()
    previous = sorted(p.name for p in (tmp_path / ".previous").iterdir())
    assert len(previous) == 2 and previous[0].startswith("combo_f-") and previous[1].startswith("summary-")
    kept = json.loads(
        (tmp_path / ".previous" / previous[0] / "results.json").read_text(encoding="utf-8")
    )
    assert kept["questions"][0]["answer"] == "OLD ANSWER"
    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run["combos"]["f"]["status"] == "failed: RuntimeError"


def test_reruns_in_the_same_second_keep_every_previous_copy(tmp_path):
    for n in range(3):
        target = tmp_path / "combo_f"
        target.mkdir()
        (target / "marker").write_text(str(n), encoding="utf-8")
        er.retire_previous(target)
    copies = sorted((tmp_path / ".previous").iterdir())
    assert sorted((c / "marker").read_text(encoding="utf-8") for c in copies) == ["0", "1", "2"]


# --- summary ----------------------------------------------------------------


def fake_results(combo_id: str, dists: list[float], answer_prefix: str) -> dict:
    combo = {k: er.COMBO_MAP[combo_id][k] for k in COMBO_KEYS}
    return {
        "combo": combo,
        "model": "gpt-test",
        "retrieval_only": False,
        "generated_at": "2026-11-09T00:00:00+00:00",
        "questions": [
            {
                "index": i,
                "question": q,
                "expanded_queries": None,
                "cites": [
                    {
                        "chunk_id": f"doc{i}.txt::{n}",
                        "source_file": f"doc{i}.txt",
                        "chunk_index": n,
                        "section_title": "",
                        "distance": d,
                        "distance_to_question": d,
                        "retrieved_by": "dense",
                        "text": "t",
                    }
                    for n, d in enumerate(dists)
                ],
                "answer": f"{answer_prefix}-{i}",
                "error": None,
            }
            for i, q in enumerate(er.QUESTIONS, 1)
        ],
    }


def write_results(out_dir, payload) -> None:
    combo_dir = out_dir / f"combo_{payload['combo']['id']}"
    combo_dir.mkdir(parents=True)
    (combo_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def test_summary_is_built_from_every_results_file_in_out(tmp_path):
    write_results(tmp_path, fake_results("f", [0.2, 0.3], "answer-f"))
    write_results(tmp_path, fake_results("e", [0.4, 0.5], "answer-e"))
    (tmp_path / "notes").mkdir()

    path = er.generate_summary(tmp_path)

    assert path == tmp_path / "summary.md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# RAG 2 組實驗比較（E、F）")
    assert "answer-e-1" in text and "answer-f-5" in text
    assert "0.250" in text and "0.450" in text
    assert text.index("Combo E") < text.index("Combo F")


def test_separate_runs_still_produce_one_summary(tmp_path):
    write_results(tmp_path, fake_results("f", [0.2], "answer-f"))
    er.generate_summary(tmp_path)
    write_results(tmp_path, fake_results("g", [0.3], "answer-g"))

    text = er.generate_summary(tmp_path).read_text(encoding="utf-8")

    assert text.startswith("# RAG 2 組實驗比較（F、G）")
    assert "answer-f-1" in text and "answer-g-1" in text


def test_unreadable_results_are_skipped_with_a_warning(tmp_path, capsys):
    write_results(tmp_path, fake_results("e", [0.4], "answer-e"))
    broken = tmp_path / "combo_x"
    broken.mkdir()
    (broken / "results.json").write_text("{oops", encoding="utf-8")

    text = er.generate_summary(tmp_path).read_text(encoding="utf-8")

    assert text.startswith("# RAG 1 組實驗比較（E）")
    assert "JSONDecodeError" in capsys.readouterr().err


# --- run.json ---------------------------------------------------------------


def ok_info(chunk_count: int = 11) -> dict:
    return {"chunk_count": chunk_count, "corpus_sha256": "d" * 64, "status": "ok", "failed_questions": []}


def test_main_writes_run_metadata_without_secrets(tmp_path, monkeypatch):
    def fake_run_combo(combo, out_dir, **kwargs):
        write_results(out_dir, fake_results(combo["id"], [0.2], f"answer-{combo['id']}"))
        return ok_info()

    monkeypatch.setattr(er, "run_combo", fake_run_combo)
    monkeypatch.setenv("LITELLM_MODEL", "gpt-test-model")
    monkeypatch.setenv("LITELLM_API_KEY", "dummy-key-for-test")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://proxy.invalid")

    assert er.main(["--combos", "f", "--out", str(tmp_path)]) == 0
    raw = (tmp_path / "run.json").read_text(encoding="utf-8")
    run = json.loads(raw)

    assert "dummy-key-for-test" not in raw
    assert "proxy.invalid" not in raw
    assert run["llm"] == {"model": "gpt-test-model", "proxy_configured": True}
    assert run["combos"]["f"]["chunk_count"] == 11
    assert run["combos"]["f"]["collection"] == "cypher_og_current_combo_f"
    assert run["questions"] == er.QUESTIONS
    assert run["corpus"]["name"] == "current"  # 預設語料：data/raw 全部文件
    assert run["corpus"]["document_count"] == 42
    assert run["corpus"]["documents"] == [p.name for p in er.list_raw_documents()]
    assert run["corpus"]["documents_digest"] == document_list_digest(run["corpus"]["documents"])
    assert run["corpus"]["matches_pin"] is True
    assert len(run["corpus"]["processed_sha256"]) == 42
    assert run["corpus"]["processed_digest"] == er.corpus_digest(run["corpus"]["processed_sha256"])
    assert run["reuse_processed"] is True
    assert run["retrieval_only"] is False
    assert set(run["git"]) == {"commit", "dirty"}
    assert run["generated_at"].endswith("+00:00")
    assert (tmp_path / "summary.md").is_file()

    monkeypatch.delenv("LITELLM_API_KEY")
    assert er.main(["--combos", "g", "--out", str(tmp_path)]) == 0
    second = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert second["llm"]["proxy_configured"] is False
    assert list(second["combos"]) == ["g"]
    assert [list(r["combos"]) for r in second["previous_runs"]] == [["f"]]
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith("# RAG 2 組實驗比較（F、G）")


def test_main_returns_nonzero_when_a_combo_fails(tmp_path, monkeypatch, capsys):
    def failing(combo, out_dir, **kwargs):
        raise RuntimeError(f"chroma exploded under {er.ROOT / 'chroma_db'}")

    monkeypatch.setattr(er, "run_combo", failing)
    assert er.main(["--combos", "e", "--out", str(tmp_path)]) == 1
    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run["combos"]["e"]["status"] == "failed: RuntimeError"
    err = capsys.readouterr().err
    assert "RuntimeError" in err
    assert "chroma_db" in err
    assert str(er.ROOT) not in err  # 不洩漏本機絕對路徑


def test_main_reports_os_errors_without_absolute_paths(tmp_path, monkeypatch, capsys):
    def failing(combo, out_dir, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", str(tmp_path / "deep" / "missing.txt"))

    monkeypatch.setattr(er, "run_combo", failing)
    assert er.main(["--combos", "e", "--out", str(tmp_path / "out")]) == 1
    err = capsys.readouterr().err
    assert "deep/missing.txt" in err
    assert str(tmp_path) not in err


def test_main_returns_nonzero_for_a_degraded_combo(tmp_path, monkeypatch, capsys):
    def degraded(combo, out_dir, **kwargs):
        write_results(out_dir, fake_results(combo["id"], [0.2], "answer"))
        return {**ok_info(), "status": "degraded", "failed_questions": [3]}

    monkeypatch.setattr(er, "run_combo", degraded)
    assert er.main(["--combos", "g", "--out", str(tmp_path)]) == 1
    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run["combos"]["g"]["status"] == "degraded"
    assert run["combos"]["g"]["failed_questions"] == [3]
    assert "degraded" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], {"skip_rebuild": False, "retrieval_only": False, "reuse_processed": True}),
        (["--skip-rebuild", "--retrieval-only"], {"skip_rebuild": True, "retrieval_only": True, "reuse_processed": True}),
        (["--re-extract"], {"skip_rebuild": False, "retrieval_only": False, "reuse_processed": False}),
    ],
)
def test_cli_flags_reach_run_combo_and_run_json(tmp_path, monkeypatch, argv, expected):
    seen: list[dict] = []

    def recording_run_combo(combo, out_dir, **kwargs):
        seen.append(kwargs)
        write_results(out_dir, fake_results(combo["id"], [0.2], "answer"))
        return ok_info()

    monkeypatch.setattr(er, "run_combo", recording_run_combo)
    assert er.main(["--combos", "e", "--out", str(tmp_path), *argv]) == 0

    assert {k: seen[0][k] for k in expected} == expected
    assert seen[0]["environment"]["text_source"] == (
        "data/processed" if expected["reuse_processed"] else "data/raw (re-extracted in memory)"
    )
    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run["reuse_processed"] is expected["reuse_processed"]
    assert run["retrieval_only"] is expected["retrieval_only"]
    assert run["corpus"]["matches_pin"] is (True if expected["reuse_processed"] else None)


def test_skip_rebuild_and_re_extract_cannot_be_combined(capsys):
    with pytest.raises(SystemExit) as exc:
        er.parse_args(["--re-extract", "--skip-rebuild"])
    assert exc.value.code == 2


def test_default_out_dir_is_where_rag_eval_reads_round2():
    assert er.DEFAULT_OUT_DIR == rag_eval.ROUND2_DIR


def test_main_refuses_a_corpus_that_differs_from_the_pin(tmp_path, monkeypatch, capsys):
    changed = {**er.processed_corpus_hashes(), "Popping.txt": "0" * 64}
    monkeypatch.setattr(er, "processed_corpus_hashes", lambda processed_dir=None: dict(changed))
    monkeypatch.setattr(er, "run_combo", lambda *a, **k: pytest.fail("no combo may run"))

    assert er.main(["--combos", "e", "--out", str(tmp_path)]) == 2
    assert "Popping.txt" in capsys.readouterr().err
    assert not (tmp_path / "run.json").exists()


def test_re_extract_does_not_require_the_pinned_corpus(tmp_path, monkeypatch):
    changed = {**er.processed_corpus_hashes(), "Popping.txt": "0" * 64}
    monkeypatch.setattr(er, "processed_corpus_hashes", lambda processed_dir=None: dict(changed))
    monkeypatch.setattr(
        er, "run_combo", lambda combo, out_dir, **k: (write_results(out_dir, fake_results("e", [0.2], "a")), ok_info())[1]
    )
    assert er.main(["--combos", "e", "--out", str(tmp_path), "--re-extract"]) == 0


def test_dotenv_is_loaded_only_from_the_repo():
    """不帶路徑的 load_dotenv() 會往上層目錄找 .env；只允許 rag_common 以 ROOT / ".env" 載入。"""
    names = ("rag_common.py", "rag_query.py", "experiment_runner.py")
    sources = {name: (er.ROOT / name).read_text(encoding="utf-8") for name in names}
    bare_call = re.compile(r"^\s*load_dotenv\(\s*\)", re.MULTILINE)
    assert [name for name, src in sources.items() if bare_call.search(src)] == []
    assert re.search(r'^load_dotenv\(ROOT / "\.env"\)', sources["rag_common.py"], re.MULTILINE)


# --- corpus selection (--corpus round1 | current) ----------------------------

STRUCTURED_NOTES = {"popping_glossary.md", "popping_history_timeline.md", "popping_style_taxonomy.md"}


def test_collection_names_include_the_corpus():
    assert er.CORPUS_CHOICES == ("round1", "current")
    assert er.DEFAULT_CORPUS == "current"
    assert er.collection_for("round1", "e") == "cypher_og_round1_combo_e"
    combo = er.combo_for_corpus(er.COMBO_MAP["f"], "round1")
    assert combo["collection"] == "cypher_og_round1_combo_f"
    assert {k: v for k, v in combo.items() if k != "collection"} == {
        k: v for k, v in er.COMBO_MAP["f"].items() if k != "collection"
    }
    assert er.COMBO_MAP["f"]["collection"] == "cypher_og_current_combo_f"  # 共用設定不被修改
    with pytest.raises(ValueError, match="other"):
        er.combo_for_corpus(er.COMBO_MAP["f"], "other")


def test_corpus_selections_list_raw_document_names():
    round1 = er.load_corpus_selection("round1")
    current = er.load_corpus_selection("current")
    assert round1["name"] == "round1"
    assert round1["documents"] == ROUND1_DOCS
    assert current["name"] == "current"
    assert current["documents"] == [p.name for p in er.list_raw_documents()]
    assert set(current["documents"]) - set(round1["documents"]) == STRUCTURED_NOTES
    with pytest.raises(ValueError):
        er.load_corpus_selection("everything")


def test_round1_selection_refuses_documents_missing_from_data_raw(tmp_path, monkeypatch):
    path = tmp_path / "corpus_round1.json"
    path.write_text(json.dumps({"name": "round1", "documents": ["Popping.md", "gone.md"]}), encoding="utf-8")
    monkeypatch.setattr(er, "CORPUS_ROUND1_PATH", path)
    with pytest.raises(ValueError, match="gone.md"):
        er.load_corpus_selection("round1")


def test_round1_selection_refuses_a_definition_with_another_name(tmp_path, monkeypatch):
    path = tmp_path / "corpus_round1.json"
    path.write_text(json.dumps({"name": "round2", "documents": ["Popping.md"]}), encoding="utf-8")
    monkeypatch.setattr(er, "CORPUS_ROUND1_PATH", path)
    with pytest.raises(ValueError, match="round2"):
        er.load_corpus_selection("round1")


def test_corpus_environment_records_name_count_and_list_digest():
    env = er.corpus_environment(er.load_corpus_selection("round1"))
    assert env == {"corpus": "round1", "document_count": 39, "documents_digest": document_list_digest(ROUND1_DOCS)}
    built = er.build_environment(git={"commit": "c", "dirty": False}, reuse_processed=True, corpus=er.load_corpus_selection("round1"))
    assert built["corpus"] == "round1" and built["document_count"] == 39


def test_corpus_flag_defaults_to_current_and_rejects_unknown_names():
    assert er.parse_args([]).corpus == "current"
    assert er.parse_args(["--corpus", "round1"]).corpus == "round1"
    with pytest.raises(SystemExit) as exc:
        er.parse_args(["--corpus", "all"])
    assert exc.value.code == 2


def recording_run_combo(seen: list):
    def run(combo, out_dir, **kwargs):
        seen.append((combo, kwargs))
        write_results(out_dir, fake_results(combo["id"], [0.2], "answer"))
        return ok_info()

    return run


def test_main_with_the_round1_corpus_uses_its_own_collections_and_records_it(tmp_path, monkeypatch):
    seen: list = []
    monkeypatch.setattr(er, "run_combo", recording_run_combo(seen))
    monkeypatch.setattr(er, "_git_state", lambda: {"commit": "abc1234", "dirty": False})

    assert er.main(["--combos", "e", "f", "--corpus", "round1", "--retrieval-only", "--out", str(tmp_path)]) == 0

    assert [combo["collection"] for combo, _ in seen] == ["cypher_og_round1_combo_e", "cypher_og_round1_combo_f"]
    kwargs = seen[0][1]
    assert kwargs["corpus"]["name"] == "round1"
    assert kwargs["corpus"]["documents"] == ROUND1_DOCS
    assert {k: kwargs["environment"][k] for k in ("corpus", "document_count", "documents_digest")} == {
        "corpus": "round1",
        "document_count": 39,
        "documents_digest": document_list_digest(ROUND1_DOCS),
    }
    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    corpus = run["corpus"]
    assert corpus["name"] == "round1"
    assert corpus["definition"] == "experiments/corpus_round1.json"
    assert corpus["document_count"] == 39
    assert corpus["documents"] == ROUND1_DOCS
    assert corpus["documents_digest"] == document_list_digest(ROUND1_DOCS)
    # 語料指紋只涵蓋所選的 39 份文件
    assert len(corpus["processed_sha256"]) == 39
    assert corpus["processed_digest"] == pinned_digest_for(ROUND1_DOCS, load_corpus_pin())
    assert corpus["matches_pin"] is True
    assert run["combos"]["e"]["collection"] == "cypher_og_round1_combo_e"


def test_pin_check_covers_exactly_the_selected_documents(tmp_path, monkeypatch, capsys):
    """round1 不含 popping_glossary：該檔變動不影響 round1，但 current 必須拒絕。"""
    changed = {**er.processed_corpus_hashes(), "popping_glossary.txt": "0" * 64}
    monkeypatch.setattr(er, "processed_corpus_hashes", lambda processed_dir=None: dict(changed))
    monkeypatch.setattr(er, "run_combo", recording_run_combo([]))

    assert er.main(["--combos", "e", "--corpus", "round1", "--out", str(tmp_path / "r1")]) == 0
    assert er.main(["--combos", "e", "--corpus", "current", "--out", str(tmp_path / "cur")]) == 2
    assert "popping_glossary.txt" in capsys.readouterr().err
    assert not (tmp_path / "cur" / "run.json").exists()


def test_pin_check_flags_a_selected_document_without_processed_text(tmp_path, monkeypatch, capsys):
    missing = {k: v for k, v in er.processed_corpus_hashes().items() if k != "Popping.txt"}
    monkeypatch.setattr(er, "processed_corpus_hashes", lambda processed_dir=None: dict(missing))
    monkeypatch.setattr(er, "run_combo", lambda *a, **k: pytest.fail("no combo may run"))
    assert er.main(["--combos", "e", "--corpus", "round1", "--out", str(tmp_path)]) == 2
    assert "Popping.txt" in capsys.readouterr().err


def test_check_pinned_corpus_flags_unpinned_selected_documents():
    hashes = {"Popping.txt": load_corpus_pin()["Popping.txt"], "new.txt": "1" * 64}
    assert er.check_pinned_corpus(hashes, ["Popping.md", "new.md"]) == ["new.txt"]
    assert er.check_pinned_corpus(hashes, ["Popping.md"]) == []


def test_an_out_folder_named_for_another_corpus_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(er, "run_combo", lambda *a, **k: pytest.fail("no combo may run"))
    assert er.main(["--combos", "e", "--corpus", "current", "--out", str(tmp_path / "round1_corpus" / "run1")]) == 2
    assert "round1_corpus" in capsys.readouterr().err
    assert er.main(["--combos", "e", "--corpus", "round1", "--out", str(tmp_path / "current_corpus")]) == 2

    monkeypatch.setattr(er, "run_combo", recording_run_combo([]))
    assert er.main(["--combos", "e", "--corpus", "round1", "--out", str(tmp_path / "round1_corpus" / "run1")]) == 0


def test_run_combo_indexes_only_the_selected_documents(tmp_path, monkeypatch, no_llm, fake_index):
    monkeypatch.setattr(er, "retrieve_context", lambda q, **k: ("", []))
    corpus = er.load_corpus_selection("round1")
    env = er.build_environment(git={"commit": "abc", "dirty": False}, reuse_processed=True, corpus=corpus)

    er.run_combo(
        er.combo_for_corpus(er.COMBO_MAP["e"], "round1"), tmp_path, model="m", retrieval_only=True,
        environment=env, corpus=corpus,
    )

    assert fake_index[0]["include"] == ROUND1_DOCS
    assert fake_index[0]["collection_name"] == "cypher_og_round1_combo_e"
    payload = json.loads((tmp_path / "combo_e" / "results.json").read_text(encoding="utf-8"))
    assert payload["combo"]["collection"] == "cypher_og_round1_combo_e"
    assert payload["environment"]["corpus"] == "round1"
    assert payload["environment"]["document_count"] == 39


def test_run_combo_refuses_a_collection_of_another_corpus(tmp_path, monkeypatch, no_llm, fake_index):
    corpus = er.load_corpus_selection("round1")
    with pytest.raises(ValueError, match="cypher_og_round1_combo_e"):
        er.run_combo(er.COMBO_MAP["e"], tmp_path, model="m", environment=FAKE_ENV, corpus=corpus)
    assert fake_index == []


def test_skip_rebuild_fingerprints_only_the_selected_documents(tmp_path, monkeypatch, no_llm, fake_index):
    monkeypatch.setattr(er, "retrieve_context", lambda q, **k: ("", []))
    corpus = er.load_corpus_selection("round1")
    info = er.run_combo(
        er.combo_for_corpus(er.COMBO_MAP["e"], "round1"), tmp_path, model="m", skip_rebuild=True,
        retrieval_only=True, environment=FAKE_ENV, corpus=corpus,
    )
    assert info["corpus_sha256"] == pinned_digest_for(ROUND1_DOCS, load_corpus_pin())
