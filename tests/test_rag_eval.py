"""rag_eval 的測試：讀取真實的 round-1 輸出（experiments/combo_a..e）與 gold.json。

不需要 API key、網路、sentence-transformers 模型或真實 ChromaDB；
round-2 以 tmp_path 內的假 results.json 驗證。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import rag_eval  # noqa: E402
from data_update import list_raw_documents  # noqa: E402
from rag_common import (  # noqa: E402
    document_list_digest,
    load_corpus_definition,
    load_corpus_pin,
    pinned_digest_for,
)

EXPERIMENTS = ROOT / "experiments"
ROUND1_IDS = ("a", "b", "c", "d", "e")


@pytest.fixture(scope="module")
def gold() -> dict:
    return rag_eval.load_gold(EXPERIMENTS / "gold.json")


def _round1(combo_id: str, q: int) -> dict:
    return rag_eval.parse_round1_question(EXPERIMENTS / f"combo_{combo_id}" / f"q{q}.md")


# ─── gold.json（預先登錄內容）────────────────────────────────────────────────


# 預先登錄的完整別名；gold.json 在 round 2 之後若被改動（包括別名），這個測試會失敗
PREREGISTERED_MEMBERS = {
    "Boogaloo Sam": ["Boogaloo Sam"],
    "Popin' Pete": ["Popin' Pete", "Popin Pete", "Pop'in Pete", "Poppin Pete", "Poppin' Pete", "Pete"],
    "Skeeter Rabbit": ["Skeeter Rabbit"],
    "Suga Pop": ["Suga Pop", "Sugar Pop"],
    "Mr. Wiggles": ["Mr. Wiggles", "Mr Wiggles"],
    "Jazzy J": ["Jazzy J"],
    "Creepin Cid": ["Creepin Cid", "Creepin' Cid", "Creep'n Cid", "Creep'n Sid", "Creepin Sid", "Creeping Sidney"],
    "Puppet Boozer": ["Puppet Boozer"],
    "Robot Dane": ["Robot Dane"],
}
PREREGISTERED_FACT_GROUPS = {
    "format": ["2v2", "2 v 2", "2vs2", "2 vs 2", "two-person", "duo", "duos", "雙人", "兩人"],
    "city": ["Paris", "巴黎"],
    "year": ["2002", "1998"],
    "venue": ["Accor Arena", "Bercy", "Stade Pierre de Coubertin"],
    "organizer": ["Bruce Ykanji"],
}


def test_gold_is_the_preregistered_definition(gold: dict) -> None:
    q2 = gold["q2_members"]
    q4 = gold["q4_juste_debout"]
    assert q2["question_number"] == 2
    assert q4["question_number"] == 4
    assert list(q2["members"].items()) == list(PREREGISTERED_MEMBERS.items())
    assert q4["entity"] == ["Juste Debout"]
    assert list(q4["fact_groups"].items()) == list(PREREGISTERED_FACT_GROUPS.items())


def test_load_gold_rejects_incomplete_file(tmp_path: Path) -> None:
    bad = tmp_path / "gold.json"
    bad.write_text(json.dumps({"q2_members": {"members": {}}}), encoding="utf-8")
    with pytest.raises(ValueError):
        rag_eval.load_gold(bad)


# ─── normalize 與別名比對邊界 ────────────────────────────────────────────────


def test_normalize_straightens_quotes_and_collapses_whitespace() -> None:
    assert rag_eval.normalize("Pop\u2019in \n\t  Pete ") == "Pop'in Pete"
    assert rag_eval.normalize("\u2018Boogaloo\u2019") == "'Boogaloo'"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("成員Pete是第一代", ["Popin' Pete"]),  # CJK 緊鄰仍算命中
        ("Pop\u2019in Pete", ["Popin' Pete"]),  # 彎引號
        ("boogaloo   SAM", ["Boogaloo Sam"]),  # 不分大小寫、空白折疊
        ("Boogaloo\nSam", ["Boogaloo Sam"]),
        ("**Pete**：創始成員", ["Popin' Pete"]),
        ("Mr Wiggles and Sugar Pop", ["Mr. Wiggles", "Suga Pop"]),
        ("They compete every year", []),  # Pete 在單字中間
        ("Peterson", []),  # Pete 後接英文字母
        ("Jazzy Jay", []),
        ("MrXWiggles", []),  # 別名須 re.escape
        ("Creep\u2019n Sid and Robot Dane", ["Creepin Cid", "Robot Dane"]),
    ],
)
def test_members_named_boundaries(gold: dict, text: str, expected: list[str]) -> None:
    assert rag_eval.members_named(text, gold) == expected


def test_members_named_returns_sorted_canonical_names(gold: dict) -> None:
    text = "Robot Dane, Boogaloo Sam, Popin Pete, Poppin' Pete"
    assert rag_eval.members_named(text, gold) == ["Boogaloo Sam", "Popin' Pete", "Robot Dane"]


# ─── Q4 fact groups ─────────────────────────────────────────────────────────


def test_fact_groups_require_the_entity(gold: dict) -> None:
    assert rag_eval.juste_debout_fact_groups("巴黎 2002 年起的 2v2 比賽，Accor Arena", gold) == []
    assert rag_eval.juste_debout_fact_groups("Juste-Debout in Paris", gold) == []


def test_fact_groups_all_five(gold: dict) -> None:
    text = (
        "juste debout 自 2002 年起每年在巴黎舉辦，採雙人 2v2 賽制，"
        "決賽在 Accor Arena，由 Bruce Ykanji 創辦。"
    )
    assert rag_eval.juste_debout_fact_groups(text, gold) == [
        "format",
        "city",
        "year",
        "venue",
        "organizer",
    ]


def test_fact_groups_are_boundary_aware(gold: dict) -> None:
    assert rag_eval.juste_debout_fact_groups("Juste Debout: Parisian duology", gold) == []
    assert rag_eval.juste_debout_fact_groups("Juste Debout duos", gold) == ["format"]


def test_gold_chunk_count(gold: dict) -> None:
    chunks = [
        "Juste Debout returns to Bercy",  # entity + venue → 計入
        "Paris 2002 2v2",  # 沒有 entity
        "Juste Debout only",  # 沒有 fact group
        "",
    ]
    assert rag_eval.gold_chunk_count(chunks, gold) == 1


# ─── round-1 解析（真實檔案）────────────────────────────────────────────────


def test_parse_round1_question_e_q4() -> None:
    parsed = _round1("e", 4)
    assert parsed["question"] == "Juste Debout 比賽的規則和歷史？"
    assert [d for _, _, d in parsed["cites"]] == [0.422, 0.472, 0.482, 0.494, 0.498]
    assert parsed["cites"][0] == ("Salah (dancer) - Wikipedia.txt", 66, 0.422)
    assert parsed["section_labels"][0] == "Career"
    assert parsed["section_labels"][2] == ""
    assert parsed["answer"].startswith("Step 1")
    assert parsed["answer"].endswith("資料未涵蓋。")
    assert "generated by" not in parsed["answer"]


def test_parse_round1_question_handles_bracketed_labels() -> None:
    parsed = _round1("d", 1)
    assert parsed["cites"][2] == ("Popping.txt", 1, 0.311)
    assert parsed["section_labels"][2].startswith("![](https://")
    assert len(parsed["cites"]) == 5


def _fake_round1_md(n_recorded: int, cite_lines: list[str], answer: str) -> str:
    return "\n".join(
        [
            "        # Q2: 測試？",
            "",
            f"        ## 檢索結果（{n_recorded} 個 chunks，平均距離 0.300）",
            "",
            *cite_lines,
            "",
            "        ## LLM 回答",
            "",
            answer,
            "",
            "        ---",
            "        *generated by experiment_runner.py at 2026-04-05 08:29 UTC*",
            "",
        ]
    )


def test_parse_round1_answer_keeps_inner_rules(tmp_path: Path) -> None:
    path = tmp_path / "q2.md"
    cite = "   [1] `x.txt` chunk 3 (距離: 0.300)"
    path.write_text(_fake_round1_md(1, [cite], "        第一段\n---\n第二段"), encoding="utf-8")
    parsed = rag_eval.parse_round1_question(path)
    assert parsed["question"] == "測試？"
    assert parsed["cites"] == [("x.txt", 3, 0.3)]
    assert parsed["answer"] == "第一段\n---\n第二段"


def test_parse_round1_rejects_cite_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "q2.md"
    cite = "[1] `x.txt` chunk 3 (距離: 0.300)"
    path.write_text(_fake_round1_md(2, [cite], "回答"), encoding="utf-8")
    with pytest.raises(ValueError):
        rag_eval.parse_round1_question(path)


@pytest.mark.parametrize(("combo_id", "expected"), [("a", 9), ("b", 9), ("c", 3), ("d", 2), ("e", 9)])
def test_round1_q2_members_named(gold: dict, combo_id: str, expected: int) -> None:
    named = rag_eval.members_named(_round1(combo_id, 2)["answer"], gold)
    assert len(named) == expected


def test_round1_d_names_two_members(gold: dict) -> None:
    # CV：D "listed two group members versus the baseline's nine"
    assert rag_eval.members_named(_round1("d", 2)["answer"], gold) == ["Boogaloo Sam", "Popin' Pete"]


@pytest.mark.parametrize("combo_id", ROUND1_IDS)
def test_round1_q4_answers_state_no_fact_groups(gold: dict, combo_id: str) -> None:
    assert rag_eval.juste_debout_fact_groups(_round1(combo_id, 4)["answer"], gold) == []


def test_round1_mean_distances_match_cv() -> None:
    def mean_for(combo_id: str) -> float:
        lists = [[d for _, _, d in _round1(combo_id, q)["cites"]] for q in range(1, 6)]
        value, n_questions = rag_eval.mean_distance(lists)
        assert n_questions == 5
        return value

    a, d = mean_for("a"), mean_for("d")
    assert a == pytest.approx(0.399, abs=5e-4)
    assert d == pytest.approx(0.359, abs=5e-4)
    assert (a - d) / a == pytest.approx(0.10, abs=5e-3)


def test_mean_distance_skips_questions_without_cites() -> None:
    assert rag_eval.mean_distance([[0.2, 0.4], [], [0.5]]) == (pytest.approx(0.4), 2)
    assert rag_eval.mean_distance([[], []]) == (None, 0)


@pytest.mark.parametrize("combo_id", ROUND1_IDS)
def test_round1_chunk_reconstruction(combo_id: str) -> None:
    for q in range(1, 6):
        parsed = _round1(combo_id, q)
        chunks = rag_eval.reconstruct_round1_chunks(parsed["cites"], combo_id)
        assert [c["source_file"] for c in chunks] == [src for src, _, _ in parsed["cites"]]
        assert [c["chunk_index"] for c in chunks] == [idx for _, idx, _ in parsed["cites"]]
        assert all(c["text"].strip() for c in chunks)
        # 重建出的 chunk 必須重現當時記錄的 section 標籤（同一套切法才會一致）
        assert [c["section_title"] for c in chunks] == parsed["section_labels"]


def test_round1_chunkers_match_recorded_configs() -> None:
    for combo_id in ROUND1_IDS:
        config = (EXPERIMENTS / f"combo_{combo_id}" / "config.md").read_text(encoding="utf-8")
        spec = rag_eval.ROUND1_COMBOS[combo_id]
        assert f"| Chunking | {spec['chunk_desc']} |" in config
        assert f"| top-k | {spec['top_k']} |" in config
        assert f"| max_distance | {spec['max_distance']} |" in config


def test_reconstruct_rejects_path_traversal() -> None:
    with pytest.raises(ValueError):
        rag_eval.reconstruct_round1_chunks([("../../README.md", 0, 0.1)], "b")


# ─── 語料鎖定（data/processed 必須與 corpus_pin.json 一致）──────────────────


def test_committed_processed_text_matches_the_pin() -> None:
    pin = json.loads((EXPERIMENTS / "corpus_pin.json").read_text(encoding="utf-8"))
    digest = rag_eval.check_round1_corpus(rag_eval.PROCESSED_DIR, EXPERIMENTS / "corpus_pin.json")
    assert digest == pin["digest"]
    assert len(pin["files"]) == 42


@pytest.fixture
def processed_copy(tmp_path: Path) -> Path:
    copy = tmp_path / "processed"
    shutil.copytree(rag_eval.PROCESSED_DIR, copy)
    return copy


def test_changed_processed_text_is_refused(processed_copy: Path) -> None:
    target = processed_copy / "a-historical-overview-of-hip-hop-dances-johnson-imani-kai.txt"
    target.write_text(target.read_text(encoding="utf-8") + " re-extracted", encoding="utf-8")
    with pytest.raises(ValueError, match="a-historical-overview"):
        rag_eval.check_round1_corpus(processed_copy, EXPERIMENTS / "corpus_pin.json")


def test_missing_or_extra_processed_files_are_refused(processed_copy: Path) -> None:
    (processed_copy / "Popping.txt").unlink()
    (processed_copy / "extra.txt").write_text("new document", encoding="utf-8")
    with pytest.raises(ValueError, match="2 個檔案"):
        rag_eval.check_round1_corpus(processed_copy, EXPERIMENTS / "corpus_pin.json")


def test_cli_refuses_round1_when_the_corpus_changed(
    processed_copy: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = processed_copy / "Popping.txt"
    target.write_text(target.read_text(encoding="utf-8").replace("Popping", "Poping", 1), encoding="utf-8")
    monkeypatch.setattr(rag_eval, "PROCESSED_DIR", processed_copy)
    out = tmp_path / "out"
    assert rag_eval.main(["--round1", str(EXPERIMENTS), "--round2", str(tmp_path / "none"), "--out", str(out)]) == 1
    assert not (out / "metrics.json").exists()
    err = capsys.readouterr().err
    assert "Popping.txt" in err
    assert str(tmp_path) not in err  # 不洩漏本機絕對路徑


def test_round1_errors_do_not_leak_absolute_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    round1 = tmp_path / "r1"
    shutil.copytree(EXPERIMENTS / "combo_a", round1 / "combo_a")
    (round1 / "combo_a" / "q3.md").unlink()

    runs = rag_eval.load_round1_runs(round1)

    assert [r["combo_id"] for r in runs] == []
    err = capsys.readouterr().err
    assert "FileNotFoundError" in err
    assert "q3.md" in err
    assert str(tmp_path) not in err
    assert str(Path.home()) not in err


# ─── round 2（假 results.json）──────────────────────────────────────────────


def _round1_questions() -> list[str]:
    return [_round1("e", n)["question"] for n in range(1, 6)]


def _cite(text: str, dist_q: float, idx: int = 0) -> dict:
    return {
        "chunk_id": f"doc.txt::{idx}",
        "source_file": "doc.txt",
        "chunk_index": idx,
        "section_title": "",
        "distance": 0.05,  # 刻意不同：指標必須用 distance_to_question
        "distance_to_question": dist_q,
        "retrieved_by": "both",
        "text": text,
    }


PINNED_DIGEST = json.loads((EXPERIMENTS / "corpus_pin.json").read_text(encoding="utf-8"))["digest"]
ROUND1_DOCS = load_corpus_definition(EXPERIMENTS / "corpus_round1.json")["documents"]
# round1 語料（39 份）在 corpus_pin.json 下的文字指紋
ROUND1_PINNED_DIGEST = pinned_digest_for(ROUND1_DOCS, load_corpus_pin(EXPERIMENTS / "corpus_pin.json"))


def _environment(**overrides) -> dict:
    return {
        "git_commit": "abc1234def",
        "git_dirty": False,
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "text_source": "data/processed",
        "corpus": "round1",
        "document_count": 39,
        "documents_digest": document_list_digest(ROUND1_DOCS),
        "corpus_sha256": ROUND1_PINNED_DIGEST,
        "chunk_count": 1248,
        "index_rebuilt": True,
        **overrides,
    }


def _current_environment(**overrides) -> dict:
    current = [p.name for p in list_raw_documents()]
    return _environment(
        corpus="current",
        document_count=len(current),
        documents_digest=document_list_digest(current),
        corpus_sha256=PINNED_DIGEST,
        chunk_count=1279,
        **overrides,
    )


def _write_results(
    round2_dir: Path,
    combo_id: str,
    *,
    retrieval_only: bool = False,
    use_hybrid: bool = False,
    use_expand: bool = False,
    texts: list[str] | None = None,
    model: str = "gpt-test",
    environment: dict | None = None,
    errors: dict[int, str] | None = None,
) -> Path:
    texts = texts or _round1_questions()
    questions = [
        {"index": 1, "question": texts[0], "expanded_queries": None,
         "cites": [_cite("a", 0.2), _cite("b", 0.4, 1)], "answer": "x", "error": None},
        {"index": 2, "question": texts[1], "expanded_queries": None,
         "cites": [_cite("Skeeter Rabbit and Suga Pop", 0.3), _cite("Boogaloo\nSam", 0.3, 1)],
         "answer": "成員有 Boogaloo Sam 與 Popin\u2019 Pete。", "error": None},
        {"index": 3, "question": texts[2], "expanded_queries": None,
         "cites": [_cite("c", 0.5)], "answer": "x", "error": None},
        {"index": 4, "question": texts[3], "expanded_queries": ["Juste Debout history"],
         "cites": [
             _cite("Juste Debout 2002 Paris", 0.4),
             _cite("Juste Debout only", 0.6, 1),
             _cite("Paris 2v2", 0.5, 2),
         ],
         "answer": "Juste Debout 是巴黎的 2v2 比賽，始於 2002 年。", "error": None},
        {"index": 5, "question": texts[4], "expanded_queries": None,
         "cites": [_cite("d", 0.2)], "answer": "x", "error": None},
    ]
    for index, error in (errors or {}).items():
        q = questions[index - 1]
        q["error"] = error
        if error.startswith("expansion_failed"):
            q.update(cites=[], expanded_queries=None, answer="【查詢擴展失敗】")
        elif error.startswith("llm_error"):
            q["answer"] = "【LLM 呼叫失敗：RuntimeError】"
        elif error == "no_context":
            q.update(cites=[], answer="【知識庫中未找到相關資料，無法生成回答。】")
    payload = {
        "combo": {
            "id": combo_id,
            "name": f"Combo {combo_id.upper()}",
            "chunk_desc": "recursive / size=600 / overlap=100",
            "top_k": 5,
            "max_distance": 0.65,
            "truncate_to": None,
            "prompt_style": "cot",
            "use_hybrid": use_hybrid,
            "use_expand": use_expand,
            "collection": f"cypher_og_combo_{combo_id}",
        },
        "model": model,
        "retrieval_only": retrieval_only,
        "environment": environment if environment is not None else _environment(),
        "generated_at": "2026-11-09T00:00:00+00:00",
        "questions": questions,
    }
    path = round2_dir / f"combo_{combo_id}" / "results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_round2_fake_results_are_scored(tmp_path: Path, gold: dict) -> None:
    round2 = tmp_path / "round2"
    _write_results(round2, "f", use_hybrid=True)
    _write_results(round2, "e", retrieval_only=True)

    runs = rag_eval.load_round2_runs(round2)
    rows = rag_eval.compute_metrics(runs, gold)
    by_id = {r["combo_id"]: r for r in rows}

    assert [r["run"] for r in rows] == ["E\u2032 (control)", "F"]
    f = by_id["f"]
    assert f["round"] == 2
    assert f["model"] == "gpt-test"
    assert f["mean_distance_to_question"] == pytest.approx(0.36)
    assert f["questions_with_cites"] == 5
    assert f["q2_members_named"] == 2
    assert f["q2_members_named_list"] == ["Boogaloo Sam", "Popin' Pete"]
    assert f["q2_members_in_context"] == 3
    assert f["q4_gold_chunks"] == 1
    assert f["q4_chunks_retrieved"] == 3
    assert f["q4_fact_groups"] == 3
    assert f["q4_fact_groups_list"] == ["format", "city", "year"]
    assert "BM25" in f["retrieval"]

    assert f["cites_over_max_distance"] == 0
    assert f["cites_scored"] == 9
    assert f["failed_questions"] == []

    e = by_id["e"]
    assert e["q2_members_named"] is None  # retrieval-only：沒有回答可評
    assert e["q4_fact_groups"] is None
    assert e["q2_members_in_context"] == 3


def test_llm_errors_make_only_the_answer_metrics_na(tmp_path: Path, gold: dict) -> None:
    path = _write_results(
        tmp_path / "round2", "f", errors={2: "llm_error: RuntimeError", 4: "llm_error: RuntimeError"}
    )
    (row,) = rag_eval.compute_metrics([rag_eval.load_round2_run(path)], gold)

    assert row["q2_members_named"] is None
    assert row["q4_fact_groups"] is None
    assert row["q2_members_in_context"] == 3  # 檢索照常執行，context 指標仍計算
    assert row["q4_gold_chunks"] == 1
    assert row["questions_with_cites"] == 5
    assert row["failed_questions"] == [
        {"index": 2, "error": "llm_error: RuntimeError"},
        {"index": 4, "error": "llm_error: RuntimeError"},
    ]


def test_no_context_answers_are_scored_as_zero(tmp_path: Path, gold: dict) -> None:
    path = _write_results(tmp_path / "round2", "f", errors={2: "no_context", 4: "no_context"})
    (row,) = rag_eval.compute_metrics([rag_eval.load_round2_run(path)], gold)

    assert row["q2_members_named"] == 0
    assert row["q4_fact_groups"] == 0
    assert row["q2_members_in_context"] == 0
    assert row["q4_gold_chunks"] == 0
    assert row["failed_questions"] == []


def test_failed_expansion_makes_every_metric_of_that_question_na(tmp_path: Path, gold: dict) -> None:
    path = _write_results(
        tmp_path / "round2",
        "g",
        use_expand=True,
        errors={2: "expansion_failed: RuntimeError", 4: "expansion_failed: no_subqueries"},
    )
    (row,) = rag_eval.compute_metrics([rag_eval.load_round2_run(path)], gold)

    assert row["q2_members_named"] is None
    assert row["q2_members_in_context"] is None
    assert row["q4_gold_chunks"] is None
    assert row["q4_chunks_retrieved"] is None
    assert row["q4_fact_groups"] is None
    assert row["questions_with_cites"] == 3
    assert row["mean_distance_to_question"] == pytest.approx((0.3 + 0.5 + 0.2) / 3)
    assert [f["index"] for f in row["failed_questions"]] == [2, 4]


def test_expansion_failure_with_leftover_cites_is_still_na(tmp_path: Path, gold: dict) -> None:
    path = _write_results(tmp_path / "round2", "g", use_expand=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["questions"][1]["error"] = "expansion_failed: RuntimeError"  # cites 仍保留
    path.write_text(json.dumps(data), encoding="utf-8")
    (row,) = rag_eval.compute_metrics([rag_eval.load_round2_run(path)], gold)
    assert row["q2_members_in_context"] is None
    assert row["questions_with_cites"] == 4


def test_retrieval_labels_state_where_the_cutoff_applies() -> None:
    base = {"chunk_desc": "recursive / size=600 / overlap=100", "top_k": 5, "max_distance": 0.65}
    assert rag_eval.describe_retrieval(base).endswith("dense top-5; max distance 0.65")
    assert rag_eval.describe_retrieval({**base, "use_hybrid": True}).endswith(
        "dense + BM25 (RRF) top-5; max distance 0.65 (dense channel only)"
    )
    assert rag_eval.describe_retrieval({**base, "use_expand": True}).endswith(
        "dense + query expansion top-5; max distance 0.65 (per sub-query)"
    )


def test_cites_past_the_cutoff_are_counted(tmp_path: Path, gold: dict) -> None:
    path = _write_results(tmp_path / "round2", "f", use_hybrid=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["questions"][0]["cites"][1]["distance_to_question"] = 0.88  # BM25-only chunk past 0.65
    path.write_text(json.dumps(data), encoding="utf-8")
    (row,) = rag_eval.compute_metrics([rag_eval.load_round2_run(path)], gold)
    assert row["cites_over_max_distance"] == 1


def test_round2_requires_the_environment_record(tmp_path: Path) -> None:
    path = _write_results(tmp_path / "round2", "f")
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["environment"]["corpus_sha256"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="corpus_sha256"):
        rag_eval.load_round2_run(path)


@pytest.mark.parametrize(
    ("field", "f_kwargs"),
    [
        ("model", {"model": "gemini/gemini-2.5-flash"}),
        ("retrieval_only", {"retrieval_only": True}),
        ("chunk_count", {"environment": _environment(chunk_count=999)}),
        ("corpus_sha256", {"environment": _environment(corpus_sha256="0" * 64)}),
        ("git_commit", {"environment": _environment(git_commit="fffffff")}),
        ("embedding_model", {"environment": _environment(embedding_model="other-model")}),
        ("text_source", {"environment": _environment(text_source="data/raw (re-extracted in memory)")}),
        ("corpus", {"environment": _environment(corpus="current")}),
        ("document_count", {"environment": _environment(document_count=42)}),
        ("documents_digest", {"environment": _environment(documents_digest="0" * 64)}),
    ],
)
def test_round2_runs_must_share_their_run_conditions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str, f_kwargs: dict
) -> None:
    round2 = tmp_path / "round2"
    _write_results(round2, "e")
    _write_results(round2, "f", use_hybrid=True, **f_kwargs)

    runs = rag_eval.load_round2_runs(round2)
    assert list(rag_eval.round2_inconsistencies(runs)) == [field]

    out = tmp_path / "out"
    assert rag_eval.main(["--round1", str(EXPERIMENTS), "--round2", str(round2), "--out", str(out)]) == 1
    assert not (out / "metrics.md").exists()
    err = capsys.readouterr().err
    assert field in err
    assert "round2" in err  # 指出是哪個 run 資料夾


def test_combos_outside_the_preregistered_round2_set_are_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    round2 = tmp_path / "round2"
    _write_results(round2, "e")
    _write_results(round2, "a", environment=_environment(chunk_count=4))  # 不同切法的 A 不可混入
    out = tmp_path / "out"
    assert rag_eval.main(["--round1", str(EXPERIMENTS), "--round2", str(round2), "--out", str(out)]) == 0
    rows = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["rows"]
    assert [(r["run"], r["round"]) for r in rows][-1] == ("E\u2032 (control) · round1 corpus · round2", 2)
    assert [r["combo_label"] for r in rows if r["round"] == 2] == ["E\u2032 (control)"]
    assert "combo_a" in capsys.readouterr().err


def test_question_mismatches_detects_reordered_questions(tmp_path: Path) -> None:
    texts = _round1_questions()
    swapped = [texts[0], texts[3], texts[2], texts[1], texts[4]]  # Q2 與 Q4 對調
    run = rag_eval.load_round2_run(_write_results(tmp_path / "round2", "g", texts=swapped))
    reference = dict(enumerate(texts, start=1))
    assert rag_eval.question_mismatches(run, reference) == [2, 4]
    ok = rag_eval.load_round2_run(_write_results(tmp_path / "round2", "f"))
    assert rag_eval.question_mismatches(ok, reference) == []


def test_cli_drops_round2_run_with_other_questions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    round2 = tmp_path / "round2"
    _write_results(round2, "f", use_hybrid=True)
    texts = _round1_questions()
    _write_results(round2, "g", texts=[texts[0], "另一題", *texts[2:]])
    out = tmp_path / "out"
    assert rag_eval.main(["--round1", str(EXPERIMENTS), "--round2", str(round2), "--out", str(out)]) == 0
    rows = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["rows"]
    assert [r["run"] for r in rows] == ["A", "B", "C", "D", "E", "F · round1 corpus · round2"]
    assert "combo_g" in capsys.readouterr().err


def test_round2_rejects_missing_distance_to_question(tmp_path: Path) -> None:
    path = _write_results(tmp_path / "round2", "g")
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["questions"][0]["cites"][0]["distance_to_question"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        rag_eval.load_round2_run(path)


# ─── CLI ────────────────────────────────────────────────────────────────────


def test_cli_skips_round2_without_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out"
    code = rag_eval.main(
        ["--round1", str(EXPERIMENTS), "--round2", str(tmp_path / "missing"), "--out", str(out)]
    )
    assert code == 0
    assert "round 2" in capsys.readouterr().out.lower()
    rows = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["rows"]
    assert [r["run"] for r in rows] == ["A", "B", "C", "D", "E"]
    named = {r["run"]: r["q2_members_named"] for r in rows}
    assert named == {"A": 9, "B": 9, "C": 3, "D": 2, "E": 9}
    assert all(r["q4_fact_groups"] == 0 for r in rows)
    md = (out / "metrics.md").read_text(encoding="utf-8")
    assert (
        "| Run | Round | Retrieval | Mean cosine distance to question (5 Qs) | "
        "Q2 members named (/9) | Q2 members in context (/9) | "
        "Q4 Juste Debout chunks in top-k | Q4 fact groups in answer |"
    ) in md


def test_cli_includes_round2_rows(tmp_path: Path) -> None:
    round2 = tmp_path / "round2"
    _write_results(round2, "e")
    _write_results(round2, "f", use_hybrid=True, errors={4: "llm_error: RuntimeError"})
    out = tmp_path / "out"
    assert rag_eval.main(["--round1", str(EXPERIMENTS), "--round2", str(round2), "--out", str(out)]) == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert [r["run"] for r in metrics["rows"]] == [
        "A", "B", "C", "D", "E", "E\u2032 (control) · round1 corpus · round2", "F · round1 corpus · round2",
    ]
    assert [r["corpus"] for r in metrics["rows"]] == ["round1"] * 7
    (folder,) = metrics["round2_runs"]
    assert folder["shared"]["model"] == "gpt-test"
    assert folder["shared"]["corpus"] == "round1"
    assert folder["pinned_text"] is True
    md = (out / "metrics.md").read_text(encoding="utf-8")
    assert "gpt-test" in md
    assert "(round1 corpus · round2): model gpt-test" in md
    assert "indexed text matches the pinned text of its 39 documents" in md
    assert "- F · round1 corpus · round2 Q4: `llm_error: RuntimeError`; answer metrics are n/a." in md
    assert "max distance 0.65 (dense channel only)" in md
    assert str(tmp_path) not in md  # 不洩漏本機絕對路徑


def test_cli_records_the_gold_and_corpus_fingerprints(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert rag_eval.main(["--round1", str(EXPERIMENTS), "--round2", str(tmp_path / "none"), "--out", str(out)]) == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    gold_sha = hashlib.sha256((EXPERIMENTS / "gold.json").read_bytes()).hexdigest()
    assert metrics["gold_sha256"] == gold_sha
    assert metrics["corpus_pin"] == {"path": "experiments/corpus_pin.json", "digest": PINNED_DIGEST}
    assert metrics["round2_runs"] == []
    assert metrics["stability"] == []
    assert metrics["round1_corpus"]["document_count"] == 39
    md = (out / "metrics.md").read_text(encoding="utf-8")
    assert gold_sha in md
    assert "Within one run folder, rag_eval scores round 2 only when every combo shares the model" in md
    assert "No round-2 runs yet." in md
    assert "same model" not in md


# ─── round 2：多個 run 資料夾（重複執行）────────────────────────────────────


def _reference() -> dict[int, str]:
    return dict(enumerate(_round1_questions(), start=1))


def _write_run(run_dir: Path, *, environment: dict | None = None, model: str = "gpt-test") -> Path:
    """在 run_dir 寫入同一次執行的 E′、F、G 三組。"""
    _write_results(run_dir, "e", model=model, environment=environment)
    _write_results(run_dir, "f", use_hybrid=True, model=model, environment=environment)
    _write_results(run_dir, "g", use_expand=True, model=model, environment=environment)
    return run_dir


def _edit_results(path: Path, edit) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_run_folders_are_found_recursively_and_retired_copies_are_skipped(tmp_path: Path) -> None:
    root = tmp_path / "round2"
    _write_run(root / "round1_corpus" / "run1")
    _write_run(root / "round1_corpus" / "run2")
    _write_results(root / "current_corpus" / "run1", "e", environment=_current_environment())
    retired = root / "round1_corpus" / "run1" / ".previous" / "combo_f-20261109T000000Z"
    retired.mkdir(parents=True)
    (retired / "results.json").write_text("{}", encoding="utf-8")
    (root / "notes").mkdir()

    found = rag_eval.find_run_dirs([root])

    base = root.resolve()
    assert [p.relative_to(base).as_posix() for p in found] == [
        "current_corpus/run1",
        "round1_corpus/run1",
        "round1_corpus/run2",
    ]
    # 多個目錄（可重疊、可不存在）：去重後合併；目錄本身也可以是 run 資料夾
    assert rag_eval.find_run_dirs(
        [root / "round1_corpus", root / "round1_corpus" / "run2", tmp_path / "missing"]
    ) == found[1:]
    assert rag_eval.find_run_dirs([root / "round1_corpus" / "run2"]) == [found[2]]


def test_repeats_are_labelled_by_corpus_and_run_and_grouped_by_corpus(tmp_path: Path, gold: dict) -> None:
    root = tmp_path / "round2"
    for n in (2, 10, 1):
        _write_run(root / "round1_corpus" / f"run{n}")
    _write_run(root / "current_corpus" / "run1", environment=_current_environment())

    runs, folders = rag_eval.load_round2_folders([root], _reference())
    labels = [r["run"] for r in rag_eval.compute_metrics(runs, gold)]

    assert labels == [
        f"{combo} · {corpus} corpus · run {n}"
        for corpus, repeats in (("round1", (1, 2, 10)), ("current", (1,)))
        for n in repeats
        for combo in ("E′ (control)", "F", "G")
    ]
    assert [(f["corpus"], f["repeat"]) for f in folders] == [
        ("round1", "run 1"),
        ("round1", "run 2"),
        ("round1", "run 10"),
        ("current", "run 1"),
    ]
    assert folders[-1]["pinned_text"] is True  # current 語料的指紋對應 42 份鎖定文字


def test_round2_cli_accepts_several_directories(tmp_path: Path) -> None:
    _write_run(tmp_path / "a" / "round1_corpus" / "run1")
    _write_run(tmp_path / "b" / "current_corpus" / "run1", environment=_current_environment())
    out = tmp_path / "out"
    code = rag_eval.main(
        ["--round1", str(EXPERIMENTS), "--round2", str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(out)]
    )
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert [f["corpus"] for f in metrics["round2_runs"]] == ["round1", "current"]
    assert len([r for r in metrics["rows"] if r["round"] == 2]) == 6


def test_labels_name_the_model_when_a_corpus_has_several(tmp_path: Path, gold: dict) -> None:
    root = tmp_path / "round2"
    _write_run(root / "round1_corpus" / "run1")
    _write_run(root / "round1_corpus" / "run2", model="other-model")

    runs, _ = rag_eval.load_round2_folders([root], _reference())

    labels = {r["run"] for r in runs if r["combo_id"] == "f"}
    assert labels == {"F · round1 corpus · gpt-test · run 1", "F · round1 corpus · other-model · run 2"}
    stability = rag_eval.stability_rows(rag_eval.compute_metrics(runs, gold))
    assert {(s["model"], s["combo_id"], s["runs"]) for s in stability if s["combo_id"] == "f"} == {
        ("gpt-test", "f", 1),
        ("other-model", "f", 1),
    }


def test_duplicate_run_names_get_distinct_labels(tmp_path: Path) -> None:
    _write_run(tmp_path / "a" / "round1_corpus" / "run1")
    _write_run(tmp_path / "b" / "round1_corpus" / "run1")
    runs, folders = rag_eval.load_round2_folders([tmp_path / "a", tmp_path / "b"], _reference())
    assert sorted({r["run"] for r in runs if r["combo_id"] == "f"}) == [
        "F · round1 corpus · run 1 #1",
        "F · round1 corpus · run 1 #2",
    ]
    # repo 外的資料夾只顯示末兩層（不洩漏本機路徑），所以靠標籤區分
    assert [f["label"] for f in folders] == ["round1 corpus · run 1 #1", "round1 corpus · run 1 #2"]


def test_consistency_is_required_within_a_run_folder_but_not_across(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "round2"
    _write_run(root / "round1_corpus" / "run1")
    _write_run(root / "round1_corpus" / "run2", environment=_environment(git_commit="fffffff"))
    out = tmp_path / "out"
    assert rag_eval.main(["--round1", str(EXPERIMENTS), "--round2", str(root), "--out", str(out)]) == 0
    commits = [f["shared"]["git_commit"] for f in json.loads((out / "metrics.json").read_text(encoding="utf-8"))["round2_runs"]]
    assert commits == ["abc1234def", "fffffff"]

    # 同一個資料夾內混了兩份語料：語料是 run 身分的一部分 → 拒絕評估
    mixed = root / "round1_corpus" / "run3"
    _write_results(mixed, "e")
    _write_results(mixed, "f", use_hybrid=True, environment=_current_environment())
    assert rag_eval.main(["--round1", str(EXPERIMENTS), "--round2", str(root), "--out", str(tmp_path / "out2")]) == 1
    err = capsys.readouterr().err
    assert "run3" in err
    assert "corpus" in err
    assert not (tmp_path / "out2" / "metrics.md").exists()


def test_stability_reports_min_max_and_run_count_per_corpus_and_combo(tmp_path: Path, gold: dict) -> None:
    root = tmp_path / "round2"
    _write_run(root / "round1_corpus" / "run1")
    run2 = _write_run(root / "round1_corpus" / "run2")
    run3 = _write_run(root / "round1_corpus" / "run3")
    _write_run(root / "current_corpus" / "run1", environment=_current_environment())

    def fewer_members(data: dict) -> None:
        data["questions"][1]["answer"] = "成員有 Boogaloo Sam。"
        data["questions"][0]["cites"][0]["distance_to_question"] = 0.1  # Q1 平均 0.3 → 0.25，五題平均 0.36 → 0.35

    def llm_failed_q2(data: dict) -> None:
        data["questions"][1].update(error="llm_error: RuntimeError", answer="【LLM 呼叫失敗】")

    _edit_results(run2 / "combo_f" / "results.json", fewer_members)
    _edit_results(run3 / "combo_f" / "results.json", llm_failed_q2)

    runs, _ = rag_eval.load_round2_folders([root], _reference())
    stability = rag_eval.stability_rows(rag_eval.compute_metrics(runs, gold))

    assert [(s["corpus"], s["combo_id"], s["runs"]) for s in stability] == [
        ("round1", "e", 3),
        ("round1", "f", 3),
        ("round1", "g", 3),
        ("current", "e", 1),
        ("current", "f", 1),
        ("current", "g", 1),
    ]
    f = stability[1]
    assert f["combo_label"] == "F"
    assert f["metrics"]["q2_members_named"] == {"min": 1, "max": 2, "n": 2}  # run3 為 n/a
    assert f["metrics"]["mean_distance_to_question"]["min"] == pytest.approx(0.36 - 0.1 / 2 / 5)
    assert f["metrics"]["mean_distance_to_question"]["max"] == pytest.approx(0.36)
    assert f["metrics"]["q4_gold_chunks"] == {"min": 1, "max": 1, "n": 3}

    out = tmp_path / "out"
    assert rag_eval.main(["--round1", str(EXPERIMENTS), "--round2", str(root), "--out", str(out)]) == 0
    md = (out / "metrics.md").read_text(encoding="utf-8")
    stability_md = md.split("## Stability across repeats")[1]
    assert "| F | round1 | gpt-test | 3 | 0.350–0.360 | 1–2 (2 of 3 runs) | 3–3 | 1–1 | 3–3 |" in stability_md
    assert "| E′ (control) | current | gpt-test | 1 |" in stability_md
    assert md.index("## Results") < md.index("## Stability across repeats")


def test_metrics_header_states_the_known_confounds(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert rag_eval.main(["--round1", str(EXPERIMENTS), "--round2", str(tmp_path / "none"), "--out", str(out)]) == 0
    md = (out / "metrics.md").read_text(encoding="utf-8")
    header = md.split("## Metrics")[0]
    assert "first 128 tokens" in header
    assert "BM25 indexes the full chunk text" in header
    assert "`Pete`" in header and "loose" in header
    assert "39 documents" in header
    assert "gemini-2.5-flash" in header and "course proxy" in header
    assert "model and corpus recorded in their results.json" in header


# ─── round-1 重現檢查（--check-reproduction）────────────────────────────────

_REPRO_COMBO_KEYS = ("chunk_desc", "top_k", "max_distance", "truncate_to", "prompt_style", "use_hybrid", "use_expand")


def _write_reproduction_run(
    run_dir: Path, *, offset: float = 0.0004, corpus: str = "round1", skip: tuple[str, ...] = ()
) -> Path:
    """依 round-1 的 q*.md 寫出逐筆重現的 combo_a..e/results.json（距離加上 offset）與 run.json。"""
    environment = _environment() if corpus == "round1" else _current_environment()
    for combo_id, spec in rag_eval.ROUND1_COMBOS.items():
        if combo_id in skip:
            continue
        questions = []
        for n in range(1, 6):
            parsed = _round1(combo_id, n)
            cites = [
                {
                    "chunk_id": f"{src}::{idx}",
                    "source_file": src,
                    "chunk_index": idx,
                    "section_title": "",
                    "distance": round(d + offset, 6),
                    "distance_to_question": round(d + offset, 6),
                    "retrieved_by": "dense",
                    "text": "t",
                }
                for src, idx, d in parsed["cites"]
            ]
            questions.append(
                {"index": n, "question": parsed["question"], "expanded_queries": None, "cites": cites,
                 "answer": "", "error": None}
            )
        payload = {
            "combo": {
                "id": combo_id,
                "name": f"Combo {combo_id.upper()}",
                **{k: spec[k] for k in _REPRO_COMBO_KEYS},
                "collection": f"cypher_og_{corpus}_combo_{combo_id}",
            },
            "model": "gemini-2.5-flash",
            "retrieval_only": True,
            "environment": environment,
            "generated_at": "2026-11-09T00:00:00+00:00",
            "questions": questions,
        }
        path = run_dir / f"combo_{combo_id}" / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    run = {
        "generated_at": "2026-11-09T00:00:00+00:00",
        "git": {"commit": "abc1234def", "dirty": False},
        "llm": {"model": "gemini-2.5-flash", "proxy_configured": False},
        "embedding": {"model": "paraphrase-multilingual-MiniLM-L12-v2", "max_seq_length": 128},
        "corpus": {
            "name": corpus,
            "definition": "experiments/corpus_round1.json" if corpus == "round1" else "data/raw (all supported files)",
            "document_count": environment["document_count"],
            "documents_digest": environment["documents_digest"],
            "processed_digest": environment["corpus_sha256"],
            "matches_pin": True,
        },
        "reuse_processed": True,
        "retrieval_only": True,
        "combos": {
            cid: {"collection": f"cypher_og_{corpus}_combo_{cid}", "chunk_count": 1248, "status": "ok"}
            for cid in rag_eval.ROUND1_COMBOS
            if cid not in skip
        },
    }
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return run_dir


def _check(run_dir: Path, report: Path) -> int:
    return rag_eval.main(["--check-reproduction", str(run_dir), "--report-out", str(report)])


def test_reproduction_cell_comparison() -> None:
    recorded = [("a.txt", 1, 0.300), ("b.txt", 2, 0.400)]
    ok = rag_eval.compare_reproduction(recorded, [("a.txt", 1, 0.3009), ("b.txt", 2, 0.3995 + 0.0)])
    assert ok["match"] is True
    assert ok["max_abs_distance_diff"] == pytest.approx(0.0009)
    far = rag_eval.compare_reproduction(recorded, [("a.txt", 1, 0.3011), ("b.txt", 2, 0.4)])
    assert far["match"] is False and far["same_chunks"] is True
    inserted = rag_eval.compare_reproduction(recorded, [("new.txt", 0, 0.2), ("a.txt", 1, 0.3)])
    assert inserted["match"] is False
    assert inserted["added"] == ["new.txt::0"] and inserted["removed"] == ["b.txt::2"]
    swapped = rag_eval.compare_reproduction(recorded, [("b.txt", 2, 0.4), ("a.txt", 1, 0.3)])
    assert swapped["match"] is False and swapped["added"] == [] and swapped["removed"] == []
    assert rag_eval.compare_reproduction(recorded, None)["match"] is False
    assert rag_eval.compare_reproduction([], [])["match"] is True


def test_exact_reproduction_reports_25_of_25(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = _write_reproduction_run(tmp_path / "run")
    report = tmp_path / "report.md"

    assert _check(run_dir, report) == 0

    out = capsys.readouterr().out
    assert "25/25" in out
    assert "| Combo | Q1 | Q2 | Q3 | Q4 | Q5 | Matched |" in out
    text = report.read_text(encoding="utf-8")
    assert "25/25" in text
    assert "## Method" in text and "## Environment" in text and "## Corpus" in text and "## Result" in text
    assert "≤ 0.001" in text
    assert "round1" in text and "39" in text
    assert "popping_glossary.md" in text  # round1 語料排除的文件
    assert "abc1234" in text and "max_seq_length 128" in text
    assert "| A | ok" in text
    assert str(tmp_path) not in text  # 不洩漏本機絕對路徑


@pytest.mark.parametrize(("offset", "code", "matched"), [(0.0009, 0, "25/25"), (0.0015, 1, "0/25")])
def test_reproduction_tolerance_is_one_thousandth(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], offset: float, code: int, matched: str
) -> None:
    run_dir = _write_reproduction_run(tmp_path / "run", offset=offset)
    assert _check(run_dir, tmp_path / "report.md") == code
    assert matched in capsys.readouterr().out


def test_an_inserted_chunk_is_a_mismatch_and_is_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = _write_reproduction_run(tmp_path / "run")

    def insert(data: dict) -> None:
        cites = data["questions"][1]["cites"]
        new = {**cites[0], "chunk_id": "popping_glossary.txt::4", "source_file": "popping_glossary.txt", "chunk_index": 4}
        data["questions"][1]["cites"] = [cites[0], new, *cites[1:-1]]

    _edit_results(run_dir / "combo_b" / "results.json", insert)
    report = tmp_path / "report.md"

    assert _check(run_dir, report) == 1

    assert "24/25" in capsys.readouterr().out
    text = report.read_text(encoding="utf-8")
    assert "24/25" in text
    assert "MISMATCH" in text
    assert "+popping_glossary.txt::4" in text


def test_reordered_chunks_are_a_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = _write_reproduction_run(tmp_path / "run")

    def swap(data: dict) -> None:
        cites = data["questions"][0]["cites"]
        data["questions"][0]["cites"] = [cites[1], cites[0], *cites[2:]]

    _edit_results(run_dir / "combo_c" / "results.json", swap)
    assert _check(run_dir, tmp_path / "report.md") == 1
    assert "24/25" in capsys.readouterr().out
    assert "order differs" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_a_missing_combo_counts_as_five_mismatches(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = _write_reproduction_run(tmp_path / "run", skip=("e",))
    assert _check(run_dir, tmp_path / "report.md") == 1
    assert "20/25" in capsys.readouterr().out
    assert "missing" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_a_combo_with_other_settings_is_a_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = _write_reproduction_run(tmp_path / "run")
    _edit_results(run_dir / "combo_d" / "results.json", lambda data: data["combo"].update(truncate_to=None))
    assert _check(run_dir, tmp_path / "report.md") == 1
    assert "20/25" in capsys.readouterr().out
    assert "settings differ" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_reproduction_report_defaults_to_experiments(tmp_path: Path) -> None:
    assert rag_eval.REPRODUCTION_REPORT_PATH == EXPERIMENTS / "round1_reproduction.md"
    args = rag_eval.parse_args(["--check-reproduction", str(tmp_path)])
    assert args.check_reproduction == tmp_path
    assert args.report_out is None


@pytest.mark.parametrize(
    "target",
    [
        "experiments/combo_a/report.md",
        "experiments/summary.md",
        "data/report.md",
        "Experiments/Summary.md",  # macOS APFS 預設不分大小寫
        "EXPERIMENTS/Combo_E/report.md",
        "experiments/report.txt",
    ],
)
def test_reproduction_report_never_lands_in_protected_places(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], target: str
) -> None:
    # 假的 repo：即使守衛失效，也只會寫進 tmp_path，不會碰到真正的 experiments/
    fake_root = tmp_path / "repo"
    for sub in ("experiments/combo_a", "experiments/combo_e", "data"):
        (fake_root / sub).mkdir(parents=True)
    (fake_root / "experiments" / "summary.md").write_text("round-1 summary", encoding="utf-8")
    monkeypatch.setattr(rag_eval, "ROOT", fake_root)
    monkeypatch.setattr(rag_eval, "EXPERIMENTS_DIR", fake_root / "experiments")
    run_dir = _write_reproduction_run(tmp_path / "run")
    before = {p.relative_to(fake_root).as_posix(): p.stat().st_mtime_ns for p in fake_root.rglob("*")}

    assert _check(run_dir, fake_root / target) == 2

    assert {p.relative_to(fake_root).as_posix(): p.stat().st_mtime_ns for p in fake_root.rglob("*")} == before
    assert (fake_root / "experiments" / "summary.md").read_text(encoding="utf-8") == "round-1 summary"
    assert "report" in capsys.readouterr().err


def test_reproduction_report_may_go_to_a_new_file_under_experiments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_root = tmp_path / "repo"
    (fake_root / "experiments").mkdir(parents=True)
    monkeypatch.setattr(rag_eval, "ROOT", fake_root)
    monkeypatch.setattr(rag_eval, "EXPERIMENTS_DIR", fake_root / "experiments")
    run_dir = _write_reproduction_run(tmp_path / "run")
    report = fake_root / "experiments" / "round1_reproduction.md"
    argv = ["--check-reproduction", str(run_dir), "--round1", str(EXPERIMENTS), "--report-out", str(report)]
    assert rag_eval.main(argv) == 0
    assert "25/25" in (fake_root / "experiments" / "round1_reproduction.md").read_text(encoding="utf-8")
