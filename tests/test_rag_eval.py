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


def _environment(**overrides) -> dict:
    return {
        "git_commit": "abc1234def",
        "git_dirty": False,
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "text_source": "data/processed",
        "corpus_sha256": PINNED_DIGEST,
        "chunk_count": 1234,
        "index_rebuilt": True,
        **overrides,
    }


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
    assert field in capsys.readouterr().err


def test_combos_outside_the_preregistered_round2_set_are_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    round2 = tmp_path / "round2"
    _write_results(round2, "e")
    _write_results(round2, "a", environment=_environment(chunk_count=4))  # 不同切法的 A 不可混入
    out = tmp_path / "out"
    assert rag_eval.main(["--round1", str(EXPERIMENTS), "--round2", str(round2), "--out", str(out)]) == 0
    rows = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["rows"]
    assert [(r["run"], r["round"]) for r in rows][-1] == ("E\u2032 (control)", 2)
    assert [r["run"] for r in rows if r["round"] == 2] == ["E\u2032 (control)"]
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
    assert [r["run"] for r in rows] == ["A", "B", "C", "D", "E", "F"]
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
    assert [r["run"] for r in metrics["rows"]] == ["A", "B", "C", "D", "E", "E\u2032 (control)", "F"]
    assert metrics["round2_shared"]["model"] == "gpt-test"
    md = (out / "metrics.md").read_text(encoding="utf-8")
    assert "gpt-test" in md
    assert "checked that every round-2 run shares the model (gpt-test)" in md
    assert "Round-2 indexed text matches the pinned round-1 corpus." in md
    assert "- F Q4: `llm_error: RuntimeError`; answer metrics are n/a." in md
    assert "max distance 0.65 (dense channel only)" in md
    assert str(tmp_path) not in md  # 不洩漏本機絕對路徑


def test_cli_records_the_gold_and_corpus_fingerprints(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert rag_eval.main(["--round1", str(EXPERIMENTS), "--round2", str(tmp_path / "none"), "--out", str(out)]) == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    gold_sha = hashlib.sha256((EXPERIMENTS / "gold.json").read_bytes()).hexdigest()
    assert metrics["gold_sha256"] == gold_sha
    assert metrics["corpus_pin"] == {"path": "experiments/corpus_pin.json", "digest": PINNED_DIGEST}
    assert metrics["round2_shared"] is None
    md = (out / "metrics.md").read_text(encoding="utf-8")
    assert gold_sha in md
    assert "scores round 2 only when every round-2 run shares the model" in md
    assert "same model" not in md
