#!/usr/bin/env python3
"""RAG 檢索策略實驗跑分腳本。

每個 combo 使用獨立的 Chroma collection（cypher_og_combo_<id>）：重建索引後對固定 5 題
執行檢索（＋生成），結果寫到 --out 指定的目錄（預設 experiments/round2，與 rag_eval.py 一致）。

歷史紀錄：
  第一輪（2026-04）只跑了 Combo A–E，結果在 experiments/combo_a … combo_e 與
  experiments/summary.md。這些是當時的實驗證據，本腳本拒絕把 --out 指向 experiments/
  本身或其中的 combo_* 目錄（以檔案系統身分比對，大小寫不同的拼法也會拒絕），也拒絕
  repo 內 experiments/ 以外的位置（例如 data/raw）。之後的實驗請寫到子目錄，例如
  --out experiments/round2。

嚴格模式（避免把退化的結果記成有效實驗）：
  - collection 為空、或 Hybrid combo（F）沒有可用的 BM25 索引 → 該 combo 失敗，不寫 results.json。
  - Query Expansion（G）某題擴展失敗 → 該題不檢索，error 記為 "expansion_failed: <原因>"。
  - 任一題 LLM 失敗或擴展失敗 → combo 狀態記為 degraded；結束碼非 0。
  - 重跑同一 combo 時，先把舊的 combo_<id>/ 移到 <out>/.previous/，失敗的重跑不會留下舊 results.json。

Combo 一覽：
  A: Baseline（字元暴力切 + 無距離過濾 + 弱 prompt）
  B: 段落感知 chunk + 距離過濾
  C: 大 chunk + 強 prompt
  D: 小 chunk + top-10 距離截斷為前 5（沒有 reranker 模型；結果等同 top-5）
  E: CoT prompt（檢索設定同 B）
  F: E + Hybrid Search（向量 + BM25，以標準 RRF 融合）
  G: E + Query Expansion（LLM 改寫子查詢；多查詢結果去重並保留最小距離）

輸出（皆在 <out> 之下）：
  combo_<id>/config.md、combo_<id>/q<n>.md、combo_<id>/results.json（給 rag_eval.py 讀）
  run.json    執行環境紀錄（git commit、模型、embedding、語料數、各 combo chunk 數）
  summary.md  彙整 <out> 內所有 combo_*/results.json（分次執行 f、g 也會合併成一份）

預設以 data/processed 中已提交的文字建索引，且先比對 experiments/corpus_pin.json，不一致就拒絕執行；
加 --re-extract 才會從 data/raw 重新抽取（只在記憶體中，不改寫 data/processed 與 ingest manifest）。

Usage:
    python experiment_runner.py --combos e f g --out experiments/round2
    python experiment_runner.py --combos f --out experiments/round2 --skip-rebuild
    python experiment_runner.py --combos e --retrieval-only   # 只檢索、不生成回答
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# rag_common 匯入時會載入本 repo 的 .env（只讀 <repo>/.env，不往上層目錄搜尋）
from data_update import (
    chunk_text,
    list_raw_documents,
    naive_chunk_text,
    rebuild_processed_and_index,
)
from rag_common import (
    CORPUS_PIN_PATH,
    EMBEDDING_MODEL,
    corpus_digest,
    corpus_mismatches,
    describe_exception,
    display_path,
    effective_llm_model,
    get_chroma_collection,
    litellm_chat_completion,
    llm_proxy_configured,
    load_bm25_index,
    load_corpus_pin,
    loaded_embedding_max_seq_length,
    processed_corpus_hashes,
)
from rag_query import (
    build_messages,
    cosine_distances_to_query,
    expand_query,
    hybrid_retrieve,
    retrieve_context,
)

# ─── 固定測試問題 ─────────────────────────────────────────────────────────────
QUESTIONS: list[str] = [
    "What is Popping and who created it?",
    "Electric Boogaloos 的成員有哪些？他們各自的貢獻是什麼？",
    "Popping 和 Boogaloo 有什麼差別？",
    "Juste Debout 比賽的規則和歷史？",
    "如何判斷一場 Popping battle 的勝負？",
]

# ─── Combo 設定 ───────────────────────────────────────────────────────────────
COMBOS: list[dict] = [
    {
        "id": "a",
        "name": "Combo A — Baseline（字元暴力切）",
        "desc": "最差基準線：純字元切割、無距離過濾、弱 prompt。",
        "collection": "cypher_og_combo_a",
        "chunk_fn": lambda t: naive_chunk_text(t, 500),
        "chunk_desc": "naive / size=500 / overlap=0",
        "top_k": 5,
        "max_distance": 1.0,
        "truncate_to": None,
        "prompt_style": "weak",
        "use_hybrid": False,
        "use_expand": False,
    },
    {
        "id": "b",
        "name": "Combo B — 段落感知 + 距離過濾",
        "desc": "升級 chunking 與 retrieval，保持弱 prompt，單獨觀察前兩項改善效果。",
        "collection": "cypher_og_combo_b",
        "chunk_fn": lambda t: chunk_text(t, 600, 100),
        "chunk_desc": "recursive / size=600 / overlap=100",
        "top_k": 5,
        "max_distance": 0.65,
        "truncate_to": None,
        "prompt_style": "weak",
        "use_hybrid": False,
        "use_expand": False,
    },
    {
        "id": "c",
        "name": "Combo C — 大 chunk + 強 prompt",
        "desc": "大 chunk 保留完整語境，搭配強制規則 prompt 與 XML 標籤，觀察幻覺減少效果。",
        "collection": "cypher_og_combo_c",
        "chunk_fn": lambda t: chunk_text(t, 1200, 200),
        "chunk_desc": "recursive / size=1200 / overlap=200",
        "top_k": 3,
        "max_distance": 0.65,
        "truncate_to": None,
        "prompt_style": "strong",
        "use_hybrid": False,
        "use_expand": False,
    },
    {
        "id": "d",
        "name": "Combo D — 小 chunk + top-10 距離截斷",
        "desc": (
            "小 chunk + 嚴格距離過濾；先取距離最近的 10 個再截斷為前 5 個。"
            "沒有 reranker 模型，結果等同直接取 top-5。"
        ),
        "collection": "cypher_og_combo_d",
        "chunk_fn": lambda t: chunk_text(t, 300, 50),
        "chunk_desc": "recursive / size=300 / overlap=50",
        "top_k": 10,
        "max_distance": 0.5,
        "truncate_to": 5,
        "prompt_style": "strong",
        "use_hybrid": False,
        "use_expand": False,
    },
    {
        "id": "e",
        "name": "Combo E — CoT + 結構化 context",
        "desc": "與 B 相同的 chunking/retrieval，但加 Chain-of-Thought prompt，強制 LLM 先列 facts 再綜合。",
        "collection": "cypher_og_combo_e",
        "chunk_fn": lambda t: chunk_text(t, 600, 100),
        "chunk_desc": "recursive / size=600 / overlap=100",
        "top_k": 5,
        "max_distance": 0.65,
        "truncate_to": None,
        "prompt_style": "cot",
        "use_hybrid": False,
        "use_expand": False,
    },
    {
        "id": "f",
        "name": "Combo F — CoT + Hybrid Search（BM25+Vector+RRF）",
        "desc": (
            "在 Combo E 基礎上加入 BM25 詞彙檢索，與向量結果以標準 RRF 融合；"
            "距離門檻只作用於向量通道，只被 BM25 找到的 chunk 也能入選。"
        ),
        "collection": "cypher_og_combo_f",
        "chunk_fn": lambda t: chunk_text(t, 600, 100),
        "chunk_desc": "recursive / size=600 / overlap=100",
        "top_k": 5,
        "max_distance": 0.65,
        "truncate_to": None,
        "prompt_style": "cot",
        "use_hybrid": True,
        "use_expand": False,
    },
    {
        "id": "g",
        "name": "Combo G — CoT + Query Expansion（多角度子查詢）",
        "desc": (
            "在 Combo E 基礎上加入 Query Expansion：用 LLM 將查詢改寫為中英文多角度子查詢，"
            "各子查詢結果去重並保留最小距離。"
        ),
        "collection": "cypher_og_combo_g",
        "chunk_fn": lambda t: chunk_text(t, 600, 100),
        "chunk_desc": "recursive / size=600 / overlap=100",
        "top_k": 5,
        "max_distance": 0.65,
        "truncate_to": None,
        "prompt_style": "cot",
        "use_hybrid": False,
        "use_expand": True,
    },
]

COMBO_MAP = {c["id"]: c for c in COMBOS}

# results.json 中 "combo" 物件的欄位（不含 chunk_fn 等不可序列化的設定）
COMBO_PUBLIC_KEYS = (
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
)

NO_CONTEXT_ANSWER = "【知識庫中未找到相關資料，無法生成回答。】"

# 單題 error 中代表「本題結果有效」的值；其餘（llm_error、expansion_failed）會讓 combo 記為 degraded
OK_QUESTION_ERRORS: frozenset[str | None] = frozenset({None, "no_context"})

# ─── 輸出目錄 ──────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = ROOT / "experiments"
DEFAULT_OUT_DIR = EXPERIMENTS_DIR / "round2"


class ExperimentError(RuntimeError):
    """實驗前置條件不成立（collection 為空、缺 BM25 索引等）；訊息由本程式產生。"""


def _same_dir(path: Path, target: Path) -> bool:
    """path 與 target 是否為同一個目錄：存在時以檔案系統身分（samefile）比對，否則以 casefold 字串比對。

    macOS 預設的 APFS 不分大小寫，Path.resolve() 保留使用者輸入的大小寫，單純字串比對會被
    「EXPERIMENTS」之類的拼法繞過。
    """
    try:
        if path.exists() and target.exists():
            return os.path.samefile(path, target)
    except OSError as exc:
        _warn(f"[experiment_runner] 警告：無法比對路徑（{type(exc).__name__}），改用字串比對。")
    return str(path).casefold() == str(target).casefold()


def resolve_out_dir(raw: str | os.PathLike[str]) -> Path:
    """解析 --out；不符合規則時拋 ValueError。

    拒絕：experiments/ 本身、第一輪的 experiments/combo_*（含其子目錄，不分大小寫）、
    repo 根目錄與 repo 內 experiments/ 以外的位置（例如 data/raw：summary.md 會被當成語料）、
    以及已存在的檔案。repo 外的目錄（例如暫存目錄）允許。
    """
    out = Path(raw).expanduser().resolve()
    experiments = EXPERIMENTS_DIR.resolve()
    root = ROOT.resolve()
    if out.exists() and not out.is_dir():
        raise ValueError(f"--out 必須是目錄，{display_path(out)} 是已存在的檔案")
    for ancestor in (out, *out.parents):
        if _same_dir(ancestor, experiments):
            if ancestor == out:
                raise ValueError(
                    "--out 不可指向 experiments/ 本身：第一輪 Combo A–E 的結果存放於此，不可覆寫。"
                    "請改用子目錄，例如 --out experiments/round2"
                )
            first = out.relative_to(ancestor).parts[0]
            if first.casefold().startswith("combo_"):
                raise ValueError(
                    f"--out 不可位於第一輪結果目錄 experiments/{first} 之內。"
                    "請改用例如 --out experiments/round2"
                )
            return out
        if _same_dir(ancestor, root):
            raise ValueError(
                f"--out 在 repo 內時必須位於 experiments/ 之下（收到 {display_path(out) or '.'}）。"
                "請改用例如 --out experiments/round2"
            )
    return out


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _warn(message: str) -> None:
    print(message, file=sys.stderr)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mean(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None and not math.isnan(v)]
    return sum(nums) / len(nums) if nums else None


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def _combo_public(combo: dict) -> dict:
    return {k: combo.get(k) for k in COMBO_PUBLIC_KEYS}


def _cite_records(question: str, cites: list[dict], collection: str) -> list[dict]:
    """把 retriever 的 cites 轉成 results.json 的 cite 格式，並補上與「原始問題」的 cosine distance。

    契約要求 distance_to_question 必為數值；cite 皆取自同一 collection，若查不到代表索引不一致，直接拋錯。
    """
    if not cites:
        return []
    ids = [c.get("chunk_id") or f"{c['source_file']}::{c['chunk_index']}" for c in cites]
    to_question = cosine_distances_to_query(question, ids, collection_name=collection)
    missing = [cid for cid in ids if cid not in to_question]
    if missing:
        raise RuntimeError(f"collection {collection!r} 中找不到 {missing}，無法計算與原始問題的距離")
    records: list[dict] = []
    for cid, c in zip(ids, cites):
        dist_q = to_question[cid]
        records.append(
            {
                "chunk_id": cid,
                "source_file": c["source_file"],
                "chunk_index": int(c["chunk_index"]),
                "section_title": c.get("section_title", ""),
                "distance": float(c["distance"]),
                "distance_to_question": dist_q,
                "retrieved_by": c.get("retrieved_by", "dense"),
                "text": c.get("text", ""),
            }
        )
    return records


def _expansion_failure(question: str, reason: str, *, retrieval_only: bool) -> dict:
    """Query Expansion 失敗時的單題結果：不檢索、不生成，error 記為 expansion_failed（rag_eval 視為 n/a）。"""
    _warn(f"[experiment_runner] 警告：查詢擴展失敗（{reason}），本題不檢索，記為 expansion_failed。")
    return {
        "question": question,
        "expanded_queries": None,
        "context": "",
        "cites": [],
        "answer": "" if retrieval_only else f"【查詢擴展失敗：{reason}】",
        "error": f"expansion_failed: {reason}",
    }


def _run_query(combo: dict, question: str, *, retrieval_only: bool = False) -> dict:
    """執行單題檢索（＋生成），回傳結構化結果。

    expanded_queries：Query Expansion 產生的子查詢（不含原始問題）；未啟用或擴展失敗時為 None。
    擴展失敗（LLM 例外或沒有產生任何新子查詢）時不退回純向量檢索，而是回傳 error
    "expansion_failed: <原因>" 的結果；Hybrid combo 以 strict 模式檢索，缺 BM25 索引時直接拋錯。
    cites 中每筆含 distance（retriever 回報；G 可能相對於子查詢）與 distance_to_question
    （與原始問題的 cosine distance）。retrieval_only 時不呼叫 LLM 生成回答（G 仍會呼叫 LLM 擴展查詢）。
    """
    extra_queries: list[str] | None = None
    if combo.get("use_expand"):
        try:
            extra_queries = expand_query(question, model=None, strict=True)[1:]
        except Exception as exc:
            # 只記錄例外類型：供應商錯誤訊息可能含 endpoint 或部分金鑰
            return _expansion_failure(question, type(exc).__name__, retrieval_only=retrieval_only)
        if not extra_queries:
            return _expansion_failure(question, "no_subqueries", retrieval_only=retrieval_only)

    if combo.get("use_hybrid"):
        context, cites = hybrid_retrieve(
            question,
            top_k=combo["top_k"],
            max_distance=combo["max_distance"],
            collection_name=combo["collection"],
            extra_queries=extra_queries,
            strict=True,
        )
    else:
        context, cites = retrieve_context(
            question,
            top_k=combo["top_k"],
            max_distance=combo["max_distance"],
            truncate_to=combo.get("truncate_to"),
            collection_name=combo["collection"],
            extra_queries=extra_queries,
        )

    base = {
        "question": question,
        "expanded_queries": extra_queries,
        "context": context,
        "cites": _cite_records(question, cites, combo["collection"]),
    }
    if not context.strip():
        return {**base, "answer": "" if retrieval_only else NO_CONTEXT_ANSWER, "error": "no_context"}
    if retrieval_only:
        return {**base, "answer": "", "error": None}

    messages = build_messages(question, context, [], prompt_style=combo["prompt_style"])
    try:
        resp = litellm_chat_completion(model=None, messages=messages, temperature=0.2)
        return {**base, "answer": resp.choices[0].message.content or "", "error": None}
    except Exception as exc:
        # 只記錄例外類型：供應商錯誤訊息可能含 endpoint 或部分金鑰
        name = type(exc).__name__
        _warn(f"[experiment_runner] 警告：LLM 呼叫失敗（{name}）。")
        return {**base, "answer": f"【LLM 呼叫失敗：{name}】", "error": f"llm_error: {name}"}


def _settings_rows(combo: dict, model: str) -> list[tuple[str, str]]:
    truncate_to = combo.get("truncate_to")
    return [
        ("Chunking", combo["chunk_desc"]),
        ("top-k", str(combo["top_k"])),
        ("max_distance", str(combo["max_distance"])),
        ("距離截斷（truncate_to）", f"前 {truncate_to} 個" if truncate_to else "否"),
        ("Prompt 風格", combo["prompt_style"]),
        ("Hybrid Search", "是 (BM25+Vector+RRF)" if combo.get("use_hybrid") else "否"),
        ("Query Expansion", "是" if combo.get("use_expand") else "否"),
        ("LLM 模型", model),
        ("Chroma collection", f"`{combo['collection']}`"),
    ]


def _table(rows: list[tuple[str, str]], header: tuple[str, str]) -> list[str]:
    return [f"| {header[0]} | {header[1]} |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows]


def format_question_md(
    q_idx: int, result: dict, combo: dict, model: str, *, retrieval_only: bool = False
) -> str:
    """單題結果的 Markdown（與第一輪 q<n>.md 相同段落，另加模型、擴展查詢與 distance_to_question）。"""
    cites = result["cites"]
    avg = _mean([c["distance"] for c in cites])
    avg_q = _mean([c["distance_to_question"] for c in cites])
    lines = [f"# Q{q_idx + 1}: {result['question']}", "", "## 實驗設定", ""]
    lines += _table(
        [("Combo", combo["name"])] + _settings_rows(combo, model), ("項目", "值")
    )
    if result.get("error") and result["error"] not in OK_QUESTION_ERRORS:
        lines += ["", f"> 本題失敗：`{result['error']}`（rag_eval 不計分）"]
    if result.get("expanded_queries") is not None:
        lines += ["", "## 擴展查詢", ""]
        lines += [f"- {q}" for q in result["expanded_queries"]]
    lines += [
        "",
        f"## 檢索結果（{len(cites)} 個 chunks，平均距離 {_fmt(avg)}，與原問題平均距離 {_fmt(avg_q)}）",
        "",
    ]
    if cites:
        lines += [
            f"[{i}] `{c['source_file']}` chunk {c['chunk_index']}"
            + (f" [{c['section_title']}]" if c.get("section_title") else "")
            + f" (距離: {c['distance']:.3f}，與原問題距離: {_fmt(c['distance_to_question'])}，"
            + f"來源: {c['retrieved_by']})"
            for i, c in enumerate(cites, 1)
        ]
    else:
        lines.append("（無相關 chunks）")
    answer = "（--retrieval-only：未生成回答）" if retrieval_only else result["answer"]
    lines += [
        "",
        "## LLM 回答",
        "",
        answer,
        "",
        "---",
        f"*generated by experiment_runner.py at {_ts()}*",
        "",
    ]
    return "\n".join(lines)


def _write_combo_config(combo_dir: Path, combo: dict, model: str) -> None:
    lines = [f"# {combo['name']}", "", combo["desc"], ""]
    lines += _table(_settings_rows(combo, model), ("設定", "值"))
    lines += ["", f"*generated at {_ts()}*", ""]
    (combo_dir / "config.md").write_text("\n".join(lines), encoding="utf-8")


def build_environment(*, git: Mapping[str, Any], reuse_processed: bool) -> dict:
    """results.json 的 environment 基本欄位（rag_eval 用來確認 round-2 各組可比）；不含金鑰或 endpoint。"""
    return {
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "embedding_model": EMBEDDING_MODEL,
        "text_source": "data/processed" if reuse_processed else "data/raw (re-extracted in memory)",
    }


def question_failures(results: list[dict]) -> list[int]:
    """回傳結果無效（llm_error、expansion_failed 等）的題號（從 1 起算）。"""
    return [i for i, r in enumerate(results, 1) if r.get("error") not in OK_QUESTION_ERRORS]


def build_results_payload(
    combo: dict,
    results: list[dict],
    *,
    model: str,
    retrieval_only: bool,
    environment: Mapping[str, Any] | None = None,
) -> dict:
    """依 results.json 契約組出 payload（rag_eval.py 讀取此格式）。

    environment：git_commit、git_dirty、embedding_model、text_source、corpus_sha256（建索引所用文字的指紋）、
    chunk_count、index_rebuilt；rag_eval 要求 round-2 各組的這些值（index_rebuilt、git_dirty 除外）一致。
    """
    return {
        "combo": _combo_public(combo),
        "model": model,
        "retrieval_only": retrieval_only,
        "environment": dict(environment or {}),
        "generated_at": _iso_now(),
        "questions": [
            {
                "index": i,
                "question": r["question"],
                "expanded_queries": r["expanded_queries"],
                "cites": r["cites"],
                "answer": r["answer"],
                "error": r["error"],
            }
            for i, r in enumerate(results, 1)
        ],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _unique_path(path: Path) -> Path:
    """path 已存在時在副檔名前加上 -1、-2…，回傳第一個不存在的路徑。"""
    candidate, n = path, 0
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
    return candidate


def retire_previous(path: Path) -> Path | None:
    """把既有的輸出（combo_<id>/ 目錄或 summary.md）移到 <out>/.previous/<名稱>-<UTC 時間>（不刪除），回傳新位置。

    .previous/ 不符合 combo_*/results.json 的 glob，summary 與 rag_eval 都不會讀到舊結果；
    重跑失敗時 <out> 內就不會留下上一次的 results.json 被當成本次結果。
    """
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = _unique_path(path.parent / ".previous" / f"{path.stem}-{stamp}{path.suffix}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, dest)
    print(f"  先前的 {path.name} 已移到 .previous/{dest.name}")
    return dest


def _require_bm25_index(collection: str, chunk_count: int) -> None:
    """Hybrid combo 的前置檢查：BM25 索引必須可用，且 chunk 數與 collection 相同。"""
    bm25, ids, _ = load_bm25_index(collection)
    if bm25 is None:
        raise ExperimentError(
            f"Hybrid combo 需要 collection {collection!r} 的 BM25 索引，但索引不存在、損毀或未安裝 rank_bm25；"
            "不以純向量結果冒充 Hybrid Search。請不加 --skip-rebuild 重跑。"
        )
    if len(ids) != chunk_count:
        raise ExperimentError(
            f"collection {collection!r} 的 BM25 索引有 {len(ids)} 個 chunk，向量庫有 {chunk_count} 個；"
            "索引過期，請不加 --skip-rebuild 重跑。"
        )


def run_combo(
    combo: dict,
    out_dir: Path,
    *,
    model: str,
    skip_rebuild: bool = False,
    retrieval_only: bool = False,
    reuse_processed: bool = True,
    environment: Mapping[str, Any] | None = None,
) -> dict:
    """跑單一 combo：（重建索引）→ 5 題 → 寫 config.md、q<n>.md、results.json。

    開始前再次套用 --out 規則（第二道防線），並把既有的 combo_<id>/ 移到 <out>/.previous/。
    重建索引一律不寫入 data/（write_processed=False）。collection 為空、或 Hybrid combo 沒有
    可用的 BM25 索引時拋 ExperimentError，不寫 results.json。
    environment 為 build_environment() 的結果；None 時於此計算。
    回傳 {"chunk_count", "corpus_sha256", "status": "ok" | "degraded", "failed_questions": [題號]}。
    """
    resolve_out_dir(out_dir)
    print(f"\n{'='*60}")
    print(f"  {combo['id'].upper()}: {combo['name']}")
    print(f"{'='*60}")

    combo_dir = out_dir / f"combo_{combo['id']}"
    retire_previous(combo_dir)
    combo_dir.mkdir(parents=True)
    _write_combo_config(combo_dir, combo, model)

    if not skip_rebuild:
        print(f"  [1/2] 重建索引（collection: {combo['collection']}）...", end=" ", flush=True)
        t0 = time.time()
        text_hashes = rebuild_processed_and_index(
            chunk_fn=combo["chunk_fn"],
            collection_name=combo["collection"],
            reuse_processed=reuse_processed,
            write_processed=False,
        )
        corpus_sha256 = corpus_digest(text_hashes)
        print(f"完成 ({time.time()-t0:.1f}s)")
    else:
        print("  [1/2] 跳過 rebuild（--skip-rebuild）")
        # 沿用先前建立的索引：只能記錄目前 data/processed 的指紋（index_rebuilt=False 標示此情況）
        corpus_sha256 = corpus_digest(processed_corpus_hashes()) if reuse_processed else None

    chunk_count = int(get_chroma_collection(combo["collection"]).count())
    if chunk_count == 0:
        raise ExperimentError(
            f"collection {combo['collection']!r} 是空的；請不加 --skip-rebuild 重跑。"
        )
    if combo.get("use_hybrid"):
        _require_bm25_index(combo["collection"], chunk_count)

    print(f"  [2/2] 執行 {len(QUESTIONS)} 題{'檢索' if retrieval_only else '問答'}...")
    results: list[dict] = []
    for i, q in enumerate(QUESTIONS):
        print(f"    Q{i+1}: {q[:50]}...", end=" ", flush=True)
        result = _run_query(combo, q, retrieval_only=retrieval_only)
        (combo_dir / f"q{i + 1}.md").write_text(
            format_question_md(i, result, combo, model, retrieval_only=retrieval_only),
            encoding="utf-8",
        )
        print("OK" if not result["error"] else f"WARN({result['error']})")
        results.append(result)

    env = environment if environment is not None else build_environment(
        git=_git_state(), reuse_processed=reuse_processed
    )
    _write_json(
        combo_dir / "results.json",
        build_results_payload(
            combo,
            results,
            model=model,
            retrieval_only=retrieval_only,
            environment={
                **env,
                "corpus_sha256": corpus_sha256,
                "chunk_count": chunk_count,
                "index_rebuilt": not skip_rebuild,
            },
        ),
    )
    failed = question_failures(results)
    return {
        "chunk_count": chunk_count,
        "corpus_sha256": corpus_sha256,
        "status": "degraded" if failed else "ok",
        "failed_questions": failed,
    }


# ─── Summary ──────────────────────────────────────────────────────────────────


def _load_results(out_dir: Path) -> list[dict]:
    """讀取 <out>/combo_*/results.json；無法讀取的檔案在 stderr 警告後略過。依 combo id 排序。"""
    payloads: list[dict] = []
    for path in sorted(out_dir.glob("combo_*/results.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data.get("combo", {}).get("id"), str) or not isinstance(
                data.get("questions"), list
            ):
                raise ValueError("缺少 combo.id 或 questions")
        except (OSError, ValueError, AttributeError) as exc:
            _warn(f"[experiment_runner] 警告：略過無法讀取的 {path.parent.name}/results.json（{type(exc).__name__}）。")
            continue
        payloads.append(data)
    return sorted(payloads, key=lambda d: d["combo"]["id"])


def _question_map(payload: dict) -> dict[int, dict]:
    return {q["index"]: q for q in payload["questions"] if isinstance(q, dict) and "index" in q}


def _status_label(payload: dict) -> str:
    """summary 用：全部題目有效為 ok，否則列出失敗題號（degraded）。"""
    failed = [
        q.get("index")
        for q in payload["questions"]
        if isinstance(q, dict) and q.get("error") not in OK_QUESTION_ERRORS
    ]
    return "ok" if not failed else "degraded（" + "、".join(f"Q{i}" for i in failed) + "）"


def generate_summary(out_dir: Path) -> Path | None:
    """依 <out> 內所有 combo_*/results.json 產生 <out>/summary.md；沒有任何結果時回傳 None。

    out_dir 須符合 --out 規則（不會覆寫第一輪的 experiments/summary.md）。
    """
    resolve_out_dir(out_dir)
    payloads = _load_results(out_dir)
    if not payloads:
        _warn(
            f"[experiment_runner] 警告：{display_path(out_dir)} 內沒有可用的 results.json，未產生 summary"
            "（既有的 summary.md 移到 .previous/）。"
        )
        retire_previous(out_dir / "summary.md")
        return None

    ids = [p["combo"]["id"].upper() for p in payloads]
    q_maps = [_question_map(p) for p in payloads]

    def cfg_row(label: str, fn) -> str:
        return f"| {label} | " + " | ".join(fn(p) for p in payloads) + " |"

    config_table = [
        "| 面向 | " + " | ".join(f"Combo {cid}" for cid in ids) + " |",
        "|---|" + "---|" * len(payloads),
        cfg_row("Chunk 策略", lambda p: str(p["combo"].get("chunk_desc"))),
        cfg_row("top-k", lambda p: str(p["combo"].get("top_k"))),
        cfg_row("max_dist", lambda p: str(p["combo"].get("max_distance"))),
        cfg_row("距離截斷", lambda p: str(p["combo"].get("truncate_to") or "否")),
        cfg_row("Prompt", lambda p: str(p["combo"].get("prompt_style"))),
        cfg_row("Hybrid Search", lambda p: "是" if p["combo"].get("use_hybrid") else "否"),
        cfg_row("Query Expand", lambda p: "是" if p["combo"].get("use_expand") else "否"),
        cfg_row("LLM 模型", lambda p: str(p.get("model"))),
        cfg_row("模式", lambda p: "僅檢索" if p.get("retrieval_only") else "檢索 + 生成"),
        cfg_row("狀態", _status_label),
        cfg_row("產生時間", lambda p: str(p.get("generated_at"))),
    ]

    def all_dists(qm: dict[int, dict]) -> list[float | None]:
        return [c.get("distance_to_question") for q in qm.values() for c in q.get("cites", [])]

    overall = [
        "| Combo | 與原問題平均距離（全部題目的 chunks） |",
        "|---|---|",
    ] + [f"| {cid} | {_fmt(_mean(all_dists(qm)))} |" for cid, qm in zip(ids, q_maps)]

    indices = sorted({i for qm in q_maps for i in qm})
    sections: list[str] = []
    for idx in indices:
        question = next(qm[idx]["question"] for qm in q_maps if idx in qm)
        lines = [
            f"## Q{idx}: {question}",
            "",
            "| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |",
            "|---|---|---|---|",
        ]
        details: list[str] = []
        for cid, payload, qm in zip(ids, payloads, q_maps):
            if idx not in qm:
                continue
            q = qm[idx]
            cites = q.get("cites", [])
            avg = _mean([c.get("distance") for c in cites])
            avg_q = _mean([c.get("distance_to_question") for c in cites])
            lines.append(f"| {cid} | {len(cites)} | {_fmt(avg)} | {_fmt(avg_q)} |")
            refs = ", ".join(
                f"`{c['source_file']}` chunk {c['chunk_index']} ({c.get('retrieved_by', 'dense')})"
                for c in cites[:3]
            ) + ("..." if len(cites) > 3 else "")
            block = [
                f"### Combo {cid} — {payload['combo'].get('name')}",
                f"- 取回 chunks: {len(cites)}，與原問題平均距離: {_fmt(avg_q)}（retriever 平均距離: {_fmt(avg)}）",
            ]
            if q.get("expanded_queries"):
                block.append("- 擴展查詢: " + " / ".join(q["expanded_queries"]))
            block.append(f"- 引用: {refs or '（無）'}")
            if q.get("error"):
                block.append(f"- 錯誤: {q['error']}")
            answer = "（僅檢索，未生成回答）" if payload.get("retrieval_only") else q.get("answer", "")
            block += ["", answer, ""]
            details.append("\n".join(block))
        sections.append("\n".join(lines) + "\n\n" + "\n".join(details))

    content = "\n".join(
        [
            f"# RAG {len(payloads)} 組實驗比較（{'、'.join(ids)}）",
            "",
            f"*生成時間: {_ts()}；來源: {len(payloads)} 個 combo_*/results.json*",
            "",
            "距離為 cosine distance（0=完全相同）。「與原問題距離」一律以原始問題計算；",
            "retriever 回報的距離在 Query Expansion 時可能是相對於子查詢。",
            "",
            "## 組合設定對照",
            "",
            *config_table,
            "",
            "## 與原問題的平均距離",
            "",
            *overall,
            "",
            "---",
            "",
        ]
    ) + "\n\n---\n\n".join(sections)

    path = out_dir / "summary.md"
    path.write_text(content, encoding="utf-8")
    print(f"\n[summary] 已寫入: {display_path(path)}")
    return path


# ─── run.json ─────────────────────────────────────────────────────────────────


def _git_state() -> dict:
    """目前的 git commit 與是否有未提交的追蹤檔變更；git 不可用時在 stderr 警告並回傳 None 值。"""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, capture_output=True, text=True, check=True, timeout=10,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError) as exc:
        _warn(f"[experiment_runner] 警告：無法取得 git 狀態（{type(exc).__name__}）。")
        return {"commit": None, "dirty": None}


def _previous_runs(path: Path) -> list[dict]:
    """讀取既有 run.json，攤平成歷次執行紀錄（分次跑 f、g 時保留先前的執行環境）。"""
    if not path.is_file():
        return []
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(old, dict):
            raise ValueError("run.json 不是 JSON 物件")
    except (OSError, ValueError) as exc:
        _warn(f"[experiment_runner] 警告：既有 run.json 無法讀取（{type(exc).__name__}），不保留先前紀錄。")
        return []
    history = old.get("previous_runs")
    current = {k: v for k, v in old.items() if k != "previous_runs"}
    return [*(history if isinstance(history, list) else []), current]


def build_run_manifest(
    *,
    git: dict,
    model: str,
    combos: dict[str, dict],
    retrieval_only: bool,
    reuse_processed: bool,
    previous_runs: list[dict],
    processed_hashes: Mapping[str, str] | None = None,
    matches_pin: bool | None = None,
) -> dict:
    """run.json 內容：時間、git、LLM 模型與是否走 proxy（不含金鑰/endpoint）、embedding、語料、各 combo。

    corpus.processed_sha256 為執行時 data/processed 各檔的 sha256；matches_pin 為其是否等於
    experiments/corpus_pin.json（--re-extract 時不檢查，為 None）。
    """
    hashes = dict(processed_hashes if processed_hashes is not None else processed_corpus_hashes())
    return {
        "generated_at": _iso_now(),
        "git": git,
        "llm": {"model": model, "proxy_configured": llm_proxy_configured()},
        "embedding": {"model": EMBEDDING_MODEL, "max_seq_length": loaded_embedding_max_seq_length()},
        "corpus": {
            "raw_dir": "data/raw",
            "document_count": len(list_raw_documents()),
            "processed_digest": corpus_digest(hashes),
            "matches_pin": matches_pin,
            "processed_sha256": hashes,
        },
        # True：索引文字取自已提交的 data/processed；False：--re-extract 從 data/raw 重新抽取（只在記憶體中）
        "reuse_processed": reuse_processed,
        "retrieval_only": retrieval_only,
        "combos": combos,
        "questions": list(QUESTIONS),
        "previous_runs": previous_runs,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAG 檢索策略實驗跑分（Combo A–G）。第一輪 A–E 結果位於 experiments/，不會被覆寫。",
    )
    p.add_argument(
        "--combos",
        nargs="+",
        choices=list(COMBO_MAP.keys()),
        default=list(COMBO_MAP.keys()),
        metavar="COMBO_ID",
        help="要跑的 combo（a b c d e f g），預設全部；round 2 請用 --combos e f g",
    )
    p.add_argument(
        "--out",
        default=None,
        help="輸出目錄（預設 experiments/round2，rag_eval.py 的預設讀取位置）；"
        "repo 內須位於 experiments/ 之下，且不可為 experiments/ 本身或其中的 combo_* 目錄",
    )
    source = p.add_mutually_exclusive_group()
    source.add_argument(
        "--skip-rebuild",
        action="store_true",
        help="跳過 rebuild，直接使用既有 collection（適合只改 prompt 時快速重測）",
    )
    p.add_argument(
        "--retrieval-only",
        action="store_true",
        help="只檢索、不生成回答（Combo G 仍會呼叫 LLM 做查詢擴展）",
    )
    source.add_argument(
        "--re-extract",
        action="store_true",
        help="從 data/raw 重新抽取文字建索引（只在記憶體中，不改寫 data/processed 與 ingest manifest）；"
        "預設使用已提交且與 experiments/corpus_pin.json 一致的 data/processed",
    )
    return p.parse_args(argv)


def check_pinned_corpus(hashes: Mapping[str, str]) -> list[str]:
    """比對 data/processed 與 corpus_pin.json，回傳不一致的檔名；鎖定檔無法讀取時拋 OSError / ValueError。"""
    return corpus_mismatches(hashes, load_corpus_pin(CORPUS_PIN_PATH))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        out_dir = resolve_out_dir(args.out if args.out is not None else DEFAULT_OUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
    except (ValueError, OSError) as exc:
        _warn(f"[experiment_runner] 錯誤：{describe_exception(exc)}")
        return 2

    reuse_processed = not args.re_extract
    processed_hashes = processed_corpus_hashes()
    matches_pin: bool | None = None
    if reuse_processed:
        try:
            mismatched = check_pinned_corpus(processed_hashes)
        except (OSError, ValueError) as exc:
            _warn(f"[experiment_runner] 錯誤：無法讀取 {display_path(CORPUS_PIN_PATH)}（{describe_exception(exc)}）。")
            return 2
        if mismatched:
            shown = "、".join(mismatched[:3]) + ("…" if len(mismatched) > 3 else "")
            _warn(
                f"[experiment_runner] 錯誤：data/processed 有 {len(mismatched)} 個檔案與 "
                f"{display_path(CORPUS_PIN_PATH)} 不一致（{shown}）。請用 git 還原 data/processed，"
                "或明確加 --re-extract 從 data/raw 重新抽取。"
            )
            return 2
        matches_pin = True

    git = _git_state()
    model = effective_llm_model()
    environment = build_environment(git=git, reuse_processed=reuse_processed)
    combos_info: dict[str, dict] = {}

    for combo in (COMBO_MAP[cid] for cid in args.combos):
        try:
            info = run_combo(
                combo,
                out_dir,
                model=model,
                skip_rebuild=args.skip_rebuild,
                retrieval_only=args.retrieval_only,
                reuse_processed=reuse_processed,
                environment=environment,
            )
            combos_info[combo["id"]] = {
                "collection": combo["collection"],
                "chunk_count": info["chunk_count"],
                "corpus_sha256": info["corpus_sha256"],
                "status": info["status"],
                "failed_questions": info["failed_questions"],
            }
            if info["status"] != "ok":
                _warn(
                    f"[experiment_runner] 警告：Combo {combo['id'].upper()} 第 "
                    f"{info['failed_questions']} 題失敗，狀態記為 degraded（rag_eval 不計這些題目）。"
                )
        except Exception as exc:
            _warn(f"\n[ERROR] Combo {combo['id'].upper()} 失敗（{describe_exception(exc)}）")
            combos_info[combo["id"]] = {
                "collection": combo["collection"],
                "chunk_count": None,
                "corpus_sha256": None,
                "status": f"failed: {type(exc).__name__}",
                "failed_questions": None,
            }

    run_path = out_dir / "run.json"
    _write_json(
        run_path,
        build_run_manifest(
            git=git,
            model=model,
            combos=combos_info,
            retrieval_only=args.retrieval_only,
            reuse_processed=reuse_processed,
            previous_runs=_previous_runs(run_path),
            processed_hashes=processed_hashes,
            matches_pin=matches_pin,
        ),
    )
    generate_summary(out_dir)

    print(f"\n全部完成。結果存放於 {display_path(out_dir)}")
    return 0 if all(c["status"] == "ok" for c in combos_info.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
