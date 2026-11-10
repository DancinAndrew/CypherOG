#!/usr/bin/env python3
"""Round-2 預先登錄（pre-registered）評估：以內容指標比較 round 1（A–E）與 round 2（E′、F、G）。

背景：round 1 只用平均 cosine distance 比較各組；但在固定 k 下，dense top-k 本來就是
距離總和最小的集合，hybrid（F）或 query expansion（G）只可能持平或拉高平均距離。
因此在 round 2 執行前，先於 experiments/gold.json 固定兩個內容指標：

- Q2（Electric Boogaloos 成員）：回答點名了語料列出的九位成員中的幾位、檢索 context 中出現幾位。
- Q4（Juste Debout）：top-k 中同時提到 Juste Debout 與任一 fact group 的 chunk 數，
  以及回答（須提到 Juste Debout）陳述了幾個 fact group。

另列出「對原始問題的平均 cosine distance（5 題）」作為背景數據。

資料來源：
- Round 1：experiments/combo_a..e/q1..q5.md 的既有記錄（不重跑）；chunk 原文未儲存，
  以各組 round-1 切法重新切 data/processed/<source_file> 還原。還原前先確認 data/processed
  與 experiments/corpus_pin.json 完全一致，否則拒絕評估。Round 1 的索引只收錄 39 份文件
  （experiments/corpus_round1.json）。
- Round 2：--round2 的每個目錄（預設 experiments/round2）都會遞迴搜尋 run 資料夾，也就是直接含
  combo_*/results.json 的目錄（例如 experiments/round2/round1_corpus/run1；略過 .previous/ 等隱藏目錄）。
  results.json 由 experiment_runner.py 寫出，契約見 load_round2_run。同一個 run 資料夾內各組的模型、
  檢索模式、embedding、語料（名稱、文件數、文件清單指紋）、chunk 數、建索引文字的指紋與 git commit
  必須一致，否則拒絕評估；不同 run 資料夾依 (語料, 模型) 分組，列標籤如「E′ (control) · round1 corpus · run 2」，
  並另列各 (語料, 模型, 組別) 跨重複執行的 min–max 穩定度表。
  查詢擴展失敗的題目，檢索與回答指標皆為 n/a；LLM 失敗的題目，回答指標為 n/a。

輸出：<out>/metrics.json 與 <out>/metrics.md。

重現檢查（--check-reproduction <run_dir>）：把 experiment_runner.py --combos a b c d e --retrieval-only
的結果與 round-1 記錄（experiments/combo_*/q*.md）逐題比對（chunk id 與順序相同、距離差 <= 0.001），
印出 5×5 表並寫出報告（預設 experiments/round1_reproduction.md，可用 --report-out 改寫到其他位置）；
有任何不符時結束碼為 1。

Usage:
    python rag_eval.py
    python rag_eval.py --round1 experiments --round2 experiments/round2 --out experiments
    python rag_eval.py --round2 experiments/round2/round1_corpus experiments/round2/current_corpus
    python rag_eval.py --check-reproduction experiments/round1_reproduction_run
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
GOLD_PATH = EXPERIMENTS_DIR / "gold.json"
CORPUS_PIN_PATH = EXPERIMENTS_DIR / "corpus_pin.json"
CORPUS_ROUND1_PATH = EXPERIMENTS_DIR / "corpus_round1.json"
ROUND2_DIR = EXPERIMENTS_DIR / "round2"
PROCESSED_DIR = ROOT / "data" / "processed"
REPRODUCTION_REPORT_PATH = EXPERIMENTS_DIR / "round1_reproduction.md"

QUESTION_COUNT = 5

# Round 1 的 q*.md 未記錄模型名稱；當時 LITELLM_MODEL 預設為 gemini-2.5-flash，經課程提供的代理呼叫。
ROUND1_MODEL = "gemini-2.5-flash (course OpenAI-compatible proxy)"

# 預設 embedding 模型的 max_seq_length：chunk 超過 128 個 token 的部分不進入向量（BM25 則索引全文）
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MAX_TOKENS = 128

# 重現檢查的距離容許誤差：round-1 的 q*.md 只記錄到小數點後 3 位
REPRODUCTION_TOLERANCE = 0.001

# 語料的顯示順序（round1 為主要比較對象）
CORPUS_ORDER: tuple[str, ...] = ("round1", "current")

# Round-1 各組設定（刻意在此定義，不依賴 experiment_runner.py 的現行設定）。
# chunker: (種類, size, overlap)；"naive" → naive_chunk_text(t, size)，"recursive" → chunk_text(t, size, overlap)。
# truncate_to: D 記錄為「top-10 + rerank」，實際只把依距離排序的 top-10 截成 5 筆（沒有 reranker）。
ROUND1_COMBOS: dict[str, dict[str, Any]] = {
    "a": {
        "chunker": ("naive", 500, 0),
        "chunk_desc": "naive / size=500 / overlap=0",
        "top_k": 5,
        "max_distance": 1.0,
        "truncate_to": None,
        "prompt_style": "weak",
        "use_hybrid": False,
        "use_expand": False,
    },
    "b": {
        "chunker": ("recursive", 600, 100),
        "chunk_desc": "recursive / size=600 / overlap=100",
        "top_k": 5,
        "max_distance": 0.65,
        "truncate_to": None,
        "prompt_style": "weak",
        "use_hybrid": False,
        "use_expand": False,
    },
    "c": {
        "chunker": ("recursive", 1200, 200),
        "chunk_desc": "recursive / size=1200 / overlap=200",
        "top_k": 3,
        "max_distance": 0.65,
        "truncate_to": None,
        "prompt_style": "strong",
        "use_hybrid": False,
        "use_expand": False,
    },
    "d": {
        "chunker": ("recursive", 300, 50),
        "chunk_desc": "recursive / size=300 / overlap=50",
        "top_k": 10,
        "max_distance": 0.5,
        "truncate_to": 5,
        "prompt_style": "strong",
        "use_hybrid": False,
        "use_expand": False,
    },
    "e": {
        "chunker": ("recursive", 600, 100),
        "chunk_desc": "recursive / size=600 / overlap=100",
        "top_k": 5,
        "max_distance": 0.65,
        "truncate_to": None,
        "prompt_style": "cot",
        "use_hybrid": False,
        "use_expand": False,
    },
}

# Round 2 的 E 與 round 1 的 E 設定相同但模型與索引不同，標為對照組。
ROUND2_LABELS: dict[str, str] = {"e": "E\u2032 (control)"}

# 預先登錄的 round-2 組別；<round2> 內其他 combo 的結果不納入比較
ROUND2_COMBO_IDS: tuple[str, ...] = ("e", "f", "g")

# 同一個 run 資料夾內各組必須相同的執行條件（results.json 的 model、retrieval_only 與 environment 欄位）；
# corpus、document_count、documents_digest 是 run 身分的一部分
ROUND2_ENVIRONMENT_KEYS: tuple[str, ...] = (
    "embedding_model",
    "corpus",
    "document_count",
    "documents_digest",
    "chunk_count",
    "text_source",
    "corpus_sha256",
    "git_commit",
)
ROUND2_CONSISTENCY_KEYS: tuple[str, ...] = ("model", "retrieval_only", *ROUND2_ENVIRONMENT_KEYS)

# 視為「有回答可評」的錯誤碼：no_context 時系統回固定訊息，回答指標記 0 而非 n/a。
_SCOREABLE_ERRORS: frozenset[str | None] = frozenset({None, "no_context"})

TABLE_HEADER = (
    "| Run | Round | Retrieval | Mean cosine distance to question (5 Qs) | "
    "Q2 members named (/9) | Q2 members in context (/9) | "
    "Q4 Juste Debout chunks in top-k | Q4 fact groups in answer |"
)


# ─── 正規化與別名比對 ─────────────────────────────────────────────────────────

_QUOTE_TABLE = str.maketrans({"\u2019": "'", "\u2018": "'"})


def normalize(text: str) -> str:
    """把彎引號（U+2019、U+2018）換成直引號，並把連續空白折疊成單一空格（含去頭尾）。"""
    return " ".join(text.translate(_QUOTE_TABLE).split())


@functools.lru_cache(maxsize=None)
def _alias_regex(alias: str) -> re.Pattern[str]:
    """別名 → 不分大小寫、前後不可緊鄰 ASCII 英文字母的 regex。

    不用 \\b：Python 的 \\b 把中日韓文字視為 word 字元，「成員Pete」會比對失敗。
    """
    return re.compile(
        r"(?<![A-Za-z])" + re.escape(normalize(alias)) + r"(?![A-Za-z])",
        re.IGNORECASE,
    )


def _mentions_any(normalized_text: str, aliases: Iterable[str]) -> bool:
    """normalized_text 是否命中任一別名（呼叫端須先 normalize）。"""
    return any(_alias_regex(a).search(normalized_text) for a in aliases)


def members_named(text: str, gold: Mapping[str, Any]) -> list[str]:
    """回傳 text 中點名到的 Q2 成員（gold 的標準名稱，依字母排序）。gold 為 load_gold() 的結果。"""
    norm = normalize(text)
    members: Mapping[str, Sequence[str]] = gold["q2_members"]["members"]
    return sorted(name for name, aliases in members.items() if _mentions_any(norm, aliases))


def juste_debout_fact_groups(text: str, gold: Mapping[str, Any]) -> list[str]:
    """回傳 text 陳述的 Q4 fact group 名稱（依 gold 順序）；text 未提到 Juste Debout 時回空 list。"""
    q4 = gold["q4_juste_debout"]
    norm = normalize(text)
    if not _mentions_any(norm, q4["entity"]):
        return []
    groups: Mapping[str, Sequence[str]] = q4["fact_groups"]
    return [name for name, aliases in groups.items() if _mentions_any(norm, aliases)]


def gold_chunk_count(chunk_texts: Iterable[str], gold: Mapping[str, Any]) -> int:
    """計算同時提到 Juste Debout 與至少一個 fact group 的 chunk 數。"""
    return sum(1 for t in chunk_texts if juste_debout_fact_groups(t, gold))


def members_in_chunks(chunk_texts: Iterable[str], gold: Mapping[str, Any]) -> list[str]:
    """逐 chunk 比對成員後取聯集（不把 chunk 串接，避免跨 chunk 邊界誤配）。"""
    found = {name for t in chunk_texts for name in members_named(t, gold)}
    return sorted(found)


def mean_distance(distance_lists: Sequence[Sequence[float]]) -> tuple[float | None, int]:
    """先算每題平均距離，再對「有檢索結果的題目」取平均。回傳 (平均值或 None, 納入題數)。"""
    per_question = [sum(ds) / len(ds) for ds in distance_lists if ds]
    if not per_question:
        return None, 0
    return sum(per_question) / len(per_question), len(per_question)


def load_gold(path: Path = GOLD_PATH) -> dict[str, Any]:
    """讀取並檢查 gold.json（預先登錄的成員別名與 Juste Debout fact groups）。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        q2 = data["q2_members"]
        q4 = data["q4_juste_debout"]
        int(q2["question_number"])
        int(q4["question_number"])
        non_empty = bool(q2["members"]) and bool(q4["entity"]) and bool(q4["fact_groups"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"gold.json 缺少必要欄位（{type(exc).__name__}: {exc}）") from exc
    if not non_empty:
        raise ValueError("gold.json 的 members / entity / fact_groups 不可為空")
    return data


# ─── Round 1：解析 q*.md 與還原 chunk ────────────────────────────────────────

_QUESTION_RE = re.compile(r"^\s*#\s+Q\d+:\s*(?P<q>.+?)\s*$")
_CITE_RE = re.compile(
    r"^\s*\[\d+\]\s+`(?P<src>[^`]+)`\s+chunk\s+(?P<idx>\d+)"
    r"(?:\s+\[(?P<section>.*)\])?"
    r"\s+\(距離:\s*(?P<dist>\d*\.?\d+)\)\s*$"
)
_RESULTS_COUNT_RE = re.compile(r"##\s+檢索結果（(\d+) 個 chunks")
_ANSWER_HEADING_RE = re.compile(r"^\s*##\s+LLM 回答\s*$")
_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)


def parse_round1_question(path: Path) -> dict[str, Any]:
    """解析 round-1 的 q{N}.md。

    回傳 dict：
      question       — 標題列「# Q{N}: ...」的問題文字
      cites          — [(source_file, chunk_index, distance)]，依檔案中的順序
      section_labels — 與 cites 對齊的 section 標籤（無標籤為 ""）
      answer         — 「## LLM 回答」之後、「*generated by」前最後一條 --- 之前的文字
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    answer_start = next((i for i, ln in enumerate(lines) if _ANSWER_HEADING_RE.match(ln)), None)
    if answer_start is None:
        raise ValueError(f"{path.name}: 找不到「## LLM 回答」標題")

    question = next(
        (m.group("q") for ln in lines[:answer_start] if (m := _QUESTION_RE.match(ln))),
        None,
    )
    if question is None:
        raise ValueError(f"{path.name}: 找不到「# Q{{N}}: ...」問題標題")

    matches = [m for ln in lines[:answer_start] if (m := _CITE_RE.match(ln))]
    cites = [(m.group("src"), int(m.group("idx")), float(m.group("dist"))) for m in matches]
    labels = [m.group("section") or "" for m in matches]
    recorded = next((int(m.group(1)) for ln in lines[:answer_start] if (m := _RESULTS_COUNT_RE.search(ln))), None)
    if recorded is not None and recorded != len(cites):
        raise ValueError(f"{path.name}: 解析到 {len(cites)} 個 cite，但檢索結果標題記錄 {recorded} 個")

    generated = [i for i in range(answer_start + 1, len(lines)) if lines[i].strip().startswith("*generated by")]
    answer_end = len(lines)
    if generated:
        rules = [i for i in range(answer_start + 1, generated[-1]) if lines[i].strip() == "---"]
        answer_end = rules[-1] if rules else generated[-1]
    answer = "\n".join(lines[answer_start + 1 : answer_end]).strip()

    return {"question": question, "cites": cites, "section_labels": labels, "answer": answer}


def round1_chunker(combo_id: str) -> Callable[[str], list[str]]:
    """回傳該組 round-1 使用的切割函式（函式本體取自 data_update）。"""
    spec = ROUND1_COMBOS.get(combo_id)
    if spec is None:
        raise ValueError(f"未定義 round-1 combo: {combo_id!r}")
    # 延遲匯入：純函數（比對、解析）不必載入 data_update / rag_common
    from data_update import chunk_text, naive_chunk_text

    kind, size, overlap = spec["chunker"]
    if kind == "naive":
        return lambda t: naive_chunk_text(t, size)
    return lambda t: chunk_text(t, size, overlap)


def section_title_for(text: str, chunk: str) -> str:
    """依 round-1 data_update 的規則重算 chunk 的 section 標籤：chunk 起點前最近的 #～### 標題。"""
    offset = text.find(chunk[:40]) if len(chunk) >= 40 else text.find(chunk)
    headings = _HEADING_RE.findall(text[:offset] if offset > 0 else "")
    return headings[-1].strip() if headings else ""


def reconstruct_round1_chunks(
    cites: Sequence[tuple[str, int, float]],
    combo_id: str,
    processed_dir: Path = PROCESSED_DIR,
) -> list[dict[str, Any]]:
    """以該組 round-1 切法重切 data/processed/<source_file>，取回每個 cite 的 chunk。

    回傳 [{source_file, chunk_index, text, section_title}]；section_title 為重算值，
    可與 q*.md 記錄的標籤比對，檢查還原是否一致。
    """
    chunker = round1_chunker(combo_id)
    base = processed_dir.resolve()
    cache: dict[str, tuple[str, list[str]]] = {}
    out: list[dict[str, Any]] = []
    for source_file, chunk_index, _distance in cites:
        path = (base / source_file).resolve()
        if path.parent != base:
            raise ValueError(f"source_file 不在 processed 目錄內: {source_file!r}")
        if source_file not in cache:
            text = path.read_text(encoding="utf-8")
            cache[source_file] = (text, chunker(text))
        text, chunks = cache[source_file]
        if not 0 <= chunk_index < len(chunks):
            raise ValueError(
                f"{source_file}: chunk {chunk_index} 超出範圍（round-1 combo {combo_id} 只切出 {len(chunks)} 塊）"
            )
        chunk = chunks[chunk_index]
        out.append(
            {
                "source_file": source_file,
                "chunk_index": chunk_index,
                "text": chunk,
                "section_title": section_title_for(text, chunk),
            }
        )
    return out


def describe_retrieval(combo: Mapping[str, Any]) -> str:
    """把 combo 設定轉成表格用的檢索描述。

    F 的距離門檻只作用於向量通道（只被 BM25 找到的 chunk 不受限）；G 的門檻是對各子查詢的距離套用。
    """
    mode = "dense + BM25 (RRF)" if combo.get("use_hybrid") else "dense"
    if combo.get("use_expand"):
        mode += " + query expansion"
    k = f"top-{combo['top_k']}"
    truncate_to = combo.get("truncate_to")
    if truncate_to is not None and truncate_to < combo["top_k"]:
        k += f" cut to {truncate_to}"
    scopes = [
        scope
        for flag, scope in (("use_hybrid", "dense channel only"), ("use_expand", "per sub-query"))
        if combo.get(flag)
    ]
    cutoff = f"max distance {combo['max_distance']}" + (f" ({', '.join(scopes)})" if scopes else "")
    return f"{combo.get('chunk_desc', '?')}; {mode} {k}; {cutoff}"


def _display_path(path: Path) -> str:
    """輸出用路徑：repo 內顯示相對路徑，repo 外只顯示最後兩層（不洩漏本機絕對路徑）。"""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return "/".join(resolved.parts[-2:])


def _warn(message: str) -> None:
    print(f"[rag_eval] 警告: {message}", file=sys.stderr)


def _describe_exc(exc: BaseException) -> str:
    """例外 → 「型別: 說明」，不含本機絕對路徑（OSError 只顯示檔名的相對/末兩層路徑）。"""
    from rag_common import describe_exception  # 延遲匯入：純函數不必載入 rag_common

    return describe_exception(exc)


def check_round1_corpus(processed_dir: Path, pin_path: Path) -> str:
    """確認 processed_dir 與 corpus_pin.json 完全一致，回傳鎖定的語料 digest；不一致時拋 ValueError。

    round-1 的 chunk 原文是重切 data/processed 還原的；文字一變（例如 data_update.py --rebuild 以新版
    pypdf 重新抽取），還原出的 chunk 就不是當時的內容，而 section 標籤檢查不一定看得出來。
    """
    from rag_common import corpus_digest, corpus_mismatches, load_corpus_pin, processed_corpus_hashes

    pinned = load_corpus_pin(pin_path)
    bad = corpus_mismatches(processed_corpus_hashes(processed_dir), pinned)
    if bad:
        shown = ", ".join(bad[:3]) + (", …" if len(bad) > 3 else "")
        raise ValueError(
            f"{_display_path(processed_dir)} 有 {len(bad)} 個檔案與 {_display_path(pin_path)} 不一致（{shown}）；"
            "round-1 chunk 無法可靠還原，請用 git 還原 data/processed"
        )
    return corpus_digest(pinned)


def load_round1_run(
    combo_dir: Path,
    combo_id: str,
    processed_dir: Path = PROCESSED_DIR,
) -> dict[str, Any]:
    """讀取一組 round-1 的 q1..q5.md，轉成 compute_metrics 的共同輸入格式。"""
    spec = ROUND1_COMBOS[combo_id]
    questions: list[dict[str, Any]] = []
    checked = matched = 0
    for n in range(1, QUESTION_COUNT + 1):
        parsed = parse_round1_question(combo_dir / f"q{n}.md")
        chunks = reconstruct_round1_chunks(parsed["cites"], combo_id, processed_dir)
        for chunk, label in zip(chunks, parsed["section_labels"]):
            checked += 1
            if chunk["section_title"] == label:
                matched += 1
            else:
                _warn(
                    f"combo_{combo_id} q{n} `{chunk['source_file']}` chunk {chunk['chunk_index']}: "
                    f"重建 chunk 的 section 標籤 {chunk['section_title']!r} 與記錄 {label!r} 不符"
                )
        questions.append(
            {
                "index": n,
                "question": parsed["question"],
                "distances": [d for _, _, d in parsed["cites"]],
                "chunk_texts": [c["text"] for c in chunks],
                "answer": parsed["answer"],
                "error": None,
            }
        )
    return {
        "run": combo_id.upper(),
        "combo_label": combo_id.upper(),
        "round": 1,
        "combo_id": combo_id,
        "corpus": "round1",
        "repeat": None,
        "run_folder": None,
        "order": 0,
        "retrieval": describe_retrieval(spec),
        "max_distance": float(spec["max_distance"]),
        "prompt_style": spec["prompt_style"],
        "model": ROUND1_MODEL,
        "source": _display_path(combo_dir),
        "retrieval_only": False,
        "questions": questions,
        "reconstruction": {"cites": checked, "section_label_matches": matched},
        "consistency": None,
    }


def load_round1_runs(round1_dir: Path, processed_dir: Path = PROCESSED_DIR) -> list[dict[str, Any]]:
    """讀取 round1_dir/combo_a..e；缺少或格式錯誤的組別在 stderr 警告後略過。"""
    runs: list[dict[str, Any]] = []
    for combo_id in ROUND1_COMBOS:
        combo_dir = round1_dir / f"combo_{combo_id}"
        if not combo_dir.is_dir():
            _warn(f"找不到 round-1 目錄 {_display_path(combo_dir)}，略過")
            continue
        try:
            runs.append(load_round1_run(combo_dir, combo_id, processed_dir))
        except (OSError, ValueError) as exc:
            _warn(f"round-1 combo_{combo_id} 無法評估（{_describe_exc(exc)}），略過")
    return runs


# ─── Round 2：讀取 results.json ─────────────────────────────────────────────


def _as_distance(value: Any) -> float:
    """把 JSON 數值轉成 float；非數值或 NaN 視為契約錯誤。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"distance_to_question 不是數值: {value!r}")
    result = float(value)
    if math.isnan(result):
        raise ValueError("distance_to_question 為 NaN")
    return result


def retrieval_ran(error: Any) -> bool:
    """該題的檢索是否照 combo 設定執行：None、no_context、llm_error 為是；expansion_failed 等其他錯誤為否。"""
    return error is None or error == "no_context" or str(error).startswith("llm_error")


def _round2_question(raw: Mapping[str, Any], retrieval_only: bool) -> dict[str, Any]:
    """把 results.json 的單題轉成共同輸入格式。

    沒有可評回答時 answer 為 None；檢索未照設定執行（例如 expansion_failed）時 distances 為空、
    chunk_texts 為 None，該題的檢索指標記為 n/a，也不納入平均距離。
    """
    cites = raw["cites"]
    error = raw.get("error")
    distances = [_as_distance(c["distance_to_question"]) for c in cites]
    texts = [str(c["text"]) for c in cites]
    ran = retrieval_ran(error)
    scoreable = ran and not retrieval_only and error in _SCOREABLE_ERRORS
    return {
        "index": int(raw["index"]),
        "question": str(raw["question"]),
        "distances": distances if ran else [],
        "chunk_texts": texts if ran else None,
        "answer": str(raw.get("answer") or "") if scoreable else None,
        "error": error,
    }


def load_round2_run(results_path: Path) -> dict[str, Any]:
    """讀取 <run 資料夾>/combo_<id>/results.json（experiment_runner.py 寫出）。

    契約重點：combo{id, chunk_desc, top_k, max_distance, truncate_to, prompt_style, use_hybrid,
    use_expand, ...}、model、retrieval_only、environment{embedding_model, corpus, document_count,
    documents_digest, chunk_count, text_source, corpus_sha256, git_commit, ...}、questions[{index, question,
    cites[{text, distance_to_question, ...}], answer, error}]。distance_to_question 為「原始問題」與 chunk 的
    cosine distance（G 的 distance 可能是對子查詢量的，故不使用）。不符契約時拋 ValueError。
    回傳的 run 標籤只有組別名稱（例如 E′ (control)）；跨 run 資料夾的標籤由 load_round2_folders 加上。
    """
    data = json.loads(results_path.read_text(encoding="utf-8"))
    try:
        combo = data["combo"]
        combo_id = str(combo["id"]).lower()
        retrieval_only = bool(data.get("retrieval_only", False))
        questions = [_round2_question(q, retrieval_only) for q in data["questions"]]
        retrieval = describe_retrieval(combo)
        max_distance = float(combo["max_distance"])
        prompt_style = str(combo.get("prompt_style", "?"))
        model = str(data["model"])
        environment = data["environment"]
        consistency = {
            "model": model,
            "retrieval_only": retrieval_only,
            **{key: environment[key] for key in ROUND2_ENVIRONMENT_KEYS},
        }
        corpus = str(environment["corpus"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{_display_path(results_path)} 不符合 results.json 契約（{type(exc).__name__}: {exc}）"
        ) from exc
    run_dir = results_path.parent.parent
    label = ROUND2_LABELS.get(combo_id, combo_id.upper())
    return {
        "run": label,
        "combo_label": label,
        "round": 2,
        "combo_id": combo_id,
        "corpus": corpus,
        "repeat": None,
        "run_folder": _display_path(run_dir),
        "run_dir": run_dir.resolve(),
        "order": 0,
        "retrieval": retrieval,
        "max_distance": max_distance,
        "prompt_style": prompt_style,
        "model": model,
        "source": _display_path(results_path),
        "retrieval_only": retrieval_only,
        "questions": questions,
        "reconstruction": None,
        "consistency": consistency,
    }


def round2_results_paths(round2_dir: Path) -> list[Path]:
    """列出單一 run 資料夾的 round2_dir/combo_*/results.json（不遞迴；目錄不存在時回空 list）。"""
    if not round2_dir.is_dir():
        return []
    return sorted(round2_dir.glob("combo_*/results.json"))


def find_run_dirs(roots: Sequence[Path]) -> list[Path]:
    """在每個 root 之下（含 root 本身）遞迴找出 run 資料夾：直接含 combo_*/results.json 的目錄。

    略過隱藏目錄（例如 experiment_runner 把舊結果移入的 .previous/）；不存在的 root 略過。
    回傳去重、排序後的絕對路徑。
    """
    found: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        base = root.resolve()
        for path in base.rglob("results.json"):
            rel_parts = path.relative_to(base).parts
            if len(rel_parts) < 2 or not rel_parts[-2].startswith("combo_"):
                continue
            if any(part.startswith(".") for part in rel_parts):
                continue
            found.add(path.parent.parent)
    return sorted(found)


def load_round2_runs(round2_dir: Path) -> list[dict[str, Any]]:
    """讀取單一 run 資料夾內所有 round-2 results.json；格式錯誤的檔案在 stderr 警告後略過。"""
    runs: list[dict[str, Any]] = []
    for path in round2_results_paths(round2_dir):
        try:
            runs.append(load_round2_run(path))
        except (OSError, ValueError) as exc:
            _warn(f"{_display_path(path)} 無法評估（{_describe_exc(exc)}），略過")
    return runs


def keep_preregistered_round2(runs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """只保留預先登錄的 round-2 組別（E′、F、G）；其餘在 stderr 警告後略過。"""
    kept: list[Mapping[str, Any]] = []
    for run in runs:
        if run["combo_id"] not in ROUND2_COMBO_IDS:
            _warn(f"{run['source']} 不在預先登錄的 round-2 組別（E′、F、G）之內，不納入比較")
            continue
        kept.append(run)
    return kept


def round2_inconsistencies(runs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """回傳 round-2 各 run 之間取值不同的執行條件：{欄位: {run 名稱: 值}}；空 dict 代表可互相比較。"""
    conflicts: dict[str, dict[str, Any]] = {}
    for key in ROUND2_CONSISTENCY_KEYS:
        values = {run["run"]: run["consistency"][key] for run in runs}
        if len({json.dumps(v, sort_keys=True) for v in values.values()}) > 1:
            conflicts[key] = values
    return conflicts


def question_mismatches(run: Mapping[str, Any], reference: Mapping[int, str]) -> list[int]:
    """回傳 run 中問題文字與 reference（round-1 記錄的題目）不同或缺漏的題號。

    指標綁定題號（Q2、Q4）與五題平均距離；題目順序或文字一變，數字就不可比。
    """
    by_index = {q["index"]: q["question"] for q in run["questions"]}
    return [
        n
        for n, text in sorted(reference.items())
        if normalize(str(by_index.get(n, ""))) != normalize(text)
    ]


def keep_consistent_runs(
    runs: Sequence[Mapping[str, Any]],
    reference: Mapping[int, str],
) -> list[Mapping[str, Any]]:
    """只保留題目與 reference 一致的 run；其餘在 stderr 警告後略過（reference 為空時不檢查）。"""
    kept: list[Mapping[str, Any]] = []
    for run in runs:
        bad = question_mismatches(run, reference) if reference else []
        if bad:
            _warn(f"{run['source']} 的題目 {bad} 與 round-1 記錄不同，不納入比較")
            continue
        kept.append(run)
    return kept


# ─── Round 2：多個 run 資料夾（重複執行）────────────────────────────────────

_RUN_NAME_RE = re.compile(r"run[_-]?(\d+)", re.IGNORECASE)


def _corpus_rank(corpus: str) -> tuple[int, str]:
    """語料的排序鍵：round1、current 在前，其餘依名稱。"""
    return (CORPUS_ORDER.index(corpus) if corpus in CORPUS_ORDER else len(CORPUS_ORDER), corpus)


def _repeat_name(run_dir: Path) -> tuple[str, tuple[int, int, str]]:
    """run 資料夾 → (顯示名稱, 排序鍵)：run2 → ("run 2", (0, 2, ""))；其他名稱原樣保留、排在編號之後。"""
    match = _RUN_NAME_RE.fullmatch(run_dir.name)
    if match:
        n = int(match.group(1))
        return f"run {n}", (0, n, "")
    return run_dir.name, (1, 0, run_dir.name)


def pinned_corpus_digests(pin_path: Path = CORPUS_PIN_PATH) -> dict[str, str | None]:
    """各語料在 corpus_pin.json 下應有的文字指紋：{"round1": …, "current": …}（文件未全被鎖定時為 None）。

    與 results.json 的 corpus_sha256 比較，可確認 round 2 建索引用的就是已鎖定的文字。
    """
    from data_update import list_raw_documents  # 延遲匯入：純函數不必載入 data_update
    from rag_common import load_corpus_definition, load_corpus_pin, pinned_digest_for

    pinned = load_corpus_pin(pin_path)
    round1 = load_corpus_definition(CORPUS_ROUND1_PATH)["documents"]
    current = [p.name for p in list_raw_documents()]
    return {"round1": pinned_digest_for(round1, pinned), "current": pinned_digest_for(current, pinned)}


def label_round2_folders(
    loaded: Sequence[tuple[Path, Sequence[Mapping[str, Any]]]],
    pinned_digests: Mapping[str, str | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """為各 run 資料夾的 run 加上「組別 · 語料 corpus · run N」標籤，並依 (語料, 模型, 重複編號) 排序。

    同一語料有多個模型時，標籤另含模型名稱；(語料, 模型, 名稱) 重複的資料夾依路徑加上 #1、#2。
    回傳 (新的 run list, 資料夾摘要 list)；不修改輸入。
    """
    entries = []
    for run_dir, runs in loaded:
        shared = runs[0]["consistency"]
        repeat, repeat_key = _repeat_name(run_dir)
        entries.append(
            {"run_dir": run_dir, "runs": runs, "shared": shared, "corpus": str(shared["corpus"]),
             "model": str(shared["model"]), "repeat": repeat, "repeat_key": repeat_key}
        )
    entries.sort(key=lambda e: (_corpus_rank(e["corpus"]), e["model"], e["repeat_key"], str(e["run_dir"])))

    models: dict[str, set[str]] = {}
    names: dict[tuple[str, str, str], list[Path]] = {}
    for e in entries:
        models.setdefault(e["corpus"], set()).add(e["model"])
        names.setdefault((e["corpus"], e["model"], e["repeat"]), []).append(e["run_dir"])

    labelled: list[dict[str, Any]] = []
    folders: list[dict[str, Any]] = []
    for order, e in enumerate(entries, start=1):
        same_name = names[(e["corpus"], e["model"], e["repeat"])]
        repeat = e["repeat"] if len(same_name) == 1 else f"{e['repeat']} #{same_name.index(e['run_dir']) + 1}"
        model_part = f" · {e['model']}" if len(models[e["corpus"]]) > 1 else ""
        suffix = f"{e['corpus']} corpus{model_part} · {repeat}"
        expected = pinned_digests.get(e["corpus"])
        folders.append(
            {
                "folder": _display_path(e["run_dir"]),
                "label": suffix,
                "corpus": e["corpus"],
                "model": e["model"],
                "repeat": repeat,
                "combos": sorted(run["combo_id"] for run in e["runs"]),
                "shared": dict(e["shared"]),
                "pinned_text": None if expected is None else e["shared"]["corpus_sha256"] == expected,
            }
        )
        labelled += [
            {**run, "run": f"{run['combo_label']} · {suffix}", "repeat": repeat, "order": order}
            for run in e["runs"]
        ]
    return labelled, folders


def load_round2_folders(
    roots: Sequence[Path],
    reference: Mapping[int, str],
    pinned_digests: Mapping[str, str | None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """讀取 roots 之下所有 run 資料夾的 E′、F、G，回傳 (加上標籤的 run list, 資料夾摘要 list)。

    每個資料夾只保留預先登錄的組別與題目一致的 run；同一個資料夾內執行條件（含語料）不一致時
    拋 ValueError（訊息指出資料夾），不同資料夾之間不要求一致，依 (語料, 模型) 分組。
    pinned_digests 為 None 時以 pinned_corpus_digests() 計算。
    """
    loaded: list[tuple[Path, list[Mapping[str, Any]]]] = []
    for run_dir in find_run_dirs(roots):
        runs = keep_consistent_runs(keep_preregistered_round2(load_round2_runs(run_dir)), reference)
        if not runs:
            _warn(f"{_display_path(run_dir)} 沒有可評估的 round-2 結果，略過")
            continue
        conflicts = round2_inconsistencies(runs)
        if conflicts:
            detail = "; ".join(f"{key}: {values}" for key, values in conflicts.items())
            raise ValueError(
                f"{_display_path(run_dir)} 內各組的執行條件不一致，不可互相比較（{detail}）"
            )
        loaded.append((run_dir, runs))
    digests = pinned_digests if pinned_digests is not None else (pinned_corpus_digests() if loaded else {})
    return label_round2_folders(loaded, digests)


# ─── 計分 ──────────────────────────────────────────────────────────────────


def _score_run(run: Mapping[str, Any], gold: Mapping[str, Any]) -> dict[str, Any]:
    """單一 run → 一列指標（純函數）。"""
    by_index = {q["index"]: q for q in run["questions"]}
    q2 = by_index.get(gold["q2_members"]["question_number"])
    q4 = by_index.get(gold["q4_juste_debout"]["question_number"])
    mean, n_questions = mean_distance([q["distances"] for q in run["questions"]])

    named = members_named(q2["answer"], gold) if q2 and q2["answer"] is not None else None
    q2_chunks = q2["chunk_texts"] if q2 else None
    q4_chunks = q4["chunk_texts"] if q4 else None
    in_context = members_in_chunks(q2_chunks, gold) if q2_chunks is not None else None
    facts = juste_debout_fact_groups(q4["answer"], gold) if q4 and q4["answer"] is not None else None
    all_distances = [d for q in run["questions"] for d in q["distances"]]

    return {
        "run": run["run"],
        "combo_label": run.get("combo_label", run["run"]),
        "round": run["round"],
        "combo_id": run["combo_id"],
        "corpus": run.get("corpus"),
        "repeat": run.get("repeat"),
        "run_folder": run.get("run_folder"),
        "retrieval": run["retrieval"],
        "prompt_style": run["prompt_style"],
        "model": run["model"],
        "source": run["source"],
        "retrieval_only": run["retrieval_only"],
        "mean_distance_to_question": None if mean is None else round(mean, 6),
        "questions_with_cites": n_questions,
        "q2_members_named": None if named is None else len(named),
        "q2_members_named_list": named,
        "q2_members_in_context": None if in_context is None else len(in_context),
        "q2_members_in_context_list": in_context,
        "q4_gold_chunks": gold_chunk_count(q4_chunks, gold) if q4_chunks is not None else None,
        "q4_chunks_retrieved": len(q4_chunks) if q4_chunks is not None else None,
        "q4_fact_groups": None if facts is None else len(facts),
        "q4_fact_groups_list": facts,
        # 與原始問題的距離超過 max_distance 的 cite 數（F 的 BM25 通道、G 的子查詢可能越過門檻）
        "cites_over_max_distance": sum(1 for d in all_distances if d > run["max_distance"]),
        "cites_scored": len(all_distances),
        "failed_questions": [
            {"index": q["index"], "error": q["error"]}
            for q in run["questions"]
            if q["error"] not in _SCOREABLE_ERRORS
        ],
        "reconstruction": run.get("reconstruction"),
    }


def compute_metrics(runs: Sequence[Mapping[str, Any]], gold: Mapping[str, Any]) -> list[dict[str, Any]]:
    """所有 run → 指標列，依 (round, run 資料夾順序, combo_id) 排序（round 2 依語料、模型、重複編號分組）。"""
    ordered = sorted(runs, key=lambda r: (r["round"], r.get("order", 0), r["combo_id"]))
    return [_score_run(run, gold) for run in ordered]


# 穩定度表的指標（metrics row 的欄位, 表頭）
STABILITY_METRICS: tuple[tuple[str, str], ...] = (
    ("mean_distance_to_question", "Mean cosine distance to question (5 Qs)"),
    ("q2_members_named", "Q2 members named (/9)"),
    ("q2_members_in_context", "Q2 members in context (/9)"),
    ("q4_gold_chunks", "Q4 Juste Debout chunks in top-k"),
    ("q4_fact_groups", "Q4 fact groups in answer"),
)


def stability_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """round-2 每個 (語料, 模型, 組別) 跨重複執行的穩定度：各指標的 min、max 與有值的 run 數 n。

    runs 為該組的 run 資料夾數；n < runs 代表部分 run 的該指標為 n/a（不納入 min–max）。
    依 (語料, 模型, combo_id) 排序。純函數。
    """
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["round"] == 2:
            groups.setdefault((str(row["corpus"]), str(row["model"]), row["combo_id"]), []).append(row)

    def summary(members: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
        values = [m[key] for m in members if m[key] is not None]
        return {"min": min(values) if values else None, "max": max(values) if values else None, "n": len(values)}

    ordered = sorted(groups, key=lambda k: (_corpus_rank(k[0]), k[1], k[2]))
    return [
        {
            "corpus": corpus,
            "model": model,
            "combo_id": combo_id,
            "combo_label": groups[(corpus, model, combo_id)][0]["combo_label"],
            "runs": len(groups[(corpus, model, combo_id)]),
            "repeats": [m["repeat"] for m in groups[(corpus, model, combo_id)]],
            "metrics": {key: summary(groups[(corpus, model, combo_id)], key) for key, _ in STABILITY_METRICS},
        }
        for corpus, model, combo_id in ordered
    ]


# ─── 輸出 ──────────────────────────────────────────────────────────────────


def _cell(value: Any) -> str:
    return "n/a" if value is None else str(value).replace("|", "\\|")


def _format_row(row: Mapping[str, Any]) -> str:
    mean = row["mean_distance_to_question"]
    if mean is None:
        mean_cell = "n/a"
    elif row["questions_with_cites"] == QUESTION_COUNT:
        mean_cell = f"{mean:.3f}"
    else:
        mean_cell = f"{mean:.3f} ({row['questions_with_cites']} Qs)"
    q4_chunks = (
        "n/a"
        if row["q4_gold_chunks"] is None
        else f"{row['q4_gold_chunks']}/{row['q4_chunks_retrieved']}"
    )
    cells = [
        _cell(row["run"]),
        _cell(row["round"]),
        _cell(row["retrieval"]),
        mean_cell,
        _cell(row["q2_members_named"]),
        _cell(row["q2_members_in_context"]),
        q4_chunks,
        _cell(row["q4_fact_groups"]),
    ]
    return "| " + " | ".join(cells) + " |"


def _prompt_legend(rows: Sequence[Mapping[str, Any]]) -> str:
    """「A, B: weak; …」：每個組別名稱只列一次（round 2 的重複執行不重複列出）。"""
    styles: dict[str, list[str]] = {}
    for row in rows:
        labels = styles.setdefault(row["prompt_style"], [])
        if row["combo_label"] not in labels:
            labels.append(row["combo_label"])
    return "; ".join(f"{', '.join(labels)}: {style}" for style, labels in styles.items())


def _short(value: Any, width: int = 12) -> str:
    return "unknown" if value is None else f"{str(value)[:width]}"


def _round2_check_note(folders: Sequence[Mapping[str, Any]]) -> list[str]:
    """round-2 可比性的說明：檢查規則，以及每個 run 資料夾通過檢查的共同條件。"""
    rule = (
        "- Within one run folder, rag_eval scores round 2 only when every combo shares the model, retrieval "
        "mode, embedding model, corpus (name, document count, document-list digest), chunk count, "
        "indexed-text hash and git commit; otherwise it refuses to score. Across run folders, rows are grouped "
        "by (corpus, model) and each label names the corpus and the run folder. Compare F and G with E′ from "
        "the same run folder. Round-1 E vs E′ mixes a model change with the re-index."
    )
    notes = [rule]
    for folder in folders:
        shared = folder["shared"]
        pinned = {
            True: f"matches the pinned text of its {shared['document_count']} documents",
            False: f"does **not** match the pinned text of its {shared['document_count']} documents",
            None: "cannot be checked against `experiments/corpus_pin.json` (not every document is pinned)",
        }[folder["pinned_text"]]
        mode = "retrieval only" if shared["retrieval_only"] else "retrieval + answer"
        notes.append(
            f"- `{folder['folder']}` ({folder['label']}): model {shared['model']}, {mode}, embedding "
            f"{shared['embedding_model']}, {shared['corpus']} corpus of {shared['document_count']} documents "
            f"(list sha256 `{_short(shared['documents_digest'])}…`), {shared['chunk_count']} chunks, indexed text "
            f"{pinned} ({shared['text_source']}, sha256 `{_short(shared['corpus_sha256'])}…`), git commit "
            f"`{_short(shared['git_commit'], 7)}`."
        )
    return notes


def _failed_question_notes(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """列出因 LLM 或查詢擴展失敗而不計分的題目。"""
    notes = []
    for row in rows:
        for failure in row.get("failed_questions") or []:
            scope = (
                "answer metrics are n/a"
                if str(failure["error"]).startswith("llm_error")
                else "all metrics are n/a and it is left out of the mean distance"
            )
            notes.append(f"- {row['run']} Q{failure['index']}: `{failure['error']}`; {scope}.")
    return notes


def _stability_cell(summary: Mapping[str, Any], runs: int, distance: bool) -> str:
    """min–max；部分 run 為 n/a 時附「(k of n runs)」，全部 n/a 時為 n/a。"""
    if summary["n"] == 0:
        return "n/a"
    fmt = (lambda v: f"{v:.3f}") if distance else str
    cell = f"{fmt(summary['min'])}–{fmt(summary['max'])}"
    return cell if summary["n"] == runs else f"{cell} ({summary['n']} of {runs} runs)"


def _stability_section(stability: Sequence[Mapping[str, Any]]) -> list[str]:
    """metrics.md 的「Stability across repeats」段落。"""
    lines = [
        "## Stability across repeats",
        "",
        "For each (corpus, model, combo) in round 2: the min–max of each metric across the repeated run folders "
        "and the number of runs. \"(k of n runs)\" marks a metric that is n/a in some runs; those runs are left "
        "out of its range. The Q4 column counts hits only.",
        "",
    ]
    if not stability:
        return [*lines, "No round-2 runs yet.", ""]
    header = "| Combo | Corpus | Model | Runs | " + " | ".join(title for _, title in STABILITY_METRICS) + " |"
    rows = [
        "| "
        + " | ".join(
            [
                _cell(s["combo_label"]),
                _cell(s["corpus"]),
                _cell(s["model"]),
                str(s["runs"]),
                *[
                    _stability_cell(s["metrics"][key], s["runs"], key == "mean_distance_to_question")
                    for key, _ in STABILITY_METRICS
                ],
            ]
        )
        + " |"
        for s in stability
    ]
    return [*lines, header, "|---|---|---|---|" + "---|" * len(STABILITY_METRICS), *rows, ""]


def render_markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    generated_at: str,
    gold: Mapping[str, Any],
    round2_status: str,
    corpus_docs: int,
    round1_docs: int,
    gold_sha256: str,
    corpus_pin_digest: str,
    round2_folders: Sequence[Mapping[str, Any]] = (),
    stability: Sequence[Mapping[str, Any]] = (),
) -> str:
    """產生 metrics.md：開頭的判讀注意事項、指標說明、兩輪差異、逐 run 比較表與跨重複執行的穩定度表。

    corpus_docs：data/processed 目前的文件數；round1_docs：round-1 索引收錄的文件數（corpus_round1.json）。
    round2_folders：load_round2_folders 回傳的資料夾摘要（已通過資料夾內一致性檢查）；stability：stability_rows 的結果。
    """
    members = ", ".join(gold["q2_members"]["members"])
    groups = ", ".join(gold["q4_juste_debout"]["fact_groups"])
    round1_rows = [r for r in rows if r["round"] == 1]
    round2_rows = [r for r in rows if r["round"] == 2]
    round2_models = sorted({r["model"] for r in round2_rows})
    checks = [r["reconstruction"] for r in round1_rows if r.get("reconstruction")]
    checked = sum(c["cites"] for c in checks)
    matched = sum(c["section_label_matches"] for c in checks)
    model_note = ", ".join(round2_models) if round2_models else "set via LITELLM_MODEL; recorded in each results.json"
    cutoff_note = (
        [
            "- Cites whose distance to the original question exceeds the combo's max distance: "
            + ", ".join(f"{r['run']} {r['cites_over_max_distance']}/{r['cites_scored']}" for r in round2_rows)
            + ".",
        ]
        if round2_rows
        else []
    )
    failures = _failed_question_notes(rows)

    lines = [
        "# Pre-registered retrieval evaluation (round 1 vs round 2)",
        "",
        f"*Generated by `rag_eval.py` at {generated_at}. Metric definitions and aliases live in "
        f"`experiments/gold.json` (sha256 `{gold_sha256}`), registered before any round-2 run.*",
        "",
        "**Read before comparing rows:**",
        "",
        f"- **Embedding window.** The embedding model (`{EMBEDDING_MODEL_NAME}`) reads only the first "
        f"{EMBEDDING_MAX_TOKENS} tokens of each chunk; the rest of a longer chunk never reaches dense retrieval, "
        "while BM25 indexes the full chunk text. This is a possible confound when reading F: part of a difference "
        "between F and E′ may come from BM25 seeing text the embedding never read, not from lexical matching as such.",
        "- **Loose alias.** The gold alias `Pete` (for Popin' Pete) is loose: any standalone \"Pete\" counts, so "
        "the Q2 member counts can include a different Pete.",
        f"- **What each round used.** Round-1 rows come from an index of {round1_docs} documents "
        f"(`experiments/corpus_round1.json`) and from {ROUND1_MODEL.split(' (')[0]} answers via a course proxy. "
        "Round-2 rows use the model and corpus recorded in their results.json; each round-2 row label names its "
        "corpus and run folder.",
        "",
        "## Metrics",
        "",
        "- **Mean cosine distance to question (5 Qs)**: for each fixed question, the mean cosine distance "
        "between the original question and each retrieved chunk, averaged over the five questions "
        "(round 1: distances recorded in `q1.md`–`q5.md`; round 2: `distance_to_question`). Context, not a "
        "quality score: with a fixed k the dense top-k is by construction the minimum-distance set, so hybrid "
        "fusion (F) or query expansion (G) can only keep or raise it.",
        f"- **Q2 members named (/9)**: how many of the nine notable Electric Boogaloos members listed in the "
        f"corpus ({members}) the Q2 answer names. Matching uses the gold aliases, case-insensitive, never "
        "inside a longer Latin-letter word.",
        "- **Q2 members in context (/9)**: the same nine names found in the retrieved chunk texts "
        "(file names in the prompt's source headers are not counted).",
        f"- **Q4 Juste Debout chunks in top-k**: retrieved Q4 chunks that mention \"Juste Debout\" together "
        f"with at least one fact group ({groups}), shown as hits/retrieved.",
        "- **Q4 fact groups in answer**: fact groups (0–5) the Q4 answer states, counted only when the answer "
        "mentions Juste Debout.",
        "- `n/a`: no answer to score (retrieval-only run or LLM error). When query expansion failed for a "
        "question, its retrieval metrics are n/a too and it is left out of the mean distance.",
        "- Max distance: F applies it to the dense channel only (BM25-only chunks bypass it); G applies it "
        "to each sub-query's distance, not to the original question.",
        "",
        "## Rounds",
        "",
        "- **Round 1** (A–E, April 2026): rows reuse the answers and distances recorded in "
        "`experiments/combo_a`–`combo_e`; nothing was re-run. Model: "
        f"{ROUND1_MODEL}. The round-1 index held {round1_docs} documents (`experiments/corpus_round1.json`; "
        f"`data/processed` now holds {corpus_docs}). Chunk texts were not stored, so they are rebuilt by "
        "re-chunking `data/processed/<source_file>` with each combo's round-1 chunker, after checking that "
        f"`data/processed` matches `experiments/corpus_pin.json` (digest `{_short(corpus_pin_digest)}…`, text "
        f"committed before round 1); {matched}/{checked} rebuilt chunks reproduce the section label recorded "
        "next to their cite.",
        f"- **Round 2** (E′ control, F hybrid, G query expansion): {round2_status}. Model: {model_note}. "
        "Round-2 chunk texts and distances come from `results.json`, i.e. from the index built at run time.",
        *_round2_check_note(round2_folders),
        *cutoff_note,
        "- Round-1 D was logged as \"top-10 + rerank\", but no reranker existed: the distance-sorted "
        "top-10 was cut to 5, i.e. plain dense top-5.",
        f"- Prompt styles — {_prompt_legend(rows)}.",
        "",
        *(["## Questions not scored", "", *failures, ""] if failures else []),
        "## Results",
        "",
        TABLE_HEADER,
        "|---|---|---|---|---|---|---|---|",
        *[_format_row(r) for r in rows],
        "",
        *_stability_section(stability),
    ]
    return "\n".join(lines)


def write_outputs(out_dir: Path, payload: Mapping[str, Any], markdown: str) -> tuple[Path, Path]:
    """寫出 metrics.json 與 metrics.md，回傳兩個路徑。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "metrics.json"
    md_path = out_dir / "metrics.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


# ─── Round-1 重現檢查（--check-reproduction）──────────────────────────────

# 影響檢索結果的 round-1 設定（prompt_style 只影響回答，不檢查）
_RETRIEVAL_SETTING_KEYS: tuple[str, ...] = (
    "chunk_desc",
    "top_k",
    "max_distance",
    "truncate_to",
    "use_hybrid",
    "use_expand",
)
# --report-out 不可覆寫的 experiments/ 檔案（第一輪彙整與評估輸出）
_PROTECTED_EXPERIMENT_FILES = frozenset({"summary.md", "metrics.md"})

Cite = tuple[str, int, float]


def _chunk_ids(cites: Sequence[Cite]) -> list[str]:
    return [f"{src}::{idx}" for src, idx, _ in cites]


def compare_reproduction(
    recorded: Sequence[Cite],
    reproduced: Sequence[Cite] | None,
    tolerance: float = REPRODUCTION_TOLERANCE,
) -> dict[str, Any]:
    """單題比對：chunk id（檔名 + chunk 編號）與順序完全相同，且每個 |記錄距離 − 重現距離| <= tolerance。

    reproduced 為 None 代表沒有重現結果。回傳 {match, same_chunks, reason, added, removed,
    max_abs_distance_diff}；added / removed 為重現多出 / 缺少的 chunk id。純函數。
    """
    if reproduced is None:
        return {"match": False, "same_chunks": False, "reason": "missing", "added": [], "removed": [],
                "max_abs_distance_diff": None}
    recorded_ids, reproduced_ids = _chunk_ids(recorded), _chunk_ids(reproduced)
    if recorded_ids != reproduced_ids:
        added = [cid for cid in reproduced_ids if cid not in recorded_ids]
        removed = [cid for cid in recorded_ids if cid not in reproduced_ids]
        reason = "order differs" if not added and not removed else f"chunks differ (+{len(added)} / -{len(removed)})"
        return {"match": False, "same_chunks": False, "reason": reason, "added": added, "removed": removed,
                "max_abs_distance_diff": None}
    max_diff = max((abs(a[2] - b[2]) for a, b in zip(recorded, reproduced)), default=0.0)
    match = max_diff <= tolerance
    return {
        "match": match,
        "same_chunks": True,
        "reason": None if match else f"distance off by {max_diff:.4f}",
        "added": [],
        "removed": [],
        "max_abs_distance_diff": max_diff,
    }


def _load_reproduced_combo(results_path: Path, combo_id: str) -> tuple[dict[int, dict[str, Any]], str | None, dict[str, Any]]:
    """讀取重現 run 的 combo_<id>/results.json → ({題號: {question, cites}}, 問題說明或 None, 環境摘要)。

    檔案缺漏、無法解析、不符契約，或檢索設定與 round 1 不同時，問題說明不為 None（該組 5 題皆記為不符）。
    """
    if not results_path.is_file():
        return {}, "missing (no results.json)", {}
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        combo = data["combo"]
        environment = data.get("environment") or {}
        info = {
            "collection": combo.get("collection"),
            "chunk_count": environment.get("chunk_count"),
            "corpus": environment.get("corpus"),
            "document_count": environment.get("document_count"),
            "corpus_sha256": environment.get("corpus_sha256"),
            "retrieval_only": data.get("retrieval_only"),
        }
        if str(combo["id"]).lower() != combo_id:
            return {}, f"results.json is for combo {combo['id']!r}", info
        differs = [k for k in _RETRIEVAL_SETTING_KEYS if combo.get(k) != ROUND1_COMBOS[combo_id][k]]
        if differs:
            return {}, f"retrieval settings differ from round 1 ({', '.join(differs)})", info
        questions = {
            int(q["index"]): {
                "question": str(q["question"]),
                "cites": [(str(c["source_file"]), int(c["chunk_index"]), float(c["distance"])) for c in q["cites"]],
            }
            for q in data["questions"]
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return {}, f"unreadable results.json ({type(exc).__name__})", {}
    return questions, None, info


def _read_run_json(run_dir: Path) -> dict[str, Any] | None:
    """讀取 run.json（不存在或無法解析時在 stderr 警告並回傳 None）。"""
    path = run_dir / "run.json"
    if not path.is_file():
        _warn(f"{_display_path(path)} 不存在；報告不含執行環境")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("run.json 不是 JSON 物件")
        return data
    except (OSError, ValueError) as exc:
        _warn(f"{_display_path(path)} 無法讀取（{type(exc).__name__}）；報告不含執行環境")
        return None


def check_reproduction(run_dir: Path, round1_dir: Path = EXPERIMENTS_DIR) -> dict[str, Any]:
    """比對 run_dir/combo_a..e/results.json 與 round1_dir/combo_a..e/q1..q5.md，回傳 25 格的結果。

    round-1 記錄無法解析時拋 OSError / ValueError（記錄本身有問題時不產生報告）。
    """
    combos: list[dict[str, Any]] = []
    for combo_id in ROUND1_COMBOS:
        reproduced, problem, info = _load_reproduced_combo(run_dir / f"combo_{combo_id}" / "results.json", combo_id)
        cells: list[dict[str, Any]] = []
        for n in range(1, QUESTION_COUNT + 1):
            recorded = parse_round1_question(round1_dir / f"combo_{combo_id}" / f"q{n}.md")
            question = reproduced.get(n)
            if problem is not None:
                cell = {**compare_reproduction(recorded["cites"], None), "reason": problem}
            elif question is None:
                cell = {**compare_reproduction(recorded["cites"], None), "reason": "question missing"}
            elif normalize(question["question"]) != normalize(recorded["question"]):
                cell = {**compare_reproduction(recorded["cites"], None), "reason": "question text differs"}
            else:
                cell = compare_reproduction(recorded["cites"], question["cites"])
            cells.append(
                {
                    "question": n,
                    **cell,
                    "recorded": [list(c) for c in recorded["cites"]],
                    "reproduced": [list(c) for c in question["cites"]] if question and problem is None else None,
                }
            )
        combos.append({"combo_id": combo_id, "info": info, "cells": cells})
    matched = sum(cell["match"] for combo in combos for cell in combo["cells"])
    return {
        "run_dir": _display_path(run_dir),
        "round1_dir": _display_path(round1_dir),
        "run": _read_run_json(run_dir),
        "combos": combos,
        "matched": matched,
        "total": len(combos) * QUESTION_COUNT,
    }


def _reproduction_cell(cell: Mapping[str, Any]) -> str:
    if cell["match"]:
        return f"ok (max Δ {cell['max_abs_distance_diff']:.4f})"
    return f"MISMATCH: {cell['reason']}"


def reproduction_table(result: Mapping[str, Any]) -> list[str]:
    """5×5 結果表（每列一個 combo、每欄一題）與總計列。"""
    lines = [
        "| Combo | " + " | ".join(f"Q{n}" for n in range(1, QUESTION_COUNT + 1)) + " | Matched |",
        "|---|" + "---|" * QUESTION_COUNT + "---|",
    ]
    for combo in result["combos"]:
        ok = sum(cell["match"] for cell in combo["cells"])
        cells = " | ".join(_reproduction_cell(cell) for cell in combo["cells"])
        lines.append(f"| {combo['combo_id'].upper()} | {cells} | {ok}/{len(combo['cells'])} |")
    return lines


def _format_cites(cites: Sequence[Sequence[Any]] | None) -> str:
    if cites is None:
        return "(none)"
    return ", ".join(f"`{src}::{idx}` ({float(d):.4f})" for src, idx, d in cites) or "(no cites)"


def _corpus_lines(run: Mapping[str, Any] | None, round1_definition: Mapping[str, Any]) -> list[str]:
    """報告的「Corpus」段落：run.json 記錄的語料，與 experiments/corpus_round1.json 比對。"""
    from rag_common import document_list_digest  # 延遲匯入：純函數不必載入 rag_common

    corpus = (run or {}).get("corpus")
    if not isinstance(corpus, dict):
        return ["- run.json has no corpus record."]
    round1_digest = document_list_digest(round1_definition["documents"])
    excluded = ", ".join(f"`{name}`" for name in round1_definition["excluded"])
    same_list = corpus.get("documents_digest") == round1_digest
    lines = [
        f"- Corpus: **{corpus.get('name', 'unknown')}** ({corpus.get('definition', 'definition not recorded')}), "
        f"{corpus.get('document_count')} documents, document-list sha256 `{_short(corpus.get('documents_digest'), 16)}…`.",
        "- Document list "
        + ("matches" if same_list else "does **not** match")
        + f" `experiments/corpus_round1.json` ({len(round1_definition['documents'])} documents; it excludes "
        f"{excluded}).",
        f"- Indexed text: `data/processed` digest `{_short(corpus.get('processed_digest'), 16)}…`; matches "
        f"`experiments/corpus_pin.json`: {corpus.get('matches_pin')}.",
    ]
    return lines


def _environment_lines(result: Mapping[str, Any]) -> list[str]:
    """報告的「Environment」段落（取自 run.json 與各 combo 的 results.json）。"""
    run = result["run"] or {}
    git = run.get("git") or {}
    embedding = run.get("embedding") or {}
    lines = [
        f"- Run folder: `{result['run_dir']}`; run.json generated at {run.get('generated_at', 'unknown')}.",
        f"- Git commit `{_short(git.get('commit'), 7)}` (uncommitted tracked changes: {git.get('dirty')}).",
        f"- Embedding: {embedding.get('model', 'unknown')}, max_seq_length {embedding.get('max_seq_length')}.",
        f"- Retrieval only: {run.get('retrieval_only')} (no LLM call for combos A–E); text source: "
        + ("committed `data/processed`" if run.get("reuse_processed") else "re-extracted from `data/raw`")
        + ".",
        "",
        "| Combo | Collection | Chunks | Corpus | Status |",
        "|---|---|---|---|---|",
    ]
    statuses = run.get("combos") or {}
    for combo in result["combos"]:
        info = combo["info"]
        status = (statuses.get(combo["combo_id"]) or {}).get("status", "not in run.json")
        lines.append(
            f"| {combo['combo_id'].upper()} | `{info.get('collection')}` | {info.get('chunk_count')} | "
            f"{info.get('corpus')} | {status} |"
        )
    return lines


def render_reproduction_report(
    result: Mapping[str, Any], *, generated_at: str, round1_definition: Mapping[str, Any]
) -> str:
    """round1_reproduction.md：方法、執行環境、語料與 25 格結果（含不符的明細）。"""
    total, matched = result["total"], result["matched"]
    verdict = (
        f"**Result: {matched}/{total} cells reproduce the recorded round-1 retrieval.**"
        if matched == total
        else f"**Result: {matched}/{total} cells reproduce the recorded round-1 retrieval; "
        f"{total - matched} MISMATCH.**"
    )
    corpus_record = (result["run"] or {}).get("corpus")
    corpus_flag = f" --corpus {corpus_record['name']}" if isinstance(corpus_record, dict) and corpus_record.get("name") else ""
    details: list[str] = []
    for combo in result["combos"]:
        for cell in combo["cells"]:
            if cell["match"]:
                continue
            details += [
                f"### Combo {combo['combo_id'].upper()} · Q{cell['question']}",
                "",
                f"- Reason: {cell['reason']}",
                f"- Recorded: {_format_cites(cell['recorded'])}",
                f"- Reproduced: {_format_cites(cell['reproduced'])}",
            ]
            if cell["added"] or cell["removed"]:
                details.append(
                    "- Added: " + (", ".join(f"+{cid}" for cid in cell["added"]) or "none")
                    + "; removed: " + (", ".join(f"-{cid}" for cid in cell["removed"]) or "none")
                )
            details.append("")
    lines = [
        "# Round-1 retrieval reproduction check",
        "",
        f"*Generated by `rag_eval.py --check-reproduction` at {generated_at}.*",
        "",
        verdict,
        "",
        "## Method",
        "",
        f"- Recorded: `{result['round1_dir']}/combo_a`–`combo_e`, `q1.md`–`q5.md` (round 1, April 2026; "
        "distances printed with 3 decimals).",
        f"- Reproduced: `{result['run_dir']}/combo_a`–`combo_e/results.json`, written by "
        f"`experiment_runner.py --combos a b c d e{corpus_flag} --retrieval-only`: each combo rebuilds its own Chroma "
        "collection with its round-1 chunker and retrieves with its round-1 settings; no LLM call.",
        f"- A cell (combo × question) matches when the reproduced cites have the same chunk ids (source file + "
        f"chunk index) in the same order as the recorded cites, and every |recorded distance − reproduced "
        f"distance| ≤ {REPRODUCTION_TOLERANCE}. Retrieval settings in each results.json (chunking, top-k, max "
        "distance, truncation, hybrid, expansion) must equal the round-1 settings, otherwise all five cells "
        "of that combo count as mismatches.",
        "",
        "## Corpus",
        "",
        *_corpus_lines(result["run"], round1_definition),
        "",
        "## Environment",
        "",
        *_environment_lines(result),
        "",
        "## Result",
        "",
        *reproduction_table(result),
        "",
        f"Total: {matched}/{total} cells match.",
        "",
        *(["## Mismatches", "", *details] if details else []),
    ]
    return "\n".join(lines)


def _inside(path: Path, parent: Path) -> bool:
    """path 是否為 parent 或其下（不分大小寫：macOS APFS 預設不分大小寫）。"""
    child, base = str(path).casefold(), str(parent).casefold()
    return child == base or child.startswith(base.rstrip(os.sep) + os.sep)


def resolve_report_path(raw: Path) -> Path:
    """解析 --report-out；不符合規則時拋 ValueError。

    必須是 .md；repo 內只允許 experiments/ 之下，且不可在第一輪的 experiments/combo_* 之內，
    也不可是 experiments/summary.md、metrics.md。repo 外（例如暫存目錄）允許。
    """
    path = raw.expanduser().resolve()
    if path.suffix.lower() != ".md":
        raise ValueError(f"--report-out 必須是 .md 檔（收到 {path.name}）")
    root, experiments = ROOT.resolve(), EXPERIMENTS_DIR.resolve()
    if not _inside(path, root):
        return path
    if not _inside(path, experiments):
        raise ValueError("--report-out 在 repo 內時必須位於 experiments/ 之下")
    # 比較一律不分大小寫：resolve() 保留使用者輸入的大小寫，「Experiments/Summary.md」在 APFS 上是同一個檔案
    first = path.parts[len(experiments.parts)]
    if first.casefold().startswith("combo_"):
        raise ValueError(f"--report-out 不可位於第一輪結果目錄 experiments/{first} 之內")
    if _inside(path.parent, experiments) and _inside(experiments, path.parent):
        if path.name.casefold() in _PROTECTED_EXPERIMENT_FILES:
            raise ValueError(f"--report-out 不可覆寫 experiments/{path.name}")
    return path


def run_reproduction_check(run_dir: Path, round1_dir: Path, report_out: Path | None) -> int:
    """--check-reproduction 的流程：比對、印出 5×5 表、寫報告；全部相符回 0，有不符或錯誤回 1，報告路徑不合法回 2。"""
    try:
        report_path = resolve_report_path(report_out if report_out is not None else REPRODUCTION_REPORT_PATH)
    except ValueError as exc:
        print(f"[rag_eval] 錯誤（{_describe_exc(exc)}）", file=sys.stderr)
        return 2
    try:
        if not run_dir.is_dir():
            raise ValueError(f"找不到重現 run 目錄 {_display_path(run_dir)}")
        from rag_common import load_corpus_definition  # 延遲匯入：純函數不必載入 rag_common

        result = check_reproduction(run_dir, round1_dir)
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        report = render_reproduction_report(
            result, generated_at=generated_at, round1_definition=load_corpus_definition(CORPUS_ROUND1_PATH)
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    except Exception as exc:  # CLI 邊界：回報型別與（去除本機路徑的）訊息後以非 0 結束
        print(f"[rag_eval] 錯誤（{_describe_exc(exc)}）", file=sys.stderr)
        return 1
    print("\n".join(reproduction_table(result)))
    print(f"\n[rag_eval] round-1 reproduction: {result['matched']}/{result['total']} cells match")
    print(f"[rag_eval] 已寫入 {_display_path(report_path)}")
    return 0 if result["matched"] == result["total"] else 1


# ─── CLI ───────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="預先登錄的 RAG 評估：round 1（A–E）vs round 2（E′、F、G）")
    p.add_argument("--round1", type=Path, default=EXPERIMENTS_DIR, help="含 combo_a..e/q*.md 的目錄")
    p.add_argument(
        "--round2",
        type=Path,
        nargs="+",
        default=[ROUND2_DIR],
        help="一或多個目錄，各自遞迴搜尋 run 資料夾（直接含 combo_*/results.json 的目錄）；預設 experiments/round2",
    )
    p.add_argument("--out", type=Path, default=EXPERIMENTS_DIR, help="metrics.json / metrics.md 輸出目錄")
    p.add_argument(
        "--check-reproduction",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="改為重現檢查：比對 RUN_DIR/combo_a..e/results.json（--retrieval-only）與 round-1 的 q*.md",
    )
    p.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="重現檢查報告的輸出路徑（預設 experiments/round1_reproduction.md）",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_reproduction is not None:
        return run_reproduction_check(args.check_reproduction, args.round1, args.report_out)
    try:
        from rag_common import document_list_digest, load_corpus_definition  # 延遲匯入

        gold = load_gold(GOLD_PATH)
        gold_sha256 = hashlib.sha256(GOLD_PATH.read_bytes()).hexdigest()
        corpus_pin_digest = check_round1_corpus(PROCESSED_DIR, CORPUS_PIN_PATH)
        round1_documents = load_corpus_definition(CORPUS_ROUND1_PATH)["documents"]
        runs = load_round1_runs(args.round1, PROCESSED_DIR)
        reference = {q["index"]: q["question"] for q in runs[0]["questions"]} if runs else {}
        round2_shown = ", ".join(f"`{_display_path(d)}`" for d in args.round2)
        round2, round2_folders = load_round2_folders(args.round2, reference)
        if round2_folders:
            runs = runs + round2
            round2_status = f"read from {len(round2_folders)} run folder(s) under {round2_shown}"
        else:
            print(f"[rag_eval] round 2: no run folder (combo_*/results.json) under {round2_shown} yet; skipped.")
            round2_status = f"no results yet (no `combo_*/results.json` under {round2_shown})"
        if not runs:
            raise ValueError("沒有任何可評估的 run")

        rows = compute_metrics(runs, gold)
        stability = stability_rows(rows)
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        corpus_docs = len(list(PROCESSED_DIR.glob("*.txt")))
        markdown = render_markdown(
            rows,
            generated_at=generated_at,
            gold=gold,
            round2_status=round2_status,
            corpus_docs=corpus_docs,
            round1_docs=len(round1_documents),
            gold_sha256=gold_sha256,
            corpus_pin_digest=corpus_pin_digest,
            round2_folders=round2_folders,
            stability=stability,
        )
        payload = {
            "generated_at": generated_at,
            "gold": _display_path(GOLD_PATH),
            "gold_sha256": gold_sha256,
            "corpus_pin": {"path": _display_path(CORPUS_PIN_PATH), "digest": corpus_pin_digest},
            "round1_corpus": {
                "path": _display_path(CORPUS_ROUND1_PATH),
                "document_count": len(round1_documents),
                "documents_digest": document_list_digest(round1_documents),
            },
            "round2_status": round2_status,
            "round2_runs": round2_folders,
            "rows": rows,
            "stability": stability,
        }
        json_path, md_path = write_outputs(args.out, payload, markdown)
    except Exception as exc:  # CLI 邊界：回報型別與（去除本機路徑的）訊息後以非 0 結束
        print(f"[rag_eval] 錯誤（{_describe_exc(exc)}）", file=sys.stderr)
        return 1

    print("\n".join([TABLE_HEADER, "|---|---|---|---|---|---|---|---|", *[_format_row(r) for r in rows]]))
    print(f"\n[rag_eval] 已寫入 {_display_path(json_path)}、{_display_path(md_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
