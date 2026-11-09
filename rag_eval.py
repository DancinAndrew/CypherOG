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
  與 experiments/corpus_pin.json 完全一致，否則拒絕評估。
- Round 2：<round2>/combo_{e,f,g}/results.json（由 experiment_runner.py 寫出，契約見 load_round2_run）。
  各組的模型、檢索模式、embedding、chunk 數、建索引文字的指紋與 git commit 必須一致，否則拒絕評估。
  查詢擴展失敗的題目，檢索與回答指標皆為 n/a；LLM 失敗的題目，回答指標為 n/a。

輸出：<out>/metrics.json 與 <out>/metrics.md。

Usage:
    python rag_eval.py
    python rag_eval.py --round1 experiments --round2 experiments/round2 --out experiments
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
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
ROUND2_DIR = EXPERIMENTS_DIR / "round2"
PROCESSED_DIR = ROOT / "data" / "processed"

QUESTION_COUNT = 5

# Round 1 的 q*.md 未記錄模型名稱；當時 LITELLM_MODEL 預設為 gemini-2.5-flash，經課程提供的代理呼叫。
ROUND1_MODEL = "gemini-2.5-flash (course OpenAI-compatible proxy)"

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

# round-2 各組必須相同的執行條件（results.json 的 model、retrieval_only 與 environment 欄位）
ROUND2_ENVIRONMENT_KEYS: tuple[str, ...] = (
    "embedding_model",
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
        "round": 1,
        "combo_id": combo_id,
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
    """讀取 <round2>/combo_<id>/results.json（experiment_runner.py 寫出）。

    契約重點：combo{id, chunk_desc, top_k, max_distance, truncate_to, prompt_style, use_hybrid,
    use_expand, ...}、model、retrieval_only、environment{embedding_model, chunk_count, text_source,
    corpus_sha256, git_commit, ...}、questions[{index, question, cites[{text, distance_to_question, ...}],
    answer, error}]。distance_to_question 為「原始問題」與 chunk 的 cosine distance（G 的 distance
    可能是對子查詢量的，故不使用）。不符契約時拋 ValueError。
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
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{_display_path(results_path)} 不符合 results.json 契約（{type(exc).__name__}: {exc}）"
        ) from exc
    return {
        "run": ROUND2_LABELS.get(combo_id, combo_id.upper()),
        "round": 2,
        "combo_id": combo_id,
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
    """列出 round2_dir/combo_*/results.json（目錄不存在時回空 list）。"""
    if not round2_dir.is_dir():
        return []
    return sorted(round2_dir.glob("combo_*/results.json"))


def load_round2_runs(round2_dir: Path) -> list[dict[str, Any]]:
    """讀取所有 round-2 results.json；格式錯誤的檔案在 stderr 警告後略過。"""
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
        "round": run["round"],
        "combo_id": run["combo_id"],
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
    """所有 run → 指標列，依 (round, combo_id) 排序。"""
    ordered = sorted(runs, key=lambda r: (r["round"], r["combo_id"]))
    return [_score_run(run, gold) for run in ordered]


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
    styles: dict[str, list[str]] = {}
    for row in rows:
        styles.setdefault(row["prompt_style"], []).append(row["run"])
    return "; ".join(f"{', '.join(runs)}: {style}" for style, runs in styles.items())


def _short(value: Any, width: int = 12) -> str:
    return "unknown" if value is None else f"{str(value)[:width]}"


def _round2_check_note(shared: Mapping[str, Any] | None, corpus_pin_digest: str) -> list[str]:
    """round-2 可比性的說明：通過檢查時列出共同條件；沒有 round-2 結果時說明檢查規則。"""
    if shared is None:
        return [
            "- Compare F and G with E′ from the same round. rag_eval scores round 2 only when every round-2 "
            "run shares the model, retrieval mode, embedding model, chunk count, indexed-text hash and git "
            "commit. Round-1 E vs E′ mixes a model change with the re-index.",
        ]
    pinned = shared["corpus_sha256"] == corpus_pin_digest
    return [
        f"- Compare F and G with E′. rag_eval checked that every round-2 run shares the model "
        f"({shared['model']}), retrieval mode ({'retrieval only' if shared['retrieval_only'] else 'retrieval + answer'}), "
        f"embedding model ({shared['embedding_model']}), chunk count ({shared['chunk_count']}), indexed text "
        f"({shared['text_source']}, sha256 `{_short(shared['corpus_sha256'])}…`) and git commit "
        f"(`{_short(shared['git_commit'], 7)}`). Round-1 E vs E′ mixes a model change with the re-index.",
        "- Round-2 indexed text "
        + ("matches" if pinned else "does **not** match")
        + " the pinned round-1 corpus.",
    ]


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


def render_markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    generated_at: str,
    gold: Mapping[str, Any],
    round2_status: str,
    corpus_docs: int,
    gold_sha256: str,
    corpus_pin_digest: str,
    round2_shared: Mapping[str, Any] | None = None,
) -> str:
    """產生 metrics.md：指標說明、兩輪差異，以及單一比較表。

    round2_shared：round-2 各 run 共同的執行條件（已通過 round2_inconsistencies 檢查）；沒有 round-2 時為 None。
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
        f"{ROUND1_MODEL}. Chunk texts were not stored, so they are rebuilt by re-chunking "
        f"`data/processed/<source_file>` ({corpus_docs} documents) with each combo's round-1 chunker, after "
        f"checking that `data/processed` matches `experiments/corpus_pin.json` (digest `{_short(corpus_pin_digest)}…`, "
        f"text committed before round 1); {matched}/{checked} rebuilt chunks reproduce the section label "
        "recorded next to their cite.",
        f"- **Round 2** (E′ control, F hybrid, G query expansion): {round2_status}. Model: {model_note}. "
        "Round-2 chunk texts and distances come from `results.json`, i.e. from the index built at run time.",
        *_round2_check_note(round2_shared, corpus_pin_digest),
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


# ─── CLI ───────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="預先登錄的 RAG 評估：round 1（A–E）vs round 2（E′、F、G）")
    p.add_argument("--round1", type=Path, default=EXPERIMENTS_DIR, help="含 combo_a..e/q*.md 的目錄")
    p.add_argument("--round2", type=Path, default=ROUND2_DIR, help="含 combo_*/results.json 的目錄")
    p.add_argument("--out", type=Path, default=EXPERIMENTS_DIR, help="metrics.json / metrics.md 輸出目錄")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        gold = load_gold(GOLD_PATH)
        gold_sha256 = hashlib.sha256(GOLD_PATH.read_bytes()).hexdigest()
        corpus_pin_digest = check_round1_corpus(PROCESSED_DIR, CORPUS_PIN_PATH)
        runs = load_round1_runs(args.round1, PROCESSED_DIR)
        reference = {q["index"]: q["question"] for q in runs[0]["questions"]} if runs else {}
        round2_shown = f"{_display_path(args.round2)}/combo_*/results.json"
        round2_shared: Mapping[str, Any] | None = None
        if round2_results_paths(args.round2):
            round2 = keep_consistent_runs(
                keep_preregistered_round2(load_round2_runs(args.round2)), reference
            )
            conflicts = round2_inconsistencies(round2)
            if conflicts:
                detail = "; ".join(f"{key}: {values}" for key, values in conflicts.items())
                raise ValueError(f"round-2 各組的執行條件不一致，不可互相比較（{detail}）")
            round2_shared = round2[0]["consistency"] if round2 else None
            runs = runs + round2
            round2_status = f"read from `{round2_shown}`"
        else:
            print(f"[rag_eval] round 2: no results.json under {_display_path(args.round2)} yet; skipped.")
            round2_status = f"no results yet (`{round2_shown}` not found)"
        if not runs:
            raise ValueError("沒有任何可評估的 run")

        rows = compute_metrics(runs, gold)
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        corpus_docs = len(list(PROCESSED_DIR.glob("*.txt")))
        markdown = render_markdown(
            rows,
            generated_at=generated_at,
            gold=gold,
            round2_status=round2_status,
            corpus_docs=corpus_docs,
            gold_sha256=gold_sha256,
            corpus_pin_digest=corpus_pin_digest,
            round2_shared=round2_shared,
        )
        payload = {
            "generated_at": generated_at,
            "gold": _display_path(GOLD_PATH),
            "gold_sha256": gold_sha256,
            "corpus_pin": {"path": _display_path(CORPUS_PIN_PATH), "digest": corpus_pin_digest},
            "round2_status": round2_status,
            "round2_shared": round2_shared,
            "rows": rows,
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
