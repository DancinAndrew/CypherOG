#!/usr/bin/env python3
"""RAG 七組實驗跑分腳本。

每組 combo 獨立 rebuild Chroma collection + 跑固定 5 題 → 輸出 experiments/combo_{X}/q{N}.md
最後匯整 experiments/summary.md 橫向比較。

Combo 一覽：
  A: Baseline（字元暴力切 + 無過濾 + 弱 prompt）
  B: 段落感知 + 距離過濾
  C: 大 chunk + 強 prompt
  D: 小 chunk + rerank
  E: CoT prompt（最終選用）
  F: CoT + Hybrid Search（BM25+向量+RRF）— 人名/組織名精確匹配
  G: CoT + Query Expansion（多角度子查詢）— 跨語言召回

Usage:
    python experiment_runner.py              # 跑全部 7 組
    python experiment_runner.py --combos f g # 只跑新增組
    python experiment_runner.py --skip-rebuild  # 跳過 rebuild
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from data_update import (
    chunk_text,
    naive_chunk_text,
    rebuild_processed_and_index,
)
from rag_common import litellm_chat_completion
from rag_query import build_messages, retrieve_context, hybrid_retrieve, expand_query

# ─── 固定測試問題 ─────────────────────────────────────────────────────────────
QUESTIONS: list[str] = [
    "What is Popping and who created it?",
    "Electric Boogaloos 的成員有哪些？他們各自的貢獻是什麼？",
    "Popping 和 Boogaloo 有什麼差別？",
    "Juste Debout 比賽的規則和歷史？",
    "如何判斷一場 Popping battle 的勝負？",
]

# ─── 五組 Combo 設定 ───────────────────────────────────────────────────────────
COMBOS: list[dict] = [
    {
        "id": "a",
        "name": "Combo A — Baseline（字元暴力切）",
        "desc": "最差基準線：純字元切割、無距離過濾、弱 prompt。",
        "collection": "hw3_rag_combo_a",
        "chunk_fn": lambda t: naive_chunk_text(t, 500),
        "chunk_desc": "naive / size=500 / overlap=0",
        "top_k": 5,
        "max_distance": 1.0,
        "rerank": False,
        "prompt_style": "weak",
        "use_hybrid": False,
        "use_expand": False,
    },
    {
        "id": "b",
        "name": "Combo B — 段落感知 + 距離過濾",
        "desc": "升級 chunking 與 retrieval，保持弱 prompt，單獨觀察前兩項改善效果。",
        "collection": "hw3_rag_combo_b",
        "chunk_fn": lambda t: chunk_text(t, 600, 100),
        "chunk_desc": "recursive / size=600 / overlap=100",
        "top_k": 5,
        "max_distance": 0.65,
        "rerank": False,
        "prompt_style": "weak",
        "use_hybrid": False,
        "use_expand": False,
    },
    {
        "id": "c",
        "name": "Combo C — 大 chunk + 強 prompt",
        "desc": "大 chunk 保留完整語境，搭配強制規則 prompt 與 XML 標籤，觀察幻覺減少效果。",
        "collection": "hw3_rag_combo_c",
        "chunk_fn": lambda t: chunk_text(t, 1200, 200),
        "chunk_desc": "recursive / size=1200 / overlap=200",
        "top_k": 3,
        "max_distance": 0.65,
        "rerank": False,
        "prompt_style": "strong",
        "use_hybrid": False,
        "use_expand": False,
    },
    {
        "id": "d",
        "name": "Combo D — 小 chunk + 高 top-k + rerank",
        "desc": "小而精確的 chunk + 寬搜窄篩（top-10→rerank→top-5）+ 嚴格距離過濾。",
        "collection": "hw3_rag_combo_d",
        "chunk_fn": lambda t: chunk_text(t, 300, 50),
        "chunk_desc": "recursive / size=300 / overlap=50",
        "top_k": 10,
        "max_distance": 0.5,
        "rerank": True,
        "prompt_style": "strong",
        "use_hybrid": False,
        "use_expand": False,
    },
    {
        "id": "e",
        "name": "Combo E — CoT + 結構化 context",
        "desc": "與 B 相同的 chunking/retrieval，但加 Chain-of-Thought prompt，強制 LLM 先列 facts 再綜合。",
        "collection": "hw3_rag_combo_e",
        "chunk_fn": lambda t: chunk_text(t, 600, 100),
        "chunk_desc": "recursive / size=600 / overlap=100",
        "top_k": 5,
        "max_distance": 0.65,
        "rerank": False,
        "prompt_style": "cot",
        "use_hybrid": False,
        "use_expand": False,
    },
    {
        "id": "f",
        "name": "Combo F — CoT + Hybrid Search（BM25+Vector+RRF）",
        "desc": "在 Combo E 基礎上加入 BM25 混合搜尋，對專有名詞（Boogaloo Sam、Electric Boogaloos）精確匹配效果更佳。",
        "collection": "hw3_rag_combo_f",
        "chunk_fn": lambda t: chunk_text(t, 600, 100),
        "chunk_desc": "recursive / size=600 / overlap=100",
        "top_k": 5,
        "max_distance": 0.65,
        "rerank": False,
        "prompt_style": "cot",
        "use_hybrid": True,
        "use_expand": False,
    },
    {
        "id": "g",
        "name": "Combo G — CoT + Query Expansion（多角度子查詢）",
        "desc": "在 Combo E 基礎上加入 Query Expansion，用 LLM 將查詢改寫為中英文多角度子查詢，提高跨語言召回率。",
        "collection": "hw3_rag_combo_g",
        "chunk_fn": lambda t: chunk_text(t, 600, 100),
        "chunk_desc": "recursive / size=600 / overlap=100",
        "top_k": 5,
        "max_distance": 0.65,
        "rerank": False,
        "prompt_style": "cot",
        "use_hybrid": False,
        "use_expand": True,
    },
]

COMBO_MAP = {c["id"]: c for c in COMBOS}

# ─── 輸出目錄 ──────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = ROOT / "experiments"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _run_query(combo: dict, question: str) -> dict:
    """執行單次 retrieve + LLM，回傳結構化結果 dict。支援 Hybrid Search 和 Query Expansion。"""
    use_hybrid = combo.get("use_hybrid", False)
    use_expand = combo.get("use_expand", False)

    extra_queries: list[str] | None = None
    if use_expand:
        try:
            all_q = expand_query(question, model=None)
            extra_queries = all_q[1:]
        except Exception:
            extra_queries = None

    if use_hybrid:
        context, cites = hybrid_retrieve(
            question,
            top_k=combo["top_k"],
            max_distance=combo["max_distance"],
            collection_name=combo["collection"],
            extra_queries=extra_queries,
        )
    else:
        context, cites = retrieve_context(
            question,
            top_k=combo["top_k"],
            max_distance=combo["max_distance"],
            rerank=combo["rerank"],
            collection_name=combo["collection"],
            extra_queries=extra_queries,
        )

    if not context.strip():
        return {
            "question": question,
            "context": "",
            "cites": [],
            "answer": "【知識庫中未找到相關資料，無法生成回答。】",
            "error": "no_context",
        }
    messages = build_messages(question, context, [], prompt_style=combo["prompt_style"])
    try:
        resp = litellm_chat_completion(model=None, messages=messages, temperature=0.2)
        answer = resp.choices[0].message.content or ""
    except Exception as exc:
        answer = f"【LLM 呼叫失敗：{exc}】"
    return {
        "question": question,
        "context": context,
        "cites": cites,
        "answer": answer,
        "error": None,
    }


def _write_question_result(combo_dir: Path, q_idx: int, result: dict, combo: dict) -> None:
    """將單題結果寫入 q{N}.md。"""
    path = combo_dir / f"q{q_idx + 1}.md"
    cites = result["cites"]
    avg_dist = (sum(c["distance"] for c in cites) / len(cites)) if cites else float("nan")

    cite_lines = "\n".join(
        f"[{i+1}] `{c['source_file']}` chunk {c['chunk_index']}"
        + (f" [{c['section_title']}]" if c.get("section_title") else "")
        + f" (距離: {c['distance']:.3f})"
        for i, c in enumerate(cites)
    )

    md = dedent(f"""\
        # Q{q_idx + 1}: {result['question']}

        ## 實驗設定

        | 項目 | 值 |
        |---|---|
        | Combo | {combo['id'].upper()} — {combo['name']} |
        | Chunking | {combo['chunk_desc']} |
        | top-k | {combo['top_k']} |
        | max_distance | {combo['max_distance']} |
        | Rerank | {'是' if combo['rerank'] else '否'} |
        | Prompt 風格 | {combo['prompt_style']} |
        | Hybrid Search | {'是 (BM25+Vector+RRF)' if combo.get('use_hybrid') else '否'} |
        | Query Expansion | {'是' if combo.get('use_expand') else '否'} |

        ## 檢索結果（{len(cites)} 個 chunks，平均距離 {avg_dist:.3f}）

        {cite_lines if cite_lines else '（無相關 chunks）'}

        ## LLM 回答

        {result['answer']}

        ---
        *generated by experiment_runner.py at {_ts()}*
    """)
    path.write_text(md, encoding="utf-8")


def _write_combo_config(combo_dir: Path, combo: dict) -> None:
    path = combo_dir / "config.md"
    path.write_text(
        dedent(f"""\
            # {combo['name']}

            {combo['desc']}

            | 設定 | 值 |
            |---|---|
            | Chunking | {combo['chunk_desc']} |
            | top-k | {combo['top_k']} |
            | max_distance | {combo['max_distance']} |
            | Rerank | {'是' if combo['rerank'] else '否'} |
            | Prompt 風格 | {combo['prompt_style']} |
            | Chroma collection | `{combo['collection']}` |

            *generated at {_ts()}*
        """),
        encoding="utf-8",
    )


def run_combo(combo: dict, skip_rebuild: bool = False) -> list[dict]:
    combo_id = combo["id"].upper()
    print(f"\n{'='*60}")
    print(f"  {combo_id}: {combo['name']}")
    print(f"{'='*60}")

    combo_dir = EXPERIMENTS_DIR / f"combo_{combo['id']}"
    combo_dir.mkdir(parents=True, exist_ok=True)
    _write_combo_config(combo_dir, combo)

    if not skip_rebuild:
        print(f"  [1/2] 重建索引（collection: {combo['collection']}）...", end=" ", flush=True)
        t0 = time.time()
        rebuild_processed_and_index(
            chunk_fn=combo["chunk_fn"],
            collection_name=combo["collection"],
        )
        print(f"完成 ({time.time()-t0:.1f}s)")
    else:
        print(f"  [1/2] 跳過 rebuild（--skip-rebuild）")

    print(f"  [2/2] 執行 {len(QUESTIONS)} 題問答...")
    results: list[dict] = []
    for i, q in enumerate(QUESTIONS):
        print(f"    Q{i+1}: {q[:50]}...", end=" ", flush=True)
        result = _run_query(combo, q)
        _write_question_result(combo_dir, i, result, combo)
        status = "OK" if not result["error"] else f"WARN({result['error']})"
        print(status)
        results.append(result)

    return results


def generate_summary(all_results: dict[str, list[dict]]) -> None:
    """產生 experiments/summary.md 橫向比較表。"""
    path = EXPERIMENTS_DIR / "summary.md"
    ran_combos = [COMBO_MAP[cid] for cid in sorted(all_results.keys())]

    # 設定對照表
    header = "| 面向 | " + " | ".join(f"Combo {c['id'].upper()}" for c in ran_combos) + " |"
    sep    = "|---|" + "---|" * len(ran_combos)
    rows_config = [
        ("Chunk 策略",    [c["chunk_desc"] for c in ran_combos]),
        ("top-k",          [str(c["top_k"]) for c in ran_combos]),
        ("max_dist",       [str(c["max_distance"]) for c in ran_combos]),
        ("Rerank",         ["是" if c["rerank"] else "否" for c in ran_combos]),
        ("Prompt",         [c["prompt_style"] for c in ran_combos]),
        ("Hybrid Search",  ["是" if c.get("use_hybrid") else "否" for c in ran_combos]),
        ("Query Expand",   ["是" if c.get("use_expand") else "否" for c in ran_combos]),
    ]
    config_table = "\n".join(
        [header, sep]
        + [f"| {label} | " + " | ".join(vals) + " |" for label, vals in rows_config]
    )

    # 每題橫向比較
    question_sections: list[str] = []
    for q_idx, question in enumerate(QUESTIONS):
        section_lines = [f"## Q{q_idx+1}: {question}\n"]
        for combo in ran_combos:
            cid = combo["id"]
            if cid not in all_results:
                continue
            r = all_results[cid][q_idx]
            cites = r["cites"]
            avg_dist = (sum(c.get("distance", 0) for c in cites) / len(cites)) if cites else float("nan")
            section_lines.append(
                f"### Combo {cid.upper()} — {combo['name']}\n"
                f"- 取回 chunks: {len(cites)}，平均距離: {avg_dist:.3f}\n"
                f"- 引用: "
                + (", ".join(f"`{c['source_file']}` chunk {c['chunk_index']}" for c in cites[:3])
                   + ("..." if len(cites) > 3 else ""))
                + "\n\n"
                + r["answer"]
                + "\n"
            )
        question_sections.append("\n".join(section_lines))

    content = dedent(f"""\
        # RAG 七組實驗比較結果（A–G）

        *生成時間: {_ts()}*

        ## 組合設定對照

        {config_table}

        ---

    """) + "\n\n---\n\n".join(question_sections)

    path.write_text(content, encoding="utf-8")
    print(f"\n[summary] 已寫入: {path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG 七組實驗比較跑分（A-G）")
    p.add_argument(
        "--combos",
        nargs="+",
        choices=list(COMBO_MAP.keys()),
        default=list(COMBO_MAP.keys()),
        metavar="COMBO_ID",
        help="要跑的 combo（a b c d e f g），預設全部",
    )
    p.add_argument(
        "--skip-rebuild",
        action="store_true",
        help="跳過 rebuild，直接跑問答（適合只改 prompt 時快速重測）",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    selected = [COMBO_MAP[cid] for cid in args.combos]
    all_results: dict[str, list[dict]] = {}

    for combo in selected:
        try:
            results = run_combo(combo, skip_rebuild=args.skip_rebuild)
            all_results[combo["id"]] = results
        except Exception as exc:
            print(f"\n[ERROR] Combo {combo['id'].upper()} 失敗: {exc}", file=sys.stderr)

    if all_results:
        generate_summary(all_results)

    print("\n全部完成。結果存放於 experiments/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
