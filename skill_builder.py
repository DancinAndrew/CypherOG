#!/usr/bin/env python3
"""依固定問題做檢索，透過 LiteLLM 彙整為 skill.md（符合 Agent Skill 格式規範）。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from textwrap import dedent

from dotenv import load_dotenv

from rag_common import DATA_RAW, embed_texts, get_chroma_collection, litellm_chat_completion

load_dotenv()

DEFAULT_MODEL = os.environ.get("LITELLM_MODEL", "gemini-2.5-flash")

# 8 個全域問題，覆蓋 skill.md 所有必要章節
DEFAULT_QUESTIONS = [
    # → Overview + Core Concepts
    "這個知識庫的核心主題範圍是什麼？涵蓋哪些主要概念和子主題？請列出最重要的 10-15 個核心概念（例如 Popping、Boogaloo、Waving、Tutting、Locking 等）並簡要說明每個概念。",
    # → Key Trends
    "這個領域目前最重要的發展方向或趨勢為何？包括比賽形式演變（如 Red Bull Dance Your Style、HHI）、線上教學平台崛起（如 Funk in Focus Online、STEEZY）、跨文化傳播、以及當代舞者對傳統的詮釋方式。",
    # → Key Entities — 人物
    "知識庫中提到的關鍵人物有哪些？請分類列出：創始人與先驅（如 Boogaloo Sam、Popin' Pete）、重要傳承者（如 Mr. Wiggles、Suga Pop）、當代舞者（如 Marquese Scott、Salah）及教育者，並說明每位的主要貢獻。",
    # → Key Entities — 組織與賽事
    "知識庫中提到的重要組織、團體、賽事和平台有哪些（如 Electric Boogaloos、Infinite Force、HHI、Juste Debout、Red Bull Dance Your Style）？各自的歷史背景、規則、特色是什麼？",
    # → Methodology & Best Practices
    "這個領域中被廣泛接受的訓練方法、比賽評判標準或文化傳承原則有哪些？例如：Popping battle 如何評分？Funk 音樂與舞蹈的關係？基本功（Hit、Isolation、Musicality）的重要性？",
    # → Example Q&A
    "請回答以下 5 個問題（每個都要給出詳細答案）：1) Popping 是誰發明的？在哪裡？2) Boogaloo 和 Popping 的技術差異是什麼？3) Electric Boogaloos 的成員有哪些人？4) Juste Debout 比賽的規則和歷史是什麼？5) 如何判斷一場 Popping battle 的勝負？",
    # → Knowledge Gaps & Limitations
    "這個知識庫有哪些已知的限制或知識缺口？哪些子主題未被充分覆蓋（例如中文社群、亞洲 Popping 圈、最新賽事結果）？資料的時效性如何？有哪些來源可能存在偏差或不完整？",
    # → Key Trends (supplement)
    "Funk 音樂對 Popping 舞蹈文化的影響是什麼？Funk 的歷史、主要藝術家（如 James Brown、Parliament-Funkadelic）和文化意義，以及它如何塑造了 Popping 的節奏感和表演風格？",
]


def retrieve_for_question(q: str, top_k: int = 8) -> str:
    """為單一問題做向量檢索，回傳格式化的 context 字串。"""
    coll = get_chroma_collection()
    q_emb = embed_texts([q])[0]
    res = coll.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    parts: list[str] = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        src = meta.get("source_file", "?")
        idx = meta.get("chunk_index", i)
        section = meta.get("section_title", "")
        label = f" ({section})" if section else ""
        parts.append(f"[{src} #{idx}{label}]\n{doc}")
    return "\n\n".join(parts)


def _collect_source_references() -> str:
    """直接從 data/raw/ 掃描，確定性生成來源清單（不依賴 LLM，消除幻覺）。"""
    if not DATA_RAW.exists():
        return "（資料目錄不存在）"
    entries: list[str] = []
    type_map = {".md": "Markdown", ".txt": "Text", ".pdf": "PDF", ".markdown": "Markdown"}
    for p in sorted(DATA_RAW.iterdir()):
        if p.is_file() and p.suffix.lower() in type_map:
            fmt = type_map.get(p.suffix.lower(), p.suffix.upper())
            entries.append(f"| `{p.name}` | {fmt} | 公開資料 / 個人著作 |")
    if not entries:
        return "（無資料檔案）"
    header = "| 檔案名稱 | 格式 | 授權 |\n|---|---|---|\n"
    return header + "\n".join(entries)


def _count_sources() -> int:
    if not DATA_RAW.exists():
        return 0
    return sum(
        1 for p in DATA_RAW.iterdir()
        if p.is_file() and p.suffix.lower() in {".md", ".txt", ".pdf", ".markdown"}
    )


def build_skill_content(sections: list[tuple[str, str]], model: str, num_sources: int, date: str) -> str:
    """兩階段生成：先彙整所有 context，再強制輸出完整 skill.md 格式。"""
    blob = "\n\n---\n\n".join(
        f"### 問題: {q}\n\n#### 檢索結果\n{ctx}" for q, ctx in sections
    )

    system_prompt = dedent("""\
        你是一位頂尖知識工程師，正在將 RAG 知識庫濃縮成 AI Agent 可直接使用的 SKILL 文件。
        你的目標是讓這份文件「讀起來像領域專家整理的入門指南」，而非 LLM 的隨機輸出。

        規則：
        1. 只能根據下方「多輪檢索結果」撰寫，禁止虛構任何細節
        2. 不確定或缺乏依據的內容，明確標註「資料未涵蓋」
        3. 知識密度要高，每句話都應承載具體的資訊（人名、地點、年代、技術術語）
        4. 嚴格按照下方指定格式輸出，每個章節都必須存在，不可省略
        5. 使用繁體中文，專有名詞（英文術語、人名）保留原文
    """)

    user_prompt = dedent(f"""\
        根據以下多輪 RAG 檢索結果，輸出一份完整的 Agent SKILL 文件。
        **必須嚴格按照以下 Markdown 格式輸出，所有章節標題不可更改：**

        ---
        # Skill: Popping & Funk Style 街舞文化

        ## Metadata
        - **知識領域**：Popping 舞蹈文化與 Funk Style 街舞
        - **資料來源數量**：{num_sources} 份文件
        - **最後更新時間**：{date}
        - **適用 Agent 類型**：街舞文化領域問答機器人 / 舞蹈教育助手 / Battle 裁判助理

        ## Overview
        （在此撰寫 150-200 字，說明此 Skill 的核心知識範疇：涵蓋 Popping 的起源、流派、重要人物、賽事體系、Funk 音樂文化背景，以及此知識庫能回答的問題類型與能力邊界。）

        ## Core Concepts
        （在此條列 10-15 個最重要的概念，每個概念附 1-2 句具體說明，包含技術特徵或歷史背景。例如：Popping、Boogaloo、Waving、Tutting、Animation、Locking、Hit/Pop、Musicality、Cypher、Battle、Fresno、Funk 音樂等。）

        ## Key Trends
        （在此條列 5-8 個重要發展方向，說明 Popping 文化的演變，例如：線上教學平台崛起、全球化賽事體系、與當代音樂風格結合、亞洲 Popping 社群成長等。）

        ## Key Entities
        （在此分類條列重要實體）

        ### 創始人與先驅
        （列出電 Boogaloo Sam、Popin' Pete 等創始人物及其貢獻）

        ### 當代重要舞者
        （列出 Marquese Scott、Salah 等當代舞者及其風格特色）

        ### 重要組織與團體
        （列出 Electric Boogaloos、Infinite Force 等組織及歷史背景）

        ### 主要賽事與平台
        （列出 HHI、Juste Debout、Red Bull Dance Your Style、STEEZY 等）

        ## Methodology & Best Practices
        （在此說明被廣泛接受的訓練方法、比賽評判標準（musicality、originality、foundation、technique、execution、floor work、showmanship、creativity、costume）、文化傳承原則，以及 Funk 音樂與舞蹈的關係。）

        ## Knowledge Gaps & Limitations
        （在此誠實說明此 Skill 的知識邊界：資料截止時間、語言偏差（85% 英文）、未覆蓋的子主題（如亞洲 Popping 社群、最新 2025 賽事）、維基百科內容的可靠性限制等。）

        ## Example Q&A
        （在此列出 5 組代表性問答，每組包含完整問題和詳細答案，展示此 Skill 的問答能力範圍。）

        **Q1:** Popping 是誰發明的？在哪裡起源？
        **A1:** （根據檢索結果詳細回答）

        **Q2:** Boogaloo 和 Popping 在技術上有什麼不同？
        **A2:** （根據檢索結果詳細回答）

        **Q3:** Electric Boogaloos 的主要成員有哪些人？
        **A3:** （根據檢索結果詳細回答）

        **Q4:** Juste Debout 比賽的規則和歷史是什麼？
        **A4:** （根據檢索結果詳細回答）

        **Q5:** 如何評判一場 Popping battle 的勝負？
        **A5:** （根據檢索結果詳細回答）

        ## Source References

        （此章節由系統自動生成，列出所有資料來源。）

        ---

        以下是多輪 RAG 檢索結果，請根據這些內容填寫上述格式中的括號部分：

        {blob}
    """)

    resp = litellm_chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="從向量庫萃取知識並生成符合 Agent Skill 格式規範的 skill.md",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("skill.md"),
        help="輸出路徑（預設：skill.md）",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="LiteLLM 模型名稱（預設：gemini-2.5-flash）",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="每個問題的檢索筆數（預設：8）",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out: Path = args.output
    try:
        print(f"[skill_builder] 開始多輪 RAG 檢索（{len(DEFAULT_QUESTIONS)} 個問題，top-k={args.top_k}）...")
        sections: list[tuple[str, str]] = []
        for i, q in enumerate(DEFAULT_QUESTIONS, 1):
            print(f"  [{i}/{len(DEFAULT_QUESTIONS)}] {q[:60]}...")
            ctx = retrieve_for_question(q, top_k=args.top_k)
            sections.append((q, ctx or "（無檢索結果）"))

        num_sources = _count_sources()
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        print("[skill_builder] 呼叫 LLM 生成 skill.md...")
        body = build_skill_content(sections, args.model, num_sources, date)

        # 確定性附加 Source References（不依賴 LLM，消除幻覺）
        source_table = _collect_source_references()
        # 若 LLM 已生成 Source References 章節，替換其內容；否則附加
        source_section = f"\n## Source References\n\n{source_table}\n"
        if "## Source References" in body:
            # 取出 ## Source References 之前的部分，重新附加正確的來源表
            body = body[:body.index("## Source References")] + source_section.lstrip("\n")
        else:
            body = body.rstrip() + "\n" + source_section

        final = f"<!-- generated by skill_builder.py at {ts} -->\n\n{body}\n"
        out.write_text(final, encoding="utf-8")

    except Exception as e:
        print(f"[skill_builder] 錯誤: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    char_count = len(final)
    print(f"[skill_builder] 已寫入: {out.resolve()} ({char_count:,} 字元)")
    if char_count < 500:
        print("[skill_builder] 警告：字元數不足 500，請檢查 LLM 輸出。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
