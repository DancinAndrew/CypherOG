# RAG 5 組實驗比較（A、B、C、D、E）

*生成時間: 2026-11-10 01:08 UTC；來源: 5 個 combo_*/results.json*

距離為 cosine distance（0=完全相同）。「與原問題距離」一律以原始問題計算；
retriever 回報的距離在 Query Expansion 時可能是相對於子查詢。

## 組合設定對照

| 面向 | Combo A | Combo B | Combo C | Combo D | Combo E |
|---|---|---|---|---|---|
| Chunk 策略 | naive / size=500 / overlap=0 | recursive / size=600 / overlap=100 | recursive / size=1200 / overlap=200 | recursive / size=300 / overlap=50 | recursive / size=600 / overlap=100 |
| top-k | 5 | 5 | 3 | 10 | 5 |
| max_dist | 1.0 | 0.65 | 0.65 | 0.5 | 0.65 |
| 距離截斷 | 否 | 否 | 否 | 5 | 否 |
| Prompt | weak | weak | strong | strong | cot |
| Hybrid Search | 否 | 否 | 否 | 否 | 否 |
| Query Expand | 否 | 否 | 否 | 否 | 否 |
| LLM 模型 | gemini-2.5-flash | gemini-2.5-flash | gemini-2.5-flash | gemini-2.5-flash | gemini-2.5-flash |
| 模式 | 僅檢索 | 僅檢索 | 僅檢索 | 僅檢索 | 僅檢索 |
| 狀態 | ok | ok | ok | ok | ok |
| 產生時間 | 2026-11-10T01:08:33+00:00 | 2026-11-10T01:08:38+00:00 | 2026-11-10T01:08:40+00:00 | 2026-11-10T01:08:50+00:00 | 2026-11-10T01:08:57+00:00 |

## 與原問題的平均距離

| Combo | 與原問題平均距離（全部題目的 chunks） |
|---|---|
| A | 0.399 |
| B | 0.386 |
| C | 0.384 |
| D | 0.359 |
| E | 0.386 |

---
## Q1: What is Popping and who created it?

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| A | 5 | 0.387 | 0.387 |
| B | 5 | 0.355 | 0.355 |
| C | 3 | 0.377 | 0.377 |
| D | 5 | 0.307 | 0.307 |
| E | 5 | 0.355 | 0.355 |

### Combo A — Combo A — Baseline（字元暴力切）
- 取回 chunks: 5，與原問題平均距離: 0.387（retriever 平均距離: 0.387）
- 引用: `POPPING  societydance.txt` chunk 4 (dense), `What Is Popping (Popping Dance)  STEEZY Blog.txt` chunk 1 (dense), `POPPING  societydance.txt` chunk 6 (dense)...

（僅檢索，未生成回答）

### Combo B — Combo B — 段落感知 + 距離過濾
- 取回 chunks: 5，與原問題平均距離: 0.355（retriever 平均距離: 0.355）
- 引用: `POPPING  societydance.txt` chunk 4 (dense), `History initiative to document the origins of ‘popping’ - Fresno State News.txt` chunk 0 (dense), `POPPING  societydance.txt` chunk 6 (dense)...

（僅檢索，未生成回答）

### Combo C — Combo C — 大 chunk + 強 prompt
- 取回 chunks: 3，與原問題平均距離: 0.377（retriever 平均距離: 0.377）
- 引用: `POPPING  societydance.txt` chunk 2 (dense), `What Is Popping (Popping Dance)  STEEZY Blog.txt` chunk 0 (dense), `Straight Outta Fresno How the Popping Dance Movement Empowered Youth of Color – Tropics of Meta.txt` chunk 10 (dense)

（僅檢索，未生成回答）

### Combo D — Combo D — 小 chunk + top-10 距離截斷
- 取回 chunks: 5，與原問題平均距離: 0.307（retriever 平均距離: 0.307）
- 引用: `POPPING  societydance.txt` chunk 11 (dense), `POPPING  societydance.txt` chunk 12 (dense), `Popping.txt` chunk 1 (dense)...

（僅檢索，未生成回答）

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.355（retriever 平均距離: 0.355）
- 引用: `POPPING  societydance.txt` chunk 4 (dense), `History initiative to document the origins of ‘popping’ - Fresno State News.txt` chunk 0 (dense), `POPPING  societydance.txt` chunk 6 (dense)...

（僅檢索，未生成回答）


---

## Q2: Electric Boogaloos 的成員有哪些？他們各自的貢獻是什麼？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| A | 5 | 0.358 | 0.358 |
| B | 5 | 0.358 | 0.358 |
| C | 3 | 0.362 | 0.362 |
| D | 5 | 0.325 | 0.325 |
| E | 5 | 0.358 | 0.358 |

### Combo A — Combo A — Baseline（字元暴力切）
- 取回 chunks: 5，與原問題平均距離: 0.358（retriever 平均距離: 0.358）
- 引用: `Electric boogaloo (dance) - Wikipedia.txt` chunk 10 (dense), `POPPING  societydance.txt` chunk 11 (dense), `Popin' Pete - Wikipedia.txt` chunk 3 (dense)...

（僅檢索，未生成回答）

### Combo B — Combo B — 段落感知 + 距離過濾
- 取回 chunks: 5，與原問題平均距離: 0.358（retriever 平均距離: 0.358）
- 引用: `Popin' Pete - Wikipedia.txt` chunk 3 (dense), `Popin' Pete - Wikipedia.txt` chunk 6 (dense), `Popin' Pete - Wikipedia.txt` chunk 4 (dense)...

（僅檢索，未生成回答）

### Combo C — Combo C — 大 chunk + 強 prompt
- 取回 chunks: 3，與原問題平均距離: 0.362（retriever 平均距離: 0.362）
- 引用: `Electric boogaloo (dance) - Wikipedia.txt` chunk 5 (dense), `Popin' Pete - Wikipedia.txt` chunk 2 (dense), `Suga Pop - Wikipedia.txt` chunk 1 (dense)

（僅檢索，未生成回答）

### Combo D — Combo D — 小 chunk + top-10 距離截斷
- 取回 chunks: 5，與原問題平均距離: 0.325（retriever 平均距離: 0.325）
- 引用: `Popin' Pete - Wikipedia.txt` chunk 6 (dense), `Popin' Pete - Wikipedia.txt` chunk 10 (dense), `A Little Dance History: Popping, Boogaloo, Robot, Strutting, Fillmore, etc.txt` chunk 14 (dense)...

（僅檢索，未生成回答）

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.358（retriever 平均距離: 0.358）
- 引用: `Popin' Pete - Wikipedia.txt` chunk 3 (dense), `Popin' Pete - Wikipedia.txt` chunk 6 (dense), `Popin' Pete - Wikipedia.txt` chunk 4 (dense)...

（僅檢索，未生成回答）


---

## Q3: Popping 和 Boogaloo 有什麼差別？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| A | 5 | 0.378 | 0.378 |
| B | 5 | 0.388 | 0.388 |
| C | 3 | 0.365 | 0.365 |
| D | 5 | 0.355 | 0.355 |
| E | 5 | 0.388 | 0.388 |

### Combo A — Combo A — Baseline（字元暴力切）
- 取回 chunks: 5，與原問題平均距離: 0.378（retriever 平均距離: 0.378）
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `popping-terminology-snippet.txt` chunk 0 (dense), `POPPING  societydance.txt` chunk 3 (dense)...

（僅檢索，未生成回答）

### Combo B — Combo B — 段落感知 + 距離過濾
- 取回 chunks: 5，與原問題平均距離: 0.388（retriever 平均距離: 0.388）
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `popping-terminology-snippet.txt` chunk 0 (dense), `POPPING  societydance.txt` chunk 3 (dense)...

（僅檢索，未生成回答）

### Combo C — Combo C — 大 chunk + 強 prompt
- 取回 chunks: 3，與原問題平均距離: 0.365（retriever 平均距離: 0.365）
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `Salah (dancer) - Wikipedia.txt` chunk 2 (dense), `popping-terminology-snippet.txt` chunk 0 (dense)

（僅檢索，未生成回答）

### Combo D — Combo D — 小 chunk + top-10 距離截斷
- 取回 chunks: 5，與原問題平均距離: 0.355（retriever 平均距離: 0.355）
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 7 (dense), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `Popin' Pete - Wikipedia.txt` chunk 9 (dense)...

（僅檢索，未生成回答）

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.388（retriever 平均距離: 0.388）
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `popping-terminology-snippet.txt` chunk 0 (dense), `POPPING  societydance.txt` chunk 3 (dense)...

（僅檢索，未生成回答）


---

## Q4: Juste Debout 比賽的規則和歷史？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| A | 5 | 0.468 | 0.468 |
| B | 5 | 0.474 | 0.474 |
| C | 3 | 0.450 | 0.450 |
| D | 5 | 0.450 | 0.450 |
| E | 5 | 0.474 | 0.474 |

### Combo A — Combo A — Baseline（字元暴力切）
- 取回 chunks: 5，與原問題平均距離: 0.468（retriever 平均距離: 0.468）
- 引用: `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 8 (dense), `Guarato.Dance.Articulated.txt` chunk 67 (dense), `Salah (dancer) - Wikipedia.txt` chunk 38 (dense)...

（僅檢索，未生成回答）

### Combo B — Combo B — 段落感知 + 距離過濾
- 取回 chunks: 5，與原問題平均距離: 0.474（retriever 平均距離: 0.474）
- 引用: `Salah (dancer) - Wikipedia.txt` chunk 66 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 22 (dense), `Salah (dancer) - Wikipedia.txt` chunk 19 (dense)...

（僅檢索，未生成回答）

### Combo C — Combo C — 大 chunk + 強 prompt
- 取回 chunks: 3，與原問題平均距離: 0.450（retriever 平均距離: 0.450）
- 引用: `Popin' Pete - Wikipedia.txt` chunk 12 (dense), `Salah (dancer) - Wikipedia.txt` chunk 27 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 10 (dense)

（僅檢索，未生成回答）

### Combo D — Combo D — 小 chunk + top-10 距離截斷
- 取回 chunks: 5，與原問題平均距離: 0.450（retriever 平均距離: 0.450）
- 引用: `Salah (dancer) - Wikipedia.txt` chunk 49 (dense), `Guarato.Dance.Articulated.txt` chunk 43 (dense), `Salah (dancer) - Wikipedia.txt` chunk 34 (dense)...

（僅檢索，未生成回答）

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.474（retriever 平均距離: 0.474）
- 引用: `Salah (dancer) - Wikipedia.txt` chunk 66 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 22 (dense), `Salah (dancer) - Wikipedia.txt` chunk 19 (dense)...

（僅檢索，未生成回答）


---

## Q5: 如何判斷一場 Popping battle 的勝負？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| A | 5 | 0.405 | 0.405 |
| B | 5 | 0.355 | 0.355 |
| C | 3 | 0.368 | 0.368 |
| D | 5 | 0.358 | 0.358 |
| E | 5 | 0.355 | 0.355 |

### Combo A — Combo A — Baseline（字元暴力切）
- 取回 chunks: 5，與原問題平均距離: 0.405（retriever 平均距離: 0.405）
- 引用: `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 2 (dense), `popping-culture-notes.txt` chunk 0 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 12 (dense)...

（僅檢索，未生成回答）

### Combo B — Combo B — 段落感知 + 距離過濾
- 取回 chunks: 5，與原問題平均距離: 0.355（retriever 平均距離: 0.355）
- 引用: `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 0 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 15 (dense)...

（僅檢索，未生成回答）

### Combo C — Combo C — 大 chunk + 強 prompt
- 取回 chunks: 3，與原問題平均距離: 0.368（retriever 平均距離: 0.368）
- 引用: `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 1 (dense), `popping-culture-notes.txt` chunk 0 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 9 (dense)

（僅檢索，未生成回答）

### Combo D — Combo D — 小 chunk + top-10 距離截斷
- 取回 chunks: 5，與原問題平均距離: 0.358（retriever 平均距離: 0.358）
- 引用: `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 0 (dense), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 5 (dense), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 6 (dense)...

（僅檢索，未生成回答）

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.355（retriever 平均距離: 0.355）
- 引用: `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 0 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 15 (dense)...

（僅檢索，未生成回答）
