# Combo G — CoT + Query Expansion（多角度子查詢）

在 Combo E 基礎上加入 Query Expansion：用 LLM 將查詢改寫為中英文多角度子查詢，各子查詢結果去重並保留最小距離。

| 設定 | 值 |
|---|---|
| Chunking | recursive / size=600 / overlap=100 |
| top-k | 5 |
| max_distance | 0.65 |
| 距離截斷（truncate_to） | 否 |
| Prompt 風格 | cot |
| Hybrid Search | 否 |
| Query Expansion | 是 |
| LLM 模型 | gpt-4.1-mini-2025-04-14 |
| Chroma collection | `cypher_og_current_combo_g` |

*generated at 2026-11-10 01:15 UTC*
