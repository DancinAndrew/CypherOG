# Combo D — 小 chunk + top-10 距離截斷

小 chunk + 嚴格距離過濾；先取距離最近的 10 個再截斷為前 5 個。沒有 reranker 模型，結果等同直接取 top-5。

| 設定 | 值 |
|---|---|
| Chunking | recursive / size=300 / overlap=50 |
| top-k | 10 |
| max_distance | 0.5 |
| 距離截斷（truncate_to） | 前 5 個 |
| Prompt 風格 | strong |
| Hybrid Search | 否 |
| Query Expansion | 否 |
| LLM 模型 | gemini-2.5-flash |
| Chroma collection | `cypher_og_round1_combo_d` |

*generated at 2026-11-10 01:08 UTC*
