# Combo B — 段落感知 + 距離過濾

升級 chunking 與 retrieval，保持弱 prompt，單獨觀察前兩項改善效果。

| 設定 | 值 |
|---|---|
| Chunking | recursive / size=600 / overlap=100 |
| top-k | 5 |
| max_distance | 0.65 |
| 距離截斷（truncate_to） | 否 |
| Prompt 風格 | weak |
| Hybrid Search | 否 |
| Query Expansion | 否 |
| LLM 模型 | gemini-2.5-flash |
| Chroma collection | `cypher_og_round1_combo_b` |

*generated at 2026-11-10 01:08 UTC*
