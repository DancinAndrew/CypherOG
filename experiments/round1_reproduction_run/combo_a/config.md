# Combo A — Baseline（字元暴力切）

最差基準線：純字元切割、無距離過濾、弱 prompt。

| 設定 | 值 |
|---|---|
| Chunking | naive / size=500 / overlap=0 |
| top-k | 5 |
| max_distance | 1.0 |
| 距離截斷（truncate_to） | 否 |
| Prompt 風格 | weak |
| Hybrid Search | 否 |
| Query Expansion | 否 |
| LLM 模型 | gemini-2.5-flash |
| Chroma collection | `cypher_og_round1_combo_a` |

*generated at 2026-11-10 01:08 UTC*
