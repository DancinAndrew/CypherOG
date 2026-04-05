# Combo B — 段落感知 + 距離過濾

升級 chunking 與 retrieval，保持弱 prompt，單獨觀察前兩項改善效果。

| 設定 | 值 |
|---|---|
| Chunking | recursive / size=600 / overlap=100 |
| top-k | 5 |
| max_distance | 0.65 |
| Rerank | 否 |
| Prompt 風格 | weak |
| Chroma collection | `hw3_rag_combo_b` |

*generated at 2026-04-05 08:28 UTC*
