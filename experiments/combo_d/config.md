# Combo D — 小 chunk + 高 top-k + rerank

小而精確的 chunk + 寬搜窄篩（top-10→rerank→top-5）+ 嚴格距離過濾。

| 設定 | 值 |
|---|---|
| Chunking | recursive / size=300 / overlap=50 |
| top-k | 10 |
| max_distance | 0.5 |
| Rerank | 是 |
| Prompt 風格 | strong |
| Chroma collection | `hw3_rag_combo_d` |

*generated at 2026-04-05 08:29 UTC*
