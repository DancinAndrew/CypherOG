# Combo E — CoT + 結構化 context

與 B 相同的 chunking/retrieval，但加 Chain-of-Thought prompt，強制 LLM 先列 facts 再綜合。

| 設定 | 值 |
|---|---|
| Chunking | recursive / size=600 / overlap=100 |
| top-k | 5 |
| max_distance | 0.65 |
| Rerank | 否 |
| Prompt 風格 | cot |
| Chroma collection | `hw3_rag_combo_e` |

*generated at 2026-04-05 08:29 UTC*
