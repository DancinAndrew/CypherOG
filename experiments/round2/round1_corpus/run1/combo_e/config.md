# Combo E — CoT + 結構化 context

與 B 相同的 chunking/retrieval，但加 Chain-of-Thought prompt，強制 LLM 先列 facts 再綜合。

| 設定 | 值 |
|---|---|
| Chunking | recursive / size=600 / overlap=100 |
| top-k | 5 |
| max_distance | 0.65 |
| 距離截斷（truncate_to） | 否 |
| Prompt 風格 | cot |
| Hybrid Search | 否 |
| Query Expansion | 否 |
| LLM 模型 | gpt-4.1-mini-2025-04-14 |
| Chroma collection | `cypher_og_round1_combo_e` |

*generated at 2026-11-10 01:09 UTC*
