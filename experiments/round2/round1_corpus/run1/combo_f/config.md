# Combo F — CoT + Hybrid Search（BM25+Vector+RRF）

在 Combo E 基礎上加入 BM25 詞彙檢索，與向量結果以標準 RRF 融合；距離門檻只作用於向量通道，只被 BM25 找到的 chunk 也能入選。

| 設定 | 值 |
|---|---|
| Chunking | recursive / size=600 / overlap=100 |
| top-k | 5 |
| max_distance | 0.65 |
| 距離截斷（truncate_to） | 否 |
| Prompt 風格 | cot |
| Hybrid Search | 是 (BM25+Vector+RRF) |
| Query Expansion | 否 |
| LLM 模型 | gpt-4.1-mini-2025-04-14 |
| Chroma collection | `cypher_og_round1_combo_f` |

*generated at 2026-11-10 01:10 UTC*
