# Combo C — 大 chunk + 強 prompt

大 chunk 保留完整語境，搭配強制規則 prompt 與 XML 標籤，觀察幻覺減少效果。

| 設定 | 值 |
|---|---|
| Chunking | recursive / size=1200 / overlap=200 |
| top-k | 3 |
| max_distance | 0.65 |
| 距離截斷（truncate_to） | 否 |
| Prompt 風格 | strong |
| Hybrid Search | 否 |
| Query Expansion | 否 |
| LLM 模型 | gemini-2.5-flash |
| Chroma collection | `cypher_og_round1_combo_c` |

*generated at 2026-11-10 01:08 UTC*
