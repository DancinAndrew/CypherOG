# RAG 3 組實驗比較（E、F、G）

*生成時間: 2026-11-10 01:15 UTC；來源: 3 個 combo_*/results.json*

距離為 cosine distance（0=完全相同）。「與原問題距離」一律以原始問題計算；
retriever 回報的距離在 Query Expansion 時可能是相對於子查詢。

## 組合設定對照

| 面向 | Combo E | Combo F | Combo G |
|---|---|---|---|
| Chunk 策略 | recursive / size=600 / overlap=100 | recursive / size=600 / overlap=100 | recursive / size=600 / overlap=100 |
| top-k | 5 | 5 | 5 |
| max_dist | 0.65 | 0.65 | 0.65 |
| 距離截斷 | 否 | 否 | 否 |
| Prompt | cot | cot | cot |
| Hybrid Search | 否 | 是 | 否 |
| Query Expand | 否 | 否 | 是 |
| LLM 模型 | gpt-4.1-mini-2025-04-14 | gpt-4.1-mini-2025-04-14 | gpt-4.1-mini-2025-04-14 |
| 模式 | 檢索 + 生成 | 檢索 + 生成 | 檢索 + 生成 |
| 狀態 | ok | ok | ok |
| 產生時間 | 2026-11-10T01:14:31+00:00 | 2026-11-10T01:15:01+00:00 | 2026-11-10T01:15:30+00:00 |

## 與原問題的平均距離

| Combo | 與原問題平均距離（全部題目的 chunks） |
|---|---|
| E | 0.379 |
| F | 0.460 |
| G | 0.466 |

---
## Q1: What is Popping and who created it?

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.355 | 0.355 |
| F | 5 | 0.532 | 0.532 |
| G | 5 | 0.189 | 0.466 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.355（retriever 平均距離: 0.355）
- 引用: `POPPING  societydance.txt` chunk 4 (dense), `History initiative to document the origins of ‘popping’ - Fresno State News.txt` chunk 0 (dense), `POPPING  societydance.txt` chunk 6 (dense)...

Step 1 — 列出 Facts：
1. Popping 是一種街舞和原始的放克舞蹈風格，起源於1960年代末至1970年代的加州弗雷斯諾。[來源 1]
2. Popping 的舞蹈技術是快速收縮和放鬆肌肉，使舞者身體產生抽動，稱為 pop 或 hit，並持續配合音樂節奏和各種動作與姿勢。[來源 1]
3. Popping 是由想要像洛杉磯和奧克蘭的幫派分子一樣看起來強悍的舞者創造的，特徵是身體部分肌肉的抽動動作，舞者在抽動時會發出“Pop, Pop”聲音，因此得名。[來源 3]
4. “Hitting” 是指更用力且突然地收縮肌肉，像是在嚇人一樣。[來源 3]
5. 歷史學家 Robert Farris Thompson 描述 popping 是軀幹和四肢的節奏性角度動作，速度可快可慢。[來源 4]
6. 先驅舞者 Timothy Solomon（又名 Pop’in Pete）定義 popping 是“用腿、脖子、頭等關節用力擊打”。[來源 4]
7. Popin’ Pete 是第一代 The Electric Boogaloos 成員，1978年加入該團體並學習 popping 風格。[來源 5]
8. The Electric Boogaloos 創始人 Boogaloo Sam 教導團員 popping 基礎動作。[來源 5]

Step 2 — 綜合回答：
Popping 是一種起源於1960年代末至1970年代加州弗雷斯諾的街舞和原始放克舞蹈風格，舞者通過快速收縮和放鬆肌肉，使身體產生抽動（稱為 pop 或 hit），並配合音樂節奏進行各種動作和姿勢。這種舞蹈由想要像洛杉磯和奧克蘭幫派分子一樣看起來強悍的舞者創造，他們在抽動時會發出“Pop, Pop”的聲音，因此得名。Hitting 指的是更用力且突然的肌肉收縮。歷史學家和先驅舞者也對 popping 有相關定義，強調其節奏性和關節用力擊打的特點。Popin’ Pete 是第一代 The Electric Boogaloos 成員，該團體由 Boogaloo Sam 創立並教授 popping 基礎動作。[來源 1][來源 3][來源 4][來源 5]

Step 3 — 缺口說明：
資料未涵蓋具體個人或團體最初創造 popping 的詳細身份。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.532（retriever 平均距離: 0.532）
- 引用: `POPPING  societydance.txt` chunk 6 (both), `POPPING  societydance.txt` chunk 4 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 22 (bm25)...

Step 1 — 列出 Facts：
1. Popping 是一種街舞，起源於1960年代末至1970年代的加州弗雷斯諾，屬於原始的放克舞蹈風格之一，舞蹈技巧基於快速收縮和放鬆肌肉，造成身體的抽動，稱為pop或hit，並與各種動作和姿勢結合跳舞。[來源 2]
2. Popping 是由想要像洛杉磯和奧克蘭的幫派成員一樣看起來強悍的舞者創造的。其特徵是通過身體部分肌肉的抽動產生突兀的動作，舞者在每次肌肉抽動時會低聲說“Pop, Pop”，因此稱為Popping。當肌肉抽動更用力且突然時稱為Hitting。[來源 1]

Step 2 — 綜合回答：
Popping 是一種起源於1960年代末至1970年代加州弗雷斯諾的街舞，屬於原始的放克舞蹈風格。這種舞蹈基於快速收縮和放鬆肌肉，造成身體的抽動，稱為pop或hit，並與多種動作和姿勢結合跳舞。Popping 是由想要像洛杉磯和奧克蘭的幫派成員一樣看起來強悍的舞者所創造，其特徵是通過肌肉的抽動產生突兀的動作，舞者在每次抽動時會低聲說“Pop, Pop”，因此得名。[來源 1][來源 2]

Step 3 — 缺口說明：
資料未涵蓋具體創作者的個人姓名或團體名稱。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.466（retriever 平均距離: 0.189）
- 擴展查詢: Popping 舞蹈的起源與創始人是誰？ / History and origin of Popping dance style / Who invented Popping and what defines this dance genre?
- 引用: `What Is Popping (Popping Dance)  STEEZY Blog.txt` chunk 0 (dense), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 1 (dense), `What Is Popping (Popping Dance)  STEEZY Blog.txt` chunk 2 (dense)...

Step 1 — 列出 Facts：
1. Popping is a style of dance that started in the late 1960’s and 70’s [來源 1]。
2. Popping is characterized by sudden tensing and releasing of the muscles ("hitting") to the rhythm of beats in music [來源 1]。
3. Popping originated primarily in California in the 1970s [來源 2]。
4. The dance style started among groups of teenage friends in deindustrializing parts of the city, who created moves, practiced them, and showed them off at clubs or dance battles [來源 4]。
5. The term “Popping” (also called “hitting”) describes the collection of moves and techniques invented by the OG Poppers [來源 3]。
6. Boogaloo Sam invented the Boogaloo style, which is a part of Popping [來源 3]。

Step 2 — 綜合回答：
Popping is a dance style that originated in the late 1960s and 1970s, primarily in California. It is characterized by the sudden tensing and releasing of muscles, known as "hitting," synchronized to the rhythm of music beats. The dance began among groups of teenage friends in deindustrializing urban areas, who developed and practiced moves before showcasing them in clubs and dance battles. The term "Popping" refers to the collection of moves and techniques created by the original pioneers, known as OG Poppers. Among these pioneers, Boogaloo Sam is credited with inventing the Boogaloo style, a fluid and loose movement within Popping [來源 1][來源 2][來源 3][來源 4]。

Step 3 — 缺口說明：
資料未涵蓋具體的個人或團體完整創始者名單，僅提及Boogaloo Sam發明Boogaloo風格，未明確指出Popping整體舞蹈的唯一創造者。


---

## Q2: Electric Boogaloos 的成員有哪些？他們各自的貢獻是什麼？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.358 | 0.358 |
| F | 5 | 0.441 | 0.441 |
| G | 5 | 0.189 | 0.481 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.358（retriever 平均距離: 0.358）
- 引用: `Popin' Pete - Wikipedia.txt` chunk 3 (dense), `Popin' Pete - Wikipedia.txt` chunk 6 (dense), `Popin' Pete - Wikipedia.txt` chunk 4 (dense)...

Step 1 — 列出 Facts：
1. Electric Boogaloos 的創始人是 Boogaloo Sam，他編制了一套基本動作並創立了 The Electric Boogaloos [來源 1]。
2. Boogaloo Sam 作為創始人，教導了 Electric Boogaloos 成員基本功 [來源 2]。
3. Electric Boogaloos 成員包括 Boogaloo Sam、Popin Pete、Skeeter Rabbit、Suga Pop、Mr. Wiggles、Jazzy J、Creepin Cid、Puppet Boozer、Robot Dane 等 [來源 5]。
4. Electric Boogaloos 推廣了電波舞（electric boogaloo）、popping 及相關舞蹈風格 [來源 4]。
5. 成員們都是經驗豐富且充滿熱情的舞者，團體很快取得成功，並曾在多個美國知名節目中表演，包括 The Midnight Special、Kicks 和 Soul Train [來源 2]。

Step 2 — 綜合回答：
Electric Boogaloos 的成員包括 Boogaloo Sam、Popin Pete、Skeeter Rabbit、Suga Pop、Mr. Wiggles、Jazzy J、Creepin Cid、Puppet Boozer 和 Robot Dane 等。Boogaloo Sam 是團體的創始人，他編制了一套基本動作並教導成員基本功，對團體的成立和舞蹈風格的形成有關鍵貢獻。團體成員皆為經驗豐富且熱情的舞者，他們共同推廣了電波舞（electric boogaloo）、popping 及相關舞蹈風格，使團體迅速成功並在多個美國知名節目中表演 [來源 1][來源 2][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋各成員個別具體的詳細貢獻。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.441（retriever 平均距離: 0.441）
- 引用: `popping_style_taxonomy.txt` chunk 5 (both), `popping_style_taxonomy.txt` chunk 2 (both), `Popin' Pete - Wikipedia.txt` chunk 3 (dense)...

Step 1 — 列出 Facts：
1. Electric Boogaloos 是由 Boogaloo Sam 在 1977 年於 Fresno 成立的舞團 [來源 1]。
2. Electric Boogaloos 的重要成員包括：
   - Sam Solomon（Boogaloo Sam）：創立 Electric Boogaloos 團體，發展 Boogaloo 和 Popping 技術，1970s Fresno [來源 4]。
   - Timothy Solomon（Popin' Pete）：Boogaloo Sam 的弟弟，Electric Boogaloos 創始成員，以細膩的 Popping 著稱 [來源 4]。
   - Michael Wayne Davis（Skeeter Rabbit）：Electric Boogaloos 創始成員，以 Boogaloo 和 Worm 動作知名 [來源 4]。
   - Larry Vazquez（Sugar Pop / Suga Pop）：Electric Boogaloos 成員，以流暢的 Liquid 和 Animation 風格著稱 [來源 4]。
   - Darryl James（Dane 1）：Electric Boogaloos 成員 [來源 4]。
   - Nate Johnson（Nate "Robot" Johnson）：Electric Boogaloos 成員，Robot 風格代表人物 [來源 4]。
3. Boogaloo Sam 作為創始人，教授基本動作給團體成員，團體很快成功 [來源 5]。
4. Electric Boogaloos 成員曾在多個美國熱門節目中表演，包括 The Midnight Special、Kicks 和 Soul Train（1980 年首次出現） [來源 5]。

Step 2 — 綜合回答：
Electric Boogaloos 的成員包括創始人 Sam Solomon（Boogaloo Sam），他在 1970 年代於 Fresno 創立該團體並發展了 Boogaloo 和 Popping 技術。Sam 也負責教授團體成員基本動作，使團體迅速走紅。其弟弟 Timothy Solomon（Popin' Pete）是創始成員之一，以細膩的 Popping 技術聞名。Michael Wayne Davis（Skeeter Rabbit）也是創始成員，以 Boogaloo 和 Worm 動作著稱。Larry Vazquez（Sugar Pop / Suga Pop）以流暢的 Liquid 和 Animation 風格聞名。Darryl James（Dane 1）和 Nate Johnson（Nate "Robot" Johnson）也是團體成員，其中 Nate Johnson 是 Robot 風格的代表人物。這些成員共同推動了 Electric Boogaloos 的成功，並曾在多個美國知名節目如 The Midnight Special、Kicks 和 Soul Train 中表演 [來源 1][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋其他成員的詳細貢獻。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.481（retriever 平均距離: 0.189）
- 擴展查詢: Electric Boogaloos 成員名單及其舞蹈貢獻介紹 / Who are the members of the Electric Boogaloos and what are their individual contributions? / List of Electric Boogaloos dancers and their roles in popularizing Popping and Boogaloo styles
- 引用: `Boogaloo (funk dance) - Wikipedia.txt` chunk 1 (dense), `Electric boogaloo (dance) - Wikipedia.txt` chunk 3 (dense), `POPPING  societydance.txt` chunk 3 (dense)...

Step 1 — 列出 Facts：
1. Boogaloo Sam 是 Electric Boogaloos 的創始人，並教導團員基本功。[來源 5]
2. Electric Boogaloos 是一支1970年代中期的舞蹈團體，推廣了 electric boogaloo、popping 及相關風格。[來源 2]
3. 團員們都是經驗豐富且充滿熱情的舞者，團體很快取得成功，並曾在多個美國知名節目中表演，如 The Midnight Special、Kicks 和 Soul Train（首次出現於1980年）。[來源 5]

Step 2 — 綜合回答：
Electric Boogaloos 的成員由創始人 Boogaloo Sam 領導，他負責教導團員舞蹈的基本技巧。團員們都是非常有經驗且熱情的舞者，整個團體在1970年代中期迅速成名，推廣了 electric boogaloo 舞蹈風格以及 popping 和相關舞蹈風格。團體成員曾多次在美國知名電視節目如 The Midnight Special、Kicks 和 Soul Train 中表演，首次亮相於1980年。[來源 2][來源 5]

Step 3 — 缺口說明：
資料中未具體列出 Electric Boogaloos 的所有成員姓名及其個別貢獻。


---

## Q3: Popping 和 Boogaloo 有什麼差別？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.368 | 0.368 |
| F | 5 | 0.390 | 0.390 |
| G | 5 | 0.199 | 0.420 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.368（retriever 平均距離: 0.368）
- 引用: `popping_style_taxonomy.txt` chunk 2 (dense), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `popping-terminology-snippet.txt` chunk 0 (dense)...

Step 1 — 列出 Facts：
1. Boogaloo 是 Popping 的子流派，不是 Popping 的同義詞；Boogaloo 更強調流動性，Popping 更強調爆點清晰度 [來源 1]。
2. Boogaloo 強調身體流動性（fluidity），以臀部、膝蓋的圓形律動為核心，比 Popping 更柔軟流暢。由 Sam Solomon（Boogaloo Sam）創立 [來源 5]。
3. Popping 更注重爆點（Hit）清晰度和技術控制 [來源 1]。
4. Electric Boogaloo 結合 Boogaloo 的流動性和 Popping 的爆點，是整個 Popping 文化的代名詞 [來源 4]。

Step 2 — 綜合回答：
Popping 是一種街舞風格，強調爆點（Hit）的清晰度和技術控制，而 Boogaloo 是 Popping 的一個子流派，著重於身體的流動性和柔軟度，特別是臀部和膝蓋的圓形律動。Boogaloo 由 Sam Solomon（Boogaloo Sam）於1970年代創立，強調身體的流暢動作，與 Popping 的爆點清晰形成對比。Electric Boogaloo 風格則結合了 Boogaloo 的流動性與 Popping 的爆點技術，成為 Popping 文化中的代表風格 [來源 1][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋其他可能的差異細節。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.390（retriever 平均距離: 0.390）
- 引用: `popping_style_taxonomy.txt` chunk 5 (both), `popping_style_taxonomy.txt` chunk 2 (both), `popping_style_taxonomy.txt` chunk 1 (both)...

Step 1 — 列出 Facts：
1. Boogaloo 是 Popping 的子流派，更強調流動性，Popping 更強調爆點清晰度。[來源 2][來源 3][來源 5]
2. Boogaloo 起源於1970s加州 Richmond / Fresno，由 Sam Solomon（Boogaloo Sam）創立。[來源 3]
3. Boogaloo 技術特徵包括全身性律動，以臀部和膝蓋為中心的圓形搖擺，動作流暢而非爆裂，身體每個關節參與律動，製造蠕動感，腿部常見 dip（身體下沉）和 roll（旋轉）。[來源 3]
4. Popping 是包含 Boogaloo 在內的更大類別，強調爆點技術和節奏控制。[來源 2][來源 5]
5. Popping 和 Boogaloo 常被混淆，但兩者不是同義詞。[來源 2][來源 5]

Step 2 — 綜合回答：
Popping 是一種包含多個子流派的放克舞蹈大類，其中 Boogaloo 是其子流派之一。Boogaloo 起源於1970年代加州 Richmond 和 Fresno，由 Boogaloo Sam 創立，技術特徵強調全身流暢的律動，尤其以臀部和膝蓋為中心的圓形搖擺動作，動作較為流動且柔和，身體各關節協同製造蠕動感，腿部常見身體下沉和旋轉。相較之下，Popping 更注重爆點技術的清晰度和節奏控制，動作多以突然的肌肉收縮（爆點）為核心。雖然 Boogaloo 是 Popping 的子流派，但兩者在風格和技術重點上有所不同，且常被誤認為同義詞。[來源 2][來源 3][來源 5]

Step 3 — 缺口說明：
資料未涵蓋更詳細的技術差異或歷史背景。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.420（retriever 平均距離: 0.199）
- 擴展查詢: Popping 與 Boogaloo 兩種舞蹈風格的主要區別是什麼？ / Differences between Popping and Boogaloo dance styles / How does Boogaloo differ from Popping in street dance culture?
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `POPPING  societydance.txt` chunk 3 (dense), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 1 (dense)...

Step 1 — 列出 Facts：
1. Popping 是一種起源於1970年代加州的街舞風格，透過肌肉的節奏性收縮與放鬆，配合音樂節拍，產生獨特的「pop」或「hit」效果。[來源 3]
2. Boogaloo 是一種結合「The Robot」和模仿動作（mime）中平滑隔離動作的舞蹈風格，強調身體各部位的流暢運用，包括臀部、膝蓋和頭部的滾動。[來源 2]
3. Boogaloo 舞者會使用幻覺效果、肌肉限制、停頓、機械人動作和擺動，創造出充滿靈魂、熱情且生動的街舞形式，並融合了基礎的 popping 技巧，這些技巧最初被稱為「Posing Hard」。[來源 5]
4. Electric Boogaloo 是一種與 Boogaloo 和 popping 密切相關的舞蹈風格，結合了現代 popping 技巧和早期 Boogaloo 形式。[來源 4]

Step 2 — 綜合回答：
Popping 是一種起源於1970年代加州的街舞風格，透過肌肉的節奏性收縮與放鬆，配合音樂節拍，產生獨特的「pop」或「hit」效果，強調動作的節奏感和爆發力。[來源 3] Boogaloo 則是一種融合了「The Robot」和模仿動作中平滑隔離技巧的舞蹈風格，強調身體各部位的流暢運用，如臀部、膝蓋和頭部的滾動，並且利用幻覺效果、肌肉限制、停頓和機械人動作來創造生動且富有靈魂的表現。[來源 2][來源 5] Boogaloo 也包含了基礎的 popping 技巧，兩者在技術上有交集，但 Boogaloo 更注重流暢性和身體各部位的連貫動作，而 Popping 則著重於肌肉的節奏性收縮與爆發。[來源 5] 此外，Electric Boogaloo 是結合 Boogaloo 和 popping 技巧的舞蹈風格，顯示兩者在街舞文化中的密切關聯。[來源 4]

Step 3 — 缺口說明：
資料未涵蓋 Popping 和 Boogaloo 在音樂選擇、文化背景或表演場合上的具體差異。


---

## Q4: Juste Debout 比賽的規則和歷史？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.466 | 0.466 |
| F | 5 | 0.558 | 0.558 |
| G | 5 | 0.377 | 0.600 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.466（retriever 平均距離: 0.466）
- 引用: `Salah (dancer) - Wikipedia.txt` chunk 66 (dense), `popping_history_timeline.txt` chunk 7 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 22 (dense)...

Step 1 — 列出 Facts：
1. Juste Debout 是一個在巴黎舉辦的活動，早期聚焦 Popping 和 Locking。[來源 2]
2. 2002 年，Juste Debout 正式成為年度賽事，成為全球最重要的 Popping/Locking/House 2v2 比賽，吸引來自 60 多個國家的舞者。[來源 2]

Step 2 — 綜合回答：
Juste Debout 最初是在巴黎舉辦的活動，早期主要聚焦於 Popping 和 Locking 舞風。到了 2002 年，Juste Debout 正式成為年度賽事，並迅速發展成為全球最重要的 Popping、Locking 和 House 2 對 2 比賽，吸引了來自超過 60 個國家的舞者參與。[來源 2]

Step 3 — 缺口說明：
資料中未涵蓋 Juste Debout 比賽的具體規則細節。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.558（retriever 平均距離: 0.558）
- 引用: `popping_history_timeline.txt` chunk 7 (both), `Salah (dancer) - Wikipedia.txt` chunk 66 (dense), `popping_style_taxonomy.txt` chunk 6 (bm25)...

Step 1 — 列出 Facts：
1. Juste Debout 前身活動在巴黎舉辦，由 Bams 和 Anne Nguyen 組織，早期聚焦 Popping 和 Locking。[來源 1]
2. 2002 年，Juste Debout 正式成為年度賽事（巴黎），成為全球最重要的 Popping/Locking/House 2v2 比賽，吸引來自 60+ 個國家的舞者。[來源 1]
3. 根據 HHI 官方規則和 Juste Debout 評分標準，Popping Battle 的評判維度包括 Musicality（樂感）、Foundation（基礎）、Originality（原創性）、Execution（執行）、Technique（技術）、Creativity（創造力）、Floor Work（地板動作）、Showmanship（表演力）、Costume（服裝）。其中 Musicality 和 Foundation 為最重要的評分項目。[來源 3]

Step 2 — 綜合回答：
Juste Debout 比賽起源於巴黎，最初由 Bams 和 Anne Nguyen 組織，聚焦於 Popping 和 Locking 等街舞風格。2002 年，Juste Debout 正式成為年度賽事，並迅速發展成為全球最重要的 Popping、Locking 和 House 2v2 比賽，吸引來自超過 60 個國家的舞者參加[來源 1]。在評分規則方面，Juste Debout 採用與 HHI 官方規則相似的評判標準，Popping Battle 的評分維度包含樂感（Musicality）、基礎動作（Foundation）、原創性（Originality）、執行度（Execution）、技術（Technique）、創造力（Creativity）、地板動作（Floor Work）、表演力（Showmanship）及服裝（Costume）。其中，樂感和基礎動作被視為最重要的評分項目，分別獲得最高的評分權重[來源 3]。

Step 3 — 缺口說明：
資料未涵蓋 Juste Debout 比賽的具體規則細節（如比賽流程、回合數、勝負判定方式等）。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.600（retriever 平均距離: 0.377）
- 擴展查詢: Juste Debout competition rules and history / Juste Debout 比賽的起源與比賽規則介紹 / What are the judging criteria and background of Juste Debout dance battles?
- 引用: `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 12 (dense), `Popping Dance Battles Near You — EVERYDAY POPPING - Online Popping Dance Tutorials.txt` chunk 1 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 18 (dense)...

Step 1 — 列出 Facts：
1. Juste Debout 是世界上最大的街舞比賽之一，地點在法國巴黎。[來源 2]
2. Juste Debout 包含 popping 類別，舞者以兩人一組（2 on 2）形式競賽。[來源 2]
3. Juste Debout 強調創意和原創性，舞者在充滿活力的氛圍中展示技巧。[來源 2]

Step 2 — 綜合回答：
Juste Debout 是在法國巴黎舉辦的全球最大街舞比賽之一，包含 popping 類別，舞者以兩人一組的形式參賽。該比賽強調舞者的創意和原創性，並在充滿活力的氛圍中展現舞技。[來源 2]

Step 3 — 缺口說明：
資料未涵蓋 Juste Debout 比賽的詳細規則和歷史背景。


---

## Q5: 如何判斷一場 Popping battle 的勝負？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.346 | 0.346 |
| F | 5 | 0.380 | 0.380 |
| G | 5 | 0.294 | 0.362 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.346（retriever 平均距離: 0.346）
- 引用: `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 0 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 15 (dense)...

Step 1 — 列出 Facts：
1. Popping battle 中兩位選手各有一分鐘的表演時間，並且沒有採用積分制，評判重點在於音樂感而非競爭分數 [來源 1]。
2. 裁判會根據多個評分維度分別評判，尋找符合所有標準的「完整的 Popper」 [來源 1]。
3. Popping battle 的評分維度包括樂感（Musicality）、基礎（Foundation）、原創性（Originality）、執行（Execution）、技術（Technique）、創造力（Creativity）、地板動作（Floor Work）、表演力（Showmanship）及服裝（Costume），其中樂感與基礎最為重要 [來源 4]。
4. Battle 與 cypher 是常見的賽制，評判常關注音樂性、獨創性與完整度 [來源 5]。

Step 2 — 綜合回答：
判斷一場 Popping battle 的勝負，並非依靠積分制，而是由裁判根據多項評分標準分別評判。兩位選手各有一分鐘的表演時間，裁判會從樂感（動作與音樂的節奏、旋律、歌詞的對應程度）、基礎動作的扎實程度（如 Pop 的清晰度、Isolation 的精準度）、原創性、執行度、技術掌握、創造力、地板動作運用、表演力及服裝等多個面向評估。裁判尋找的是能夠涵蓋所有這些標準的「完整的 Popper」，因此勝負是基於誰更符合這些評判標準，而非單純的分數累計或點數決定 [來源 1][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋具體裁判如何宣布勝負的程序或是否有其他決勝機制。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.380（retriever 平均距離: 0.380）
- 引用: `popping_style_taxonomy.txt` chunk 6 (both), `popping-culture-notes.txt` chunk 0 (both), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense)...

Step 1 — 列出 Facts：
1. Popping Battle 的評判維度包括 Musicality（樂感）、Foundation（基礎）、Originality（原創性）、Execution（執行）、Technique（技術）、Creativity（創造力）、Floor Work（地板動作）、Showmanship（表演力）、Costume（服裝）[來源 1]。
2. 評分標準中，Musicality 和 Foundation 是最重要的評判維度，均為五顆星重要性[來源 1]。
3. 1vs1 Popping Battles 是兩位舞者面對面進行，每人有一分鐘的表演時間，沒有點數系統，評判會分別對各項標準進行評價，尋找符合所有評判標準的「完整的 Popping 舞者」[來源 3][來源 5]。
4. Battle 是兩位舞者的對抗性表演，由評審或觀眾判定勝負[來源 4]。
5. 評判重點在於舞者是否符合所有評判標準，強調音樂性而非直接競爭分數[來源 3]。

Step 2 — 綜合回答：
判斷一場 Popping battle 的勝負，主要依據評審對舞者在多個評分維度上的表現進行綜合評價。這些評分維度包括樂感（Musicality）、基礎動作（Foundation）、原創性（Originality）、執行度（Execution）、技術掌握（Technique）、創造力（Creativity）、地板動作（Floor Work）、表演力（Showmanship）以及服裝（Costume）。其中，樂感與基礎動作是最重要的評判標準。比賽中，兩位舞者各有一分鐘時間展示，評審不使用點數系統，而是分別評估各項標準，尋找在所有評判標準上表現均衡且完整的舞者，最終由評審判定勝負[來源 1][來源 3][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋具體評判勝負的細節流程，例如評審如何在無點數系統下決定勝者的具體方法。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.362（retriever 平均距離: 0.294）
- 擴展查詢: Popping battle 判定勝負的標準和技巧 / Criteria for winning a Popping dance battle / How to judge the winner in a Popping dance competition
- 引用: `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 9 (dense), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense), `popping_style_taxonomy.txt` chunk 6 (dense)...

Step 1 — 列出 Facts：
1. Popping battle 通常是 1 對 1 的對決，每位舞者有 1 分鐘的表演時間，沒有使用點數系統，評判重點在音樂感受而非競爭分數[來源 2]。
2. 裁判會根據多個評分維度分別評判，尋找符合所有標準的「完整舞者」[來源 2]。
3. 評分維度包括：Musicality（樂感）、Foundation（基礎）、Originality（原創性）、Execution（執行）、Technique（技術）、Creativity（創造力）、Floor Work（地板動作）、Showmanship（表演力）及 Costume（服裝）[來源 3]。
4. 其中樂感和基礎是最重要的評分標準，其他如原創性、執行、技術也非常重要[來源 3]。
5. Battle 的結果反映裁判對舞者是否符合上述評分標準的判斷[來源 2]。

Step 2 — 綜合回答：
判斷一場 Popping battle 的勝負，裁判會在每位舞者各自 1 分鐘的表演中，根據多個評分維度進行評判，這些維度包括樂感、基礎動作的扎實度、原創性、執行力、技術掌握、創造力、地板動作、表演力及服裝等。其中，樂感與基礎動作是最重要的評分標準。裁判不使用點數系統，而是根據舞者是否全面展現這些標準來決定勝負，尋找「完整舞者」。因此，勝負判斷是基於舞者在音樂配合度及技術表現上的整體優劣[來源 2][來源 3]。

Step 3 — 缺口說明：
資料未涵蓋具體勝負判定的流程細節（如是否有多輪評判、如何決定平手等）。
