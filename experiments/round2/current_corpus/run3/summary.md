# RAG 3 組實驗比較（E、F、G）

*生成時間: 2026-11-10 01:18 UTC；來源: 3 個 combo_*/results.json*

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
| 產生時間 | 2026-11-10T01:17:42+00:00 | 2026-11-10T01:18:12+00:00 | 2026-11-10T01:18:46+00:00 |

## 與原問題的平均距離

| Combo | 與原問題平均距離（全部題目的 chunks） |
|---|---|
| E | 0.379 |
| F | 0.460 |
| G | 0.471 |

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
1. Popping 是一種街舞，起源於1960年代末至1970年代的加州弗雷斯諾，屬於原始的放克舞蹈風格之一。舞蹈技術基於快速收縮和放鬆肌肉，造成身體的抽動，稱為 pop 或 hit，並持續配合音樂節奏和各種動作與姿勢進行 [來源 1]。
2. Popping 是由想在洛杉磯和奧克蘭跳舞時看起來像幫派份子的舞者創造的，特點是身體部分肌肉的抽動動作。舞者在抽動時會發出“Pop, Pop”的聲音，因此稱為 Popping。Hitting 指的是更用力且突然的肌肉抽動，像是在嚇人 [來源 3]。
3. 歷史學家 Robert Farris Thompson 描述 Popping 為“軀幹和四肢的節奏性角度變化”，節奏可快可慢。先驅舞者 Timothy Solomon（又名 Pop’in Pete）定義 Popping 是“用腿、脖子、頭等關節用力擊打” [來源 4]。
4. Popin’ Pete 是第一代 The Electric Boogaloos 成員，1978年加入該團體並開始學習 Popping。Boogaloo Sam 創立了該團體並教導基本技巧 [來源 5]。

Step 2 — 綜合回答：
Popping 是一種起源於1960年代末至1970年代加州弗雷斯諾的街舞，屬於原始的放克舞蹈風格。其舞蹈技術基於快速收縮和放鬆肌肉，造成身體抽動（稱為 pop 或 hit），並配合音樂節奏和多樣動作進行。Popping 由想在洛杉磯和奧克蘭跳舞時看起來像幫派份子的舞者創造，特點是肌肉的節奏性抽動，舞者在抽動時會發出“Pop, Pop”的聲音。Hitting 是更用力且突然的肌肉抽動。歷史學家和先驅舞者如 Timothy Solomon（Pop’in Pete）對 Popping 有詳細定義。Popin’ Pete 是第一代 The Electric Boogaloos 成員，該團體由 Boogaloo Sam 創立並教導基本技巧 [來源 1][來源 3][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋具體個人或團體最初創造 Popping 的完整詳細背景。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.532（retriever 平均距離: 0.532）
- 引用: `POPPING  societydance.txt` chunk 6 (both), `POPPING  societydance.txt` chunk 4 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 22 (bm25)...

Step 1 — 列出 Facts：
1. Popping 是一種街舞，起源於1960年代末至1970年代的加州弗雷斯諾（Fresno, California）[來源 2]。
2. Popping 的舞蹈技巧基於快速收縮和放鬆肌肉，造成身體的抽動，稱為 pop 或 hit，並持續配合音樂節奏與各種動作和姿勢[來源 2]。
3. Popping 是由想要像洛杉磯和奧克蘭的幫派成員一樣看起來強悍的舞者創造的[來源 1]。
4. 舞者在做出肌肉收縮動作時會發出“Pop, Pop”的聲音，因此這種舞蹈被稱為 Popping[來源 1]。

Step 2 — 綜合回答：
Popping 是一種起源於1960年代末至1970年代加州弗雷斯諾的街舞風格，屬於原始的放克（funk）舞蹈之一。這種舞蹈的核心技巧是快速收縮和放鬆肌肉，造成身體的抽動，稱為 pop 或 hit，並且持續配合音樂節奏進行多樣的動作和姿勢。Popping 是由想要模仿洛杉磯和奧克蘭幫派成員強悍形象的舞者所創造，舞者在做出肌肉收縮時會發出“Pop, Pop”的聲音，這也成為該舞蹈名稱的由來[來源 1][來源 2]。

Step 3 — 缺口說明：
資料未涵蓋具體創造 Popping 的個人或團體名稱。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.466（retriever 平均距離: 0.189）
- 擴展查詢: Popping 舞蹈的起源與創始人是誰？ / History and origin of Popping dance style / Who invented Popping and key characteristics of this dance?
- 引用: `What Is Popping (Popping Dance)  STEEZY Blog.txt` chunk 0 (dense), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 1 (dense), `What Is Popping (Popping Dance)  STEEZY Blog.txt` chunk 2 (dense)...

Step 1 — 列出 Facts：
1. Popping is a style of dance that started in the late 1960’s and 70’s [來源 1]。
2. Popping is characterized by sudden tensing and releasing of the muscles ("hitting") to the rhythm of beats in music [來源 1]。
3. Popping originated primarily in California in the 1970s [來源 2]。
4. The dance style started among groups of teenage friends in deindustrializing parts of the city, who created moves, practiced them, and showed them off at clubs or dance battles [來源 4]。
5. The term “Popping” (also called “hitting”) describes the collection of moves and techniques invented by the OG Poppers [來源 3]。
6. Boogaloo Sam invented the Boogaloo style, which is a part of Popping dance [來源 3]。

Step 2 — 綜合回答：
Popping is a dance style that originated in the late 1960s and 1970s, primarily in California. It is characterized by the sudden tensing and releasing of muscles, known as "hitting," synchronized to the rhythm of music beats. The dance began among groups of teenage friends in deindustrializing urban areas who created and practiced moves, then showcased them in clubs and dance battles. The term "Popping" refers to the collection of moves and techniques developed by the original pioneers, known as OG Poppers. Among these pioneers, Boogaloo Sam is credited with inventing the Boogaloo style, a fluid and loose motion within Popping [來源 1][來源 2][來源 3][來源 4]。

Step 3 — 缺口說明：
資料未涵蓋具體的個人或團體創始者的完整名單，僅提及 Boogaloo Sam 發明了 Boogaloo 風格，未明確指出整體 Popping 舞蹈的唯一創造者。


---

## Q2: Electric Boogaloos 的成員有哪些？他們各自的貢獻是什麼？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.358 | 0.358 |
| F | 5 | 0.441 | 0.441 |
| G | 5 | 0.197 | 0.435 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.358（retriever 平均距離: 0.358）
- 引用: `Popin' Pete - Wikipedia.txt` chunk 3 (dense), `Popin' Pete - Wikipedia.txt` chunk 6 (dense), `Popin' Pete - Wikipedia.txt` chunk 4 (dense)...

Step 1 — 列出 Facts：
1. Boogaloo Sam 是 Electric Boogaloos 的創始人，並教導成員基本動作 [來源 2]。
2. Popin' Pete 是 Electric Boogaloos 的成員之一 [來源 5]。
3. Electric Boogaloos 的其他成員包括 Skeeter Rabbit、Suga Pop、Mr. Wiggles、Jazzy J、Creepin Cid、Puppet Boozer、Robot Dane [來源 5]。
4. Boogaloo Sam 創造並命名了 popping 這種舞蹈風格 [來源 5]。
5. Electric Boogaloos 推廣了 popping 和相關舞蹈風格 [來源 4]。
6. Electric Boogaloos 成員都是經驗豐富且充滿熱情的舞者，團體很快取得成功，並在多個美國知名節目中表演，包括 The Midnight Special、Kicks 和 Soul Train [來源 2]。

Step 2 — 綜合回答：
Electric Boogaloos 的成員包括 Boogaloo Sam、Popin' Pete、Skeeter Rabbit、Suga Pop、Mr. Wiggles、Jazzy J、Creepin Cid、Puppet Boozer 和 Robot Dane [來源 5]。其中，Boogaloo Sam 是團體的創始人，他編輯並教導了基本的舞蹈動作，並且創造了 popping 這種舞蹈風格的名稱 [來源 2][來源 5]。Popin' Pete 也是團體的重要成員 [來源 5]。整個團體的成員都是非常有經驗且熱情的舞者，他們推廣了 popping 及其相關舞蹈風格，並在多個著名的美國電視節目中表演，促進了這些舞蹈風格的普及 [來源 2][來源 4]。

Step 3 — 缺口說明：
資料未涵蓋各成員具體的個別貢獻細節，除了 Boogaloo Sam 創立團體和命名 popping 之外，其他成員的具體貢獻未明確說明。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.441（retriever 平均距離: 0.441）
- 引用: `popping_style_taxonomy.txt` chunk 5 (both), `popping_style_taxonomy.txt` chunk 2 (both), `Popin' Pete - Wikipedia.txt` chunk 3 (dense)...

Step 1 — 列出 Facts：
1. Electric Boogaloos 是 1977 年由 Boogaloo Sam 在 Fresno 成立的舞團 [來源 1]。
2. Electric Boogaloos 的成員包括：Boogaloo Sam（Sam Solomon）、Popin' Pete（Timothy Solomon）、Skeeter Rabbit（Michael Wayne Davis）、Sugar Pop / Suga Pop（Larry Vazquez）、Dane 1（Darryl James）、Nate "Robot" Johnson（Nate Johnson）[來源 4]。
3. Boogaloo Sam 創立 Electric Boogaloos 團體，發展 Boogaloo 和 Popping 技術，1970s Fresno [來源 4]。
4. Popin' Pete 是 Boogaloo Sam 的弟弟，Electric Boogaloos 創始成員，以細膩的 Popping 著稱 [來源 4]。
5. Skeeter Rabbit 是 Electric Boogaloos 創始成員，以 Boogaloo 和 Worm 動作知名 [來源 4]。
6. Sugar Pop / Suga Pop 是 Electric Boogaloos 成員，以流暢的 Liquid 和 Animation 風格著稱 [來源 4]。
7. Dane 1 是 Electric Boogaloos 成員 [來源 4]。
8. Nate "Robot" Johnson 是 Electric Boogaloos 成員，Robot 風格代表人物 [來源 4]。

Step 2 — 綜合回答：
Electric Boogaloos 的成員包括 Boogaloo Sam（Sam Solomon）、Popin' Pete（Timothy Solomon）、Skeeter Rabbit（Michael Wayne Davis）、Sugar Pop / Suga Pop（Larry Vazquez）、Dane 1（Darryl James）和 Nate "Robot" Johnson（Nate Johnson）[來源 4]。其中，Boogaloo Sam 創立了 Electric Boogaloos 團體並發展了 Boogaloo 和 Popping 技術[來源 4]；Popin' Pete 是 Boogaloo Sam 的弟弟，為創始成員，以細膩的 Popping 技術著稱[來源 4]；Skeeter Rabbit 也是創始成員，以 Boogaloo 和 Worm 動作聞名[來源 4]；Sugar Pop 以流暢的 Liquid 和 Animation 風格著稱[來源 4]；Dane 1 是團體成員之一[來源 4]；Nate "Robot" Johnson 則是 Robot 風格的代表人物[來源 4]。

Step 3 — 缺口說明：
資料未涵蓋其他成員的具體貢獻細節。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.435（retriever 平均距離: 0.197）
- 擴展查詢: Electric Boogaloos 成員名單及其舞蹈風格貢獻 / Who are the members of the Electric Boogaloos and what are their individual contributions to popping? / Electric Boogaloos key dancers and their roles in developing Boogaloo and Fresno styles
- 引用: `Boogaloo (funk dance) - Wikipedia.txt` chunk 1 (dense), `Electric boogaloo (dance) - Wikipedia.txt` chunk 3 (dense), `Electric boogaloo (dance) - Wikipedia.txt` chunk 2 (dense)...

Step 1 — 列出 Facts：
1. Electric Boogaloos 是一個成立於 1977 年，由 Boogaloo Sam 在 Fresno 成立的舞團 [來源 5]。
2. Boogaloo Sam 是 Electric Boogaloos 的創始人，並教導成員基本舞步 [來源 4]。
3. Electric Boogaloos 的成員都是非常有經驗且熱情的舞者，團體很快取得成功 [來源 4]。
4. 成員包括 Popin' Pete（在來源 4 中提及他與其他成員一起出現在多個美國流行節目中） [來源 4]。
5. Electric Boogaloos 不僅是舞團名稱，也是舞蹈風格名稱，結合 Boogaloo 的流動性和 Popping 的爆點，是 Popping 文化的代表 [來源 3][來源 5]。

Step 2 — 綜合回答：
Electric Boogaloos 是由 Boogaloo Sam 於 1977 年在 Fresno 成立的舞團，Boogaloo Sam 作為創始人，負責教導團員基本舞步，為團體奠定基礎 [來源 4][來源 5]。團體成員皆為經驗豐富且充滿熱情的舞者，其中包括 Popin' Pete，他們共同推廣並表演 Electric Boogaloo 舞蹈風格，並在多個美國流行節目如 The Midnight Special、Kicks 和 Soul Train 中亮相，促進了該舞蹈風格的普及 [來源 4]。Electric Boogaloos 不僅是團體名稱，也代表一種結合 Boogaloo 舞蹈流動性與 Popping 技術的舞蹈風格，是 Popping 文化的重要象徵 [來源 3][來源 5]。

Step 3 — 缺口說明：
資料中未具體列出 Electric Boogaloos 的所有成員名單及其個別貢獻。


---

## Q3: Popping 和 Boogaloo 有什麼差別？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.368 | 0.368 |
| F | 5 | 0.390 | 0.390 |
| G | 5 | 0.204 | 0.407 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.368（retriever 平均距離: 0.368）
- 引用: `popping_style_taxonomy.txt` chunk 2 (dense), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `popping-terminology-snippet.txt` chunk 0 (dense)...

Step 1 — 列出 Facts：
1. Boogaloo 是 Popping 的子流派，不是 Popping 的同義詞；Boogaloo 更強調流動性，Popping 更強調爆點清晰度 [來源 1]。
2. Boogaloo 強調身體流動性（fluidity），以臀部、膝蓋的圓形律動為核心，比 Popping 更柔軟流暢。由 Sam Solomon（Boogaloo Sam）創立 [來源 5]。
3. Popping 更注重技術控制和爆點（hit）清晰度 [來源 1]。
4. Electric Boogaloo 結合 Boogaloo 的流動性和 Popping 的爆點，是整個 Popping 文化的代名詞 [來源 4]。

Step 2 — 綜合回答：
Popping 是一種街舞風格，強調爆點（hit）的清晰度和技術控制，而 Boogaloo 是 Popping 的一個子流派，起源於1970年代加州，由 Boogaloo Sam 創立。Boogaloo 更加注重身體的流動性，尤其是臀部和膝蓋的圓形律動，使動作看起來更加柔軟流暢。相較之下，Popping 的動作較為剛硬且爆點明顯。Electric Boogaloo 風格則結合了 Boogaloo 的流動性和 Popping 的爆點特點，成為 Popping 文化中的代表風格之一 [來源 1][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋其他可能的差異細節。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.390（retriever 平均距離: 0.390）
- 引用: `popping_style_taxonomy.txt` chunk 5 (both), `popping_style_taxonomy.txt` chunk 2 (both), `popping_style_taxonomy.txt` chunk 1 (both)...

Step 1 — 列出 Facts：
1. Boogaloo 是 Popping 的子流派，更強調流動性，Popping 更強調爆點清晰度 [來源 2]。
2. Boogaloo 起源於1970s加州 Richmond / Fresno，由 Sam Solomon（Boogaloo Sam）創立 [來源 3]。
3. Boogaloo 技術特徵包括全身性律動，以臀部和膝蓋為中心的圓形搖擺，動作流暢而非爆裂，身體每個關節參與律動，製造蠕動感 [來源 3]。
4. Popping 是包含 Boogaloo 在內的更大類別 [來源 5]。
5. Popping 更注重技術控制和爆點技術 [來源 1]。

Step 2 — 綜合回答：
Popping 是一種包含多個子流派的放克舞蹈大類，其中 Boogaloo 是其子流派之一。Boogaloo 起源於1970年代的加州，由 Boogaloo Sam 創立，技術特徵是全身性流暢律動，特別強調臀部和膝蓋的圓形搖擺，動作較為流暢且帶有蠕動感。相比之下，Popping 更注重爆點的清晰度和技術控制，強調身體的突然收縮和釋放動作。簡言之，Boogaloo 強調流動性和連續性，而 Popping 強調爆點和節奏感 [來源 2][來源 3][來源 5][來源 1]。

Step 3 — 缺口說明：
資料未涵蓋其他可能的文化或歷史差異。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.407（retriever 平均距離: 0.204）
- 擴展查詢: Popping 與 Boogaloo 的舞蹈風格差異 / Differences between Popping and Boogaloo dance styles / Comparison of Fresno, Popping, and Boogaloo techniques in street dance
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 1 (dense), `POPPING  societydance.txt` chunk 3 (dense)...

Step 1 — 列出 Facts：
1. Popping 是一種起源於1970年代加州的街舞風格，透過肌肉節奏性收縮和放鬆來配合音樂節拍，產生獨特的「pop」或「hit」效果。[來源 2]
2. Boogaloo 是一種舞蹈風格，特徵是身體各部位的流暢運用，包括臀部、膝蓋和頭部的滾動，結合了「機器人舞」和模仿動作的平滑隔離動作。[來源 3]
3. Boogaloo 由 Boogaloo Sam 發明，並且是 Popping 舞蹈風格中的一部分。[來源 4]
4. 1970年代初，Black Messengers 團體創新了一種稱為「Posing Hard」的 Boogaloo 技巧，舞者會在姿勢結束時以強烈的「hit」凍結動作，這種技巧影響了現代 Popping 的技術。[來源 5]

Step 2 — 綜合回答：
Popping 是一種起源於1970年代加州的街舞風格，透過肌肉節奏性收縮和放鬆來配合音樂節拍，創造出獨特的「pop」或「hit」效果。[來源 2] Boogaloo 則是一種舞蹈風格，強調身體各部位的流暢運用，如臀部、膝蓋和頭部的滾動，結合了「機器人舞」和平滑的隔離動作。[來源 3] Boogaloo 是由 Boogaloo Sam 發明，並且是 Popping 舞蹈風格中的一個子類型。[來源 4] 此外，1970年代初 Black Messengers 團體創新的「Posing Hard」技巧，即在姿勢結束時強烈凍結動作，對現代 Popping 技術產生了重要影響。[來源 5]

Step 3 — 缺口說明：
資料未涵蓋 Popping 和 Boogaloo 在音樂選擇、舞蹈氛圍或文化背景等其他可能差異的詳細說明。


---

## Q4: Juste Debout 比賽的規則和歷史？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.466 | 0.466 |
| F | 5 | 0.558 | 0.558 |
| G | 5 | 0.383 | 0.663 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.466（retriever 平均距離: 0.466）
- 引用: `Salah (dancer) - Wikipedia.txt` chunk 66 (dense), `popping_history_timeline.txt` chunk 7 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 22 (dense)...

Step 1 — 列出 Facts：
1. Juste Debout 前身活動在巴黎舉辦，由 Bams 和 Anne Nguyen 組織，早期聚焦 Popping 和 Locking。[來源 2]
2. 2002 年，Juste Debout 正式成為年度賽事（巴黎），成為全球最重要的 Popping/Locking/House 2v2 比賽，吸引來自 60+ 個國家的舞者。[來源 2]

Step 2 — 綜合回答：
Juste Debout 最初由 Bams 和 Anne Nguyen 在巴黎組織舉辦，早期活動主要聚焦於 Popping 和 Locking 舞風。到了 2002 年，Juste Debout 正式成為年度賽事，並迅速發展成為全球最重要的 Popping、Locking 和 House 2v2 比賽，吸引超過 60 個國家的舞者參加。[來源 2]

Step 3 — 缺口說明：
資料未涵蓋 Juste Debout 比賽的具體規則細節。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.558（retriever 平均距離: 0.558）
- 引用: `popping_history_timeline.txt` chunk 7 (both), `Salah (dancer) - Wikipedia.txt` chunk 66 (dense), `popping_style_taxonomy.txt` chunk 6 (bm25)...

Step 1 — 列出 Facts：
1. Juste Debout 的前身活動在巴黎舉辦，由 Bams 和 Anne Nguyen 組織，早期聚焦 Popping 和 Locking。[來源 1]
2. 2002 年，Juste Debout 正式成為年度賽事（巴黎），成為全球最重要的 Popping/Locking/House 2v2 比賽，吸引來自 60+ 個國家的舞者。[來源 1]
3. 根據 HHI 官方規則和 Juste Debout 評分標準，Popping Battle 的評判維度包括 Musicality（樂感）、Foundation（基礎）、Originality（原創性）、Execution（執行）、Technique（技術）、Creativity（創造力）、Floor Work（地板動作）、Showmanship（表演力）、Costume（服裝）。其中 Musicality 和 Foundation 為最重要的評分項目。[來源 3]

Step 2 — 綜合回答：
Juste Debout 比賽起源於巴黎，最初由 Bams 和 Anne Nguyen 組織的活動，主要聚焦於 Popping 和 Locking 兩種舞蹈風格。2002 年，Juste Debout 正式成為年度賽事，並迅速發展成為全球最重要的 Popping、Locking 和 House 2v2 比賽，吸引超過 60 個國家的舞者參與。[來源 1]  
在比賽規則方面，Juste Debout 採用與 HHI 官方規則相似的評分標準，Popping Battle 的評判維度包括樂感（Musicality）、基礎動作的扎實程度（Foundation）、原創性（Originality）、動作執行度（Execution）、技術掌握（Technique）、創造力（Creativity）、地板動作運用（Floor Work）、表演力（Showmanship）及服裝（Costume）。其中樂感與基礎動作被視為最重要的評分項目，分別獲得最高星級評價。[來源 3]

Step 3 — 缺口說明：
資料未涵蓋 Juste Debout 比賽的具體規則細節（如比賽流程、回合數、勝負判定方式等）。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.663（retriever 平均距離: 0.383）
- 擴展查詢: Juste Debout competition rules and history / Juste Debout 比賽的起源與比賽規則介紹 / What are the main guidelines and background of Juste Debout dance battles?
- 引用: `Popping Dance Battles Near You — EVERYDAY POPPING - Online Popping Dance Tutorials.txt` chunk 1 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 2 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 7 (dense)...

Step 1 — 列出 Facts：
1. Juste Debout 是世界上最大的街舞比賽之一，於法國巴黎舉行，包含一個2對2的Popping舞蹈類別，強調創意和原創性，舞者以配對形式競賽，氣氛熱烈 [來源 1]。
2. 舞蹈比賽通常有資格賽，所有參賽者可進行一次或兩次表演，由評審決定誰能繼續比賽 [來源 3]。
3. 舞蹈比賽有兩種形式：一種是全風格混合比賽，另一種是專注於單一舞蹈風格的比賽，音樂會配合該風格 [來源 2]。
4. 街舞比賽起源於當地社區中心、公園或夜店的小型活動，參賽者可現場報名參賽 [來源 4]。
5. 街舞比賽作為一種在街舞圈中獲得成功、尊重和權威的合法化元素 [來源 5]。

Step 2 — 綜合回答：
Juste Debout 是一個在法國巴黎舉辦的全球大型街舞比賽，其中包含2對2的Popping舞蹈類別，舞者以配對形式競賽，並強調創意與原創性，營造出熱烈的比賽氣氛 [來源 1]。舞蹈比賽通常從資格賽開始，所有參賽者可進行一次或兩次表演，由評審決定誰能繼續比賽 [來源 3]。Juste Debout屬於專注於特定舞蹈風格的比賽，音樂會配合該舞蹈風格進行 [來源 2]。街舞比賽的歷史起源於當地社區中心、公園或夜店的小型活動，參賽者可現場報名參賽，這些比賽成為街舞圈中獲得成功、尊重和權威的重要途徑 [來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋Juste Debout比賽的詳細規則條文及其完整歷史發展過程。


---

## Q5: 如何判斷一場 Popping battle 的勝負？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.346 | 0.346 |
| F | 5 | 0.380 | 0.380 |
| G | 5 | 0.282 | 0.382 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.346（retriever 平均距離: 0.346）
- 引用: `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 0 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 15 (dense)...

Step 1 — 列出 Facts：
1. Popping battle 中兩位舞者各有一分鐘的表演時間，雙方面對面進行比賽，無積分制度，重點在於音樂而非競爭 [來源 1]。
2. 裁判會根據多項評分標準分別評判，尋找符合所有評分標準的「完整舞者」 [來源 1]。
3. Popping battle 的評分維度包括樂感、基礎、原創性、執行、技術、創造力、地板動作、表演力及服裝，其中樂感和基礎最為重要 [來源 4]。
4. Battle 是一種學習工具，結果反映舞者是否達到裁判的評分標準 [來源 1]。
5. Battle 與 cypher 是常見賽制，評判常關注音樂性、獨創性與完整度 [來源 5]。

Step 2 — 綜合回答：
判斷一場 Popping battle 的勝負，主要依靠裁判根據多項評分標準來評判。比賽中兩位舞者各有一分鐘的表演時間，雙方面對面進行，無積分制度，重點放在舞者與音樂的契合度。裁判會分別評估樂感（動作與音樂節奏、旋律、歌詞的對應程度）、基礎（基本動作的扎實程度）、原創性、執行、技術、創造力、地板動作、表演力及服裝等多個維度，並尋找符合所有標準的「完整舞者」。最終勝負反映舞者是否達到裁判的評分標準，並非單純以積分決定，強調比賽的學習與交流功能 [來源 1][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋具體裁判如何宣布勝負的程序或是否有特定勝負判定方式。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.380（retriever 平均距離: 0.380）
- 引用: `popping_style_taxonomy.txt` chunk 6 (both), `popping-culture-notes.txt` chunk 0 (both), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense)...

Step 1 — 列出 Facts：
1. Popping Battle 的評判維度包括 Musicality（樂感）、Foundation（基礎）、Originality（原創性）、Execution（執行）、Technique（技術）、Creativity（創造力）、Floor Work（地板動作）、Showmanship（表演力）、Costume（服裝）[來源 1]。
2. 評分標準中，Musicality 和 Foundation 是最重要的評分項目，均為五顆星重要性[來源 1]。
3. 1vs1 Popping Battles 中，沒有使用點數系統，評判重點在音樂與舞者表現，而非直接競爭分數[來源 3]。
4. 裁判會根據各項評分標準分別評判，尋找符合所有評分標準的「完整舞者（COMPLETE POPPER）」[來源 3]。
5. Battle 是兩位舞者的對抗性表演，由評審或觀眾判定勝負[來源 4]。

Step 2 — 綜合回答：
判斷一場 Popping battle 的勝負，裁判會根據多個評分維度進行評判，主要包括樂感（Musicality）、基礎動作（Foundation）、原創性（Originality）、執行力（Execution）、技術（Technique）、創造力（Creativity）、地板動作（Floor Work）、表演力（Showmanship）及服裝（Costume）等。其中，樂感與基礎動作是最重要的評分項目。裁判不使用點數系統，而是分別評估每個標準，尋找能夠在所有評分維度上表現完整的舞者，即「完整舞者（COMPLETE POPPER）」。最終勝負由裁判依據這些標準判定[來源 1][來源 3][來源 4]。

Step 3 — 缺口說明：
資料未涵蓋具體的勝負判定流程細節，例如如何在評分維度中權衡或決定勝者的具體方法。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.382（retriever 平均距離: 0.282）
- 擴展查詢: Popping battle 判定標準與勝負評分方法 / Criteria for winning a Popping dance battle / How to judge the winner in a Popping dance competition
- 引用: `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense), `popping_style_taxonomy.txt` chunk 6 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 9 (dense)...

Step 1 — 列出 Facts：
1. Popping battle 中兩位舞者各有一分鐘的表演時間，沒有使用積分系統判定勝負，重點在於音樂而非競爭 [來源 1]。
2. 裁判會根據多項評分標準分別評判，尋找符合所有標準的「完整舞者（COMPLETE POPPER）」 [來源 1][來源 5]。
3. 評分標準包含樂感（Musicality）、基礎（Foundation）、原創性（Originality）、執行（Execution）、技術（Technique）、創造力（Creativity）、地板動作（Floor Work）、表演力（Showmanship）、服裝（Costume），其中樂感與基礎最為重要 [來源 2]。
4. Popping battle 通常有1至5個回合，每回合雙方輪流表演 [來源 3]。
5. 比賽的目的在於學習與自我提升，結果反映舞者是否符合裁判標準，競爭是與自身潛力的較量，而非單純擊敗對手 [來源 1][來源 5]。

Step 2 — 綜合回答：
判斷一場 Popping battle 的勝負，並非透過傳統的積分系統，而是由裁判根據多項評分標準分別評判舞者的表現，尋找符合所有標準的「完整舞者（COMPLETE POPPER）」。評分標準包括樂感、基礎動作的扎實度、原創性、執行力、技術掌握、創造力、地板動作運用、表演力及服裝等，其中樂感與基礎最為重要。比賽通常分為1至5個回合，雙方輪流表演，每回合約一分鐘。整體來說，Popping battle 更強調舞者與音樂的契合及自我提升，勝負反映舞者是否達到裁判的評判標準，而非單純以分數決定誰贏誰輸 [來源 1][來源 2][來源 3][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋具體如何宣布勝負的程序或是否有裁判投票等細節。
