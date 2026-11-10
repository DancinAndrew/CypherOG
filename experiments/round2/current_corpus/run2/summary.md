# RAG 3 組實驗比較（E、F、G）

*生成時間: 2026-11-10 01:17 UTC；來源: 3 個 combo_*/results.json*

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
| 產生時間 | 2026-11-10T01:16:08+00:00 | 2026-11-10T01:16:36+00:00 | 2026-11-10T01:17:04+00:00 |

## 與原問題的平均距離

| Combo | 與原問題平均距離（全部題目的 chunks） |
|---|---|
| E | 0.379 |
| F | 0.460 |
| G | 0.463 |

---
## Q1: What is Popping and who created it?

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.355 | 0.355 |
| F | 5 | 0.532 | 0.532 |
| G | 5 | 0.187 | 0.466 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.355（retriever 平均距離: 0.355）
- 引用: `POPPING  societydance.txt` chunk 4 (dense), `History initiative to document the origins of ‘popping’ - Fresno State News.txt` chunk 0 (dense), `POPPING  societydance.txt` chunk 6 (dense)...

Step 1 — 列出 Facts：
1. Popping 是一種街舞，屬於原始的放克（funk）舞蹈風格，起源於1960年代末至1970年代的加州弗雷斯諾（Fresno, California）[來源 1]。
2. Popping 的舞蹈技巧是透過快速收縮和放鬆肌肉，造成身體的抽動，稱為 pop 或 hit，並持續配合音樂節奏和各種動作與姿勢[來源 1]。
3. Popping 是由想要像洛杉磯和奧克蘭的幫派成員一樣看起來強悍的舞者所創造，特徵是身體部分肌肉的抽動動作，舞者在抽動時會發出「Pop, Pop」的聲音，因此得名[來源 3]。
4. 舞者 Timothy Solomon（又名 Pop’in Pete）是先驅性舞者，他形容 popping 是「用關節猛擊，像用腿、脖子、頭猛擊」[來源 4]。
5. Popin’ Pete 是第一代 The Electric Boogaloos 成員，該團體由 Boogaloo Sam 創立，Boogaloo Sam 教導團員 popping 基本技巧[來源 5]。

Step 2 — 綜合回答：
Popping 是一種起源於1960年代末至1970年代加州弗雷斯諾的街舞風格，屬於原始的放克舞蹈。其舞蹈技巧是透過快速收縮和放鬆肌肉，造成身體的抽動（稱為 pop 或 hit），並配合音樂節奏與多樣動作和姿勢來表現。Popping 是由想要像洛杉磯和奧克蘭幫派成員一樣強悍的舞者所創造，舞者在抽動肌肉時會發出「Pop, Pop」的聲音，因此得名。先驅舞者 Timothy Solomon（Pop’in Pete）描述 popping 是用關節猛擊身體的動作。Popin’ Pete 是第一代 The Electric Boogaloos 成員，而該團體由 Boogaloo Sam 創立，Boogaloo Sam 教導團員 popping 的基本技巧[來源 1][來源 3][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋具體單一創作者的明確姓名，但提及 Boogaloo Sam 創立 The Electric Boogaloos 並教導 popping 基本技巧。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.532（retriever 平均距離: 0.532）
- 引用: `POPPING  societydance.txt` chunk 6 (both), `POPPING  societydance.txt` chunk 4 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 22 (bm25)...

Step 1 — 列出 Facts：
1. Popping 是一種街舞，起源於1960年代末至1970年代的加州弗雷斯諾（Fresno, California）[來源 2]。
2. Popping 的舞蹈技術基於快速收縮和放鬆肌肉，造成身體的抽動，稱為 pop 或 hit，舞者會隨著音樂節奏持續做出這種動作並搭配各種動作和姿勢[來源 2]。
3. Popping 是由想要像洛杉磯和奧克蘭的幫派成員一樣看起來強悍的舞者所創造[來源 1]。
4. 舞者在做出肌肉收縮動作時會低聲說「Pop, Pop」，因此這種舞蹈被稱為 Popping[來源 1]。

Step 2 — 綜合回答：
Popping 是一種起源於1960年代末至1970年代加州弗雷斯諾的街舞風格，屬於原始的放克（funk）舞蹈之一。它的核心技術是快速收縮和放鬆肌肉，造成身體的抽動（稱為 pop 或 hit），舞者會隨著音樂節奏持續做出這種動作並搭配各種動作和姿勢。Popping 是由那些想要像洛杉磯和奧克蘭的幫派成員一樣看起來強悍的舞者所創造，舞者在做出肌肉收縮動作時會低聲說「Pop, Pop」，因此這種舞蹈被稱為 Popping[來源 1][來源 2]。

Step 3 — 缺口說明：
資料未涵蓋具體創始人的姓名或個人資訊。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.466（retriever 平均距離: 0.187）
- 擴展查詢: Popping 舞蹈的起源與創始人是誰？ / History and origin of Popping dance style / Who invented Popping and what defines this dance technique?
- 引用: `What Is Popping (Popping Dance)  STEEZY Blog.txt` chunk 0 (dense), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 1 (dense), `What Is Popping (Popping Dance)  STEEZY Blog.txt` chunk 2 (dense)...

Step 1 — 列出 Facts：
1. Popping is a style of dance that started in the late 1960’s and 70’s [來源 1]。
2. Popping is characterized by sudden tensing and releasing of the muscles ("hitting") to the rhythm of beats in music [來源 1]。
3. Popping originated primarily in California in the 1970s [來源 2]。
4. The dance style started among groups of teenage friends in deindustrializing parts of the city, who created moves, practiced them, and showed them off at clubs or dance battles [來源 4]。
5. The term “Popping” (also called “hitting”) describes the collection of moves and techniques invented by the OG Poppers [來源 3]。
6. Boogaloo Sam invented the Boogaloo style, a key move in Popping [來源 3]。

Step 2 — 綜合回答：
Popping is a dance style that began in the late 1960s and 1970s, primarily in California. It is characterized by the sudden tensing and releasing of muscles, known as "hitting," synchronized to the rhythm of music beats. The dance originated among groups of teenage friends in deindustrializing urban areas, who developed and practiced moves before showcasing them in clubs and dance battles. The term "Popping" refers to the collection of moves and techniques created by the original pioneers, known as OG Poppers. Among these pioneers, Boogaloo Sam is credited with inventing the Boogaloo style, a fluid and loose movement within Popping [來源 1][來源 2][來源 3][來源 4]。

Step 3 — 缺口說明：
資料未涵蓋具體個人或團體整體創造 Popping 的完整創始者名單。


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
1. Boogaloo Sam 是 Electric Boogaloos 的創始人，並教導成員基本動作。[來源 2][來源 5]
2. Popin' Pete 是 Electric Boogaloos 的成員之一。[來源 1][來源 5]
3. Electric Boogaloos 成員還包括 Skeeter Rabbit、Suga Pop、Mr. Wiggles、Jazzy J、Creepin Cid、Puppet Boozer、Robot Dane 等。[來源 5]
4. Boogaloo Sam 創造了「popping」這個詞彙，並推廣了 popping 與相關舞蹈風格。[來源 4][來源 5]
5. Electric Boogaloos 成員都是經驗豐富且充滿熱情的舞者，團體很快取得成功，並在多個知名節目中亮相。[來源 2]

Step 2 — 綜合回答：
Electric Boogaloos 的成員包括 Boogaloo Sam、Popin' Pete、Skeeter Rabbit、Suga Pop、Mr. Wiggles、Jazzy J、Creepin Cid、Puppet Boozer 和 Robot Dane 等。[來源 5] Boogaloo Sam 是該團體的創始人，他編輯了一套基本舞步並教導其他成員，且創造了「popping」這個詞彙，推廣了 popping 舞蹈及其相關風格。[來源 2][來源 4][來源 5] Popin' Pete 作為成員之一，與其他成員一起在多個知名節目中表演，推動團體的成功。[來源 1][來源 2] 其他成員如 Skeeter Rabbit、Suga Pop 等也都是經驗豐富且熱情的舞者，為團體的表演和風格發展做出貢獻。[來源 5]

Step 3 — 缺口說明：
資料未涵蓋各成員具體的個別貢獻細節。

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
9. Boogaloo Sam 作為創始人，教導 Electric Boogaloos 成員基本動作，團體很快成功 [來源 5]。
10. Electric Boogaloos 成員曾在多個美國知名節目如 The Midnight Special、Kicks 和 Soul Train 出演 [來源 5]。

Step 2 — 綜合回答：
Electric Boogaloos 的成員包括 Boogaloo Sam（Sam Solomon）、Popin' Pete（Timothy Solomon）、Skeeter Rabbit（Michael Wayne Davis）、Sugar Pop / Suga Pop（Larry Vazquez）、Dane 1（Darryl James）和 Nate "Robot" Johnson（Nate Johnson）[來源 4]。Boogaloo Sam 是團體創始人，負責發展 Boogaloo 和 Popping 技術並教導成員[來源 4][來源 5]。Popin' Pete 是 Boogaloo Sam 的弟弟，作為創始成員以細膩的 Popping 技術著稱[來源 4]。Skeeter Rabbit 以 Boogaloo 和 Worm 動作聞名[來源 4]。Sugar Pop 擅長流暢的 Liquid 和 Animation 風格[來源 4]。Dane 1 是團體成員之一[來源 4]。Nate "Robot" Johnson 是 Robot 風格的代表人物[來源 4]。這些成員共同推動了 Electric Boogaloos 的成功，並曾多次在美國知名節目中亮相[來源 5]。

Step 3 — 缺口說明：
資料未涵蓋各成員更詳細的個別貢獻細節。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.435（retriever 平均距離: 0.197）
- 擴展查詢: Electric Boogaloos 成員名單及其舞蹈風格貢獻 / Who are the members of the Electric Boogaloos and what are their individual contributions to popping? / List of Electric Boogaloos dancers and their roles in developing Boogaloo and Fresno styles
- 引用: `Boogaloo (funk dance) - Wikipedia.txt` chunk 1 (dense), `Electric boogaloo (dance) - Wikipedia.txt` chunk 3 (dense), `Electric boogaloo (dance) - Wikipedia.txt` chunk 2 (dense)...

Step 1 — 列出 Facts：
1. Electric Boogaloos 是一個成立於 1977 年，由 Boogaloo Sam 在 Fresno 創立的舞團 [來源 5]。
2. Boogaloo Sam 是該舞團的創始人，並教授舞團成員基本舞步 [來源 4]。
3. Electric Boogaloos 的成員都是非常有經驗且熱情的舞者，舞團很快取得成功 [來源 4]。
4. 舞團成員包括 Popin' Pete，他們曾在多個美國知名節目中表演，如 The Midnight Special、Kicks 和 Soul Train（首次出現於 1980 年）[來源 4]。

Step 2 — 綜合回答：
Electric Boogaloos 是由 Boogaloo Sam 於 1977 年在 Fresno 成立的舞團。Boogaloo Sam 作為創始人，負責教授舞團成員基本舞步，對舞團的技術基礎有重要貢獻。舞團成員均為經驗豐富且充滿熱情的舞者，這使得舞團迅速獲得成功。成員之一 Popin' Pete 與其他成員曾多次在美國著名節目如 The Midnight Special、Kicks 和 Soul Train 中表演，推廣了他們的舞蹈風格 [來源 4][來源 5]。

Step 3 — 缺口說明：
資料中未具體列出 Electric Boogaloos 的所有成員名單及其各自詳細貢獻。


---

## Q3: Popping 和 Boogaloo 有什麼差別？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.368 | 0.368 |
| F | 5 | 0.390 | 0.390 |
| G | 5 | 0.203 | 0.415 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.368（retriever 平均距離: 0.368）
- 引用: `popping_style_taxonomy.txt` chunk 2 (dense), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `popping-terminology-snippet.txt` chunk 0 (dense)...

Step 1 — 列出 Facts：
1. Boogaloo 是 Popping 的子流派，不是 Popping 的同義詞；Boogaloo 更強調流動性，Popping 更強調爆點清晰度 [來源 1]。
2. Boogaloo 強調身體流動性（fluidity），以臀部、膝蓋的圓形律動為核心，比 Popping 更柔軟流暢。由 Sam Solomon（Boogaloo Sam）創立 [來源 5]。
3. Popping 更注重技術控制和爆點（hit）清晰度 [來源 1]。
4. Electric Boogaloo 結合 Boogaloo 的流動性和 Popping 的爆點，是整個 Popping 文化的代名詞 [來源 4]。

Step 2 — 綜合回答：
Popping 是一種街舞風格，強調爆點（hit）的清晰度和技術控制，而 Boogaloo 是 Popping 的一個子流派，起源於1970年代加州，由 Boogaloo Sam 創立。Boogaloo 更注重身體的流動性，以臀部和膝蓋的圓形律動為核心，動作較為柔軟流暢。簡言之，Boogaloo 強調身體的流動感，而 Popping 則強調爆點的明確和技術的精準。Electric Boogaloo 作為一個風格名稱，結合了 Boogaloo 的流動性和 Popping 的爆點特點，成為 Popping 文化的重要代表 [來源 1][來源 4][來源 5]。

Step 3 — 缺口說明：
資料涵蓋了 Popping 和 Boogaloo 的主要差別，無缺口。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.390（retriever 平均距離: 0.390）
- 引用: `popping_style_taxonomy.txt` chunk 5 (both), `popping_style_taxonomy.txt` chunk 2 (both), `popping_style_taxonomy.txt` chunk 1 (both)...

Step 1 — 列出 Facts：
1. Boogaloo 是 Popping 的子流派，更強調流動性，Popping 更強調爆點清晰度 [來源 2]。
2. Boogaloo 起源於1970s加州 Richmond / Fresno，由 Sam Solomon（Boogaloo Sam）創立，是 Popping 的源頭技術 [來源 3]。
3. Boogaloo 技術特徵包括全身性律動，以臀部和膝蓋為中心的圓形搖擺，動作流暢而非爆裂，身體每個關節都參與律動，製造蠕動感 [來源 3]。
4. Popping 是包含 Boogaloo 在內的更大類別 [來源 5]。
5. Popping 更注重技術控制和爆點技術，強調爆點清晰度 [來源 1]。

Step 2 — 綜合回答：
Popping 是一個包含多個子流派的舞蹈大類，其中 Boogaloo 是其子流派之一。Boogaloo 起源於1970年代加州，由 Boogaloo Sam 創立，是 Popping 的源頭技術。Boogaloo 強調全身性的流動律動，動作流暢且以臀部和膝蓋為中心的圓形搖擺，身體各關節協同製造蠕動感，整體風格較為液態和柔和。相較之下，Popping 更注重爆點技術的清晰度和技術控制，動作呈現出明顯的爆裂感。簡言之，Boogaloo 更強調流動性，而 Popping 更強調爆點的明確性和技術性 [來源 2][來源 3][來源 5][來源 1]。

Step 3 — 缺口說明：
資料已涵蓋問題所需資訊。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.415（retriever 平均距離: 0.203）
- 擴展查詢: Popping 與 Boogaloo 之間的區別是什麼？ / Differences between Popping and Boogaloo dance styles / Comparison of dance techniques in Popping versus Boogaloo
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 1 (dense), `POPPING  societydance.txt` chunk 3 (dense)...

Step 1 — 列出 Facts：
1. Popping 是一種起源於1970年代加州的街舞風格，透過肌肉的收縮與放鬆，配合音樂節奏，產生明顯的「pop」或「hit」效果。[來源 2]
2. Boogaloo 是一種即興的街舞，與 popping 舞蹈密切相關，並以機器人舞（the Robot）聞名，也與後來的 electric boogaloo 舞蹈有關。[來源 4]
3. Boogaloo 特徵是身體各部位的流暢運用，包括臀部、膝蓋和頭部的滾動，結合了「機器人舞」和模仿動作的平滑隔離動作。[來源 3]
4. 1970年代早期，Black Messengers 團體創新了一種稱為「Posing Hard」的 Boogaloo 技巧，舞者會在動作結束時用力「hit」並凍結姿勢，這影響了現代 popping 舞蹈中的「popping」技巧。[來源 5]

Step 2 — 綜合回答：
Popping 是一種起源於1970年代加州的街舞風格，透過肌肉的節奏性收縮與放鬆，創造出明顯的「pop」或「hit」效果，強調節奏感和肌肉控制。[來源 2] Boogaloo 則是一種即興的街舞，與 popping 密切相關，著重於身體各部位的流暢動作，如臀部、膝蓋和頭部的滾動，並融合了機器人舞和平滑的隔離動作。[來源 3][來源 4] Boogaloo 在1970年代早期發展出「Posing Hard」技巧，舞者會在動作結束時凍結姿勢並用力「hit」，這種技巧對後來 popping 舞蹈中的「popping」技巧有重要影響。[來源 5] 簡言之，Boogaloo 是一種包含流暢動作和凍結姿勢的街舞風格，而 popping 則更強調肌肉的節奏性收縮與放鬆，兩者風格相關但重點不同。

Step 3 — 缺口說明：
資料未涵蓋更詳細的技術差異或歷史演變細節。


---

## Q4: Juste Debout 比賽的規則和歷史？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.466 | 0.466 |
| F | 5 | 0.558 | 0.558 |
| G | 5 | 0.370 | 0.638 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.466（retriever 平均距離: 0.466）
- 引用: `Salah (dancer) - Wikipedia.txt` chunk 66 (dense), `popping_history_timeline.txt` chunk 7 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 22 (dense)...

Step 1 — 列出 Facts：
1. Juste Debout 是一個在巴黎舉辦的活動，早期聚焦 Popping 和 Locking。[來源 2]
2. 2002 年，Juste Debout 正式成為年度賽事，成為全球最重要的 Popping/Locking/House 2v2 比賽，吸引來自 60 多個國家的舞者參加。[來源 2]

Step 2 — 綜合回答：
Juste Debout 起初是在巴黎舉辦的活動，主要聚焦於 Popping 和 Locking 舞風。到了 2002 年，Juste Debout 正式成為年度賽事，並迅速發展成為全球最重要的 Popping、Locking 和 House 2 對 2 比賽，吸引了來自超過 60 個國家的舞者參加。[來源 2]

Step 3 — 缺口說明：
資料未涵蓋 Juste Debout 比賽的具體規則細節。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.558（retriever 平均距離: 0.558）
- 引用: `popping_history_timeline.txt` chunk 7 (both), `Salah (dancer) - Wikipedia.txt` chunk 66 (dense), `popping_style_taxonomy.txt` chunk 6 (bm25)...

Step 1 — 列出 Facts：
1. Juste Debout 的前身活動在巴黎舉辦，由 Bams 和 Anne Nguyen 組織，早期聚焦 Popping 和 Locking。[來源 1]
2. 2002 年，Juste Debout 正式成為年度賽事（巴黎），成為全球最重要的 Popping/Locking/House 2v2 比賽，吸引來自 60+ 個國家的舞者。[來源 1]
3. 根據 HHI 官方規則和 Juste Debout 評分標準，Popping Battle 的評判維度包括 Musicality（樂感）、Foundation（基礎）、Originality（原創性）、Execution（執行）、Technique（技術）、Creativity（創造力）、Floor Work（地板動作）、Showmanship（表演力）、Costume（服裝），其中 Musicality 和 Foundation 最重要。[來源 3]

Step 2 — 綜合回答：
Juste Debout 比賽起源於巴黎，最初由 Bams 和 Anne Nguyen 組織，早期活動主要聚焦於 Popping 和 Locking 兩種舞蹈風格。2002 年，Juste Debout 正式成為年度賽事，成為全球最重要的 Popping、Locking 和 House 2v2 比賽，吸引來自超過 60 個國家的舞者參加。[來源 1]  
在評分規則方面，Juste Debout 採用與 HHI 官方規則相似的評判標準，Popping Battle 的評分維度包括樂感（Musicality）、基礎動作的扎實程度（Foundation）、動作的原創性（Originality）、執行的完成度（Execution）、技術掌握（Technique）、創造力（Creativity）、地板動作的運用（Floor Work）、表演力（Showmanship）以及服裝（Costume）。其中，樂感和基礎動作被視為最重要的評分項目。[來源 3]

Step 3 — 缺口說明：
資料未涵蓋 Juste Debout 比賽的具體規則細節（如比賽流程、回合數、勝負判定方式等）。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.638（retriever 平均距離: 0.370）
- 擴展查詢: Juste Debout competition rules and history / Juste Debout 比賽的起源與規則介紹 / What are the origins and judging criteria of Juste Debout dance battles?
- 引用: `Guarato.Dance.Articulated.txt` chunk 29 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 6 (dense), `Guarato.Dance.Articulated.txt` chunk 64 (dense)...

Step 1 — 列出 Facts：
1. Juste Debout 是世界上最大的街舞比賽之一，在法國巴黎舉行，包含 popping 類別，舞者以兩人一組（2 on 2）形式競賽，強調創意和原創性，舞者在充滿活力的氛圍中展示技巧。[來源 4]

Step 2 — 綜合回答：
Juste Debout 是一個在法國巴黎舉行的全球大型街舞比賽，設有 popping 類別，舞者以兩人一組（2 on 2）的形式進行比賽。該比賽強調舞者的創意和原創性，參賽者在充滿活力的氛圍中展示他們的舞蹈技巧。[來源 4]

Step 3 — 缺口說明：
資料未涵蓋 Juste Debout 比賽的詳細規則和歷史背景。


---

## Q5: 如何判斷一場 Popping battle 的勝負？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.346 | 0.346 |
| F | 5 | 0.380 | 0.380 |
| G | 5 | 0.292 | 0.362 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.346（retriever 平均距離: 0.346）
- 引用: `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 0 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 15 (dense)...

Step 1 — 列出 Facts：
1. Popping battle 中兩位舞者各有一分鐘的表演時間，彼此面對面進行較量，沒有使用積分系統，重點放在音樂而非競爭本身。[來源 1]
2. 裁判會根據多項評分標準分別評判，尋找符合所有標準的「完整舞者」作為勝出者。[來源 1]
3. Popping battle 的評分標準包含樂感（Musicality）、基礎（Foundation）、原創性（Originality）、執行（Execution）、技術（Technique）、創造力（Creativity）、地板動作（Floor Work）、表演力（Showmanship）及服裝（Costume），其中樂感與基礎最為重要。[來源 4]
4. Battle 是一種學習工具，結果反映舞者是否達到裁判的評判標準。[來源 1]
5. Battle 與 cypher 是常見的賽制，評判常關注音樂性、獨創性與完整度。[來源 5]

Step 2 — 綜合回答：
判斷一場 Popping battle 的勝負，主要依靠裁判根據多項評分標準對兩位舞者分別評判。比賽中兩位舞者各有一分鐘的表演時間，彼此面對面較量，且不使用積分系統，強調與音樂的契合度。裁判會從樂感（動作與音樂的節奏、旋律、歌詞的對應程度）、基礎（基本動作的扎實度）、原創性（動作是否具有個人特色）、執行（動作完成度）、技術（特定技術動作掌握）、創造力（動作組合和即興創作）、地板動作、表演力及服裝等多個維度分別評判，尋找同時符合所有標準的「完整舞者」作為勝出者。整體來說，樂感與基礎是最重要的評分標準。Battle 不僅是競賽，更是一種學習工具，結果反映舞者是否達到裁判的評判標準。[來源 1][來源 4][來源 5]

Step 3 — 缺口說明：
資料未涵蓋具體判定勝負的程序細節，例如是否由多位裁判投票決定勝者或其他具體勝負判定方式。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.380（retriever 平均距離: 0.380）
- 引用: `popping_style_taxonomy.txt` chunk 6 (both), `popping-culture-notes.txt` chunk 0 (both), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense)...

Step 1 — 列出 Facts：
1. Popping Battle 的評判維度包括 Musicality（樂感）、Foundation（基礎）、Originality（原創性）、Execution（執行）、Technique（技術）、Creativity（創造力）、Floor Work（地板動作）、Showmanship（表演力）、Costume（服裝）[來源 1]。
2. 評分標準中，Musicality 和 Foundation 是最重要的兩個維度，均為五顆星重要性[來源 1]。
3. 1vs1 Popping Battles 中，兩位舞者各有一分鐘回合，評判不採用點數系統，而是根據各項標準分別評判[來源 3]。
4. 裁判尋找的是符合所有評判標準的「完整的 Popper」，並根據這些標準判定勝負[來源 3]。
5. Battle 是兩位舞者的對抗性表演，由評審或觀眾判定勝負[來源 4]。

Step 2 — 綜合回答：
判斷一場 Popping battle 的勝負，主要依據裁判對舞者在多個評分維度上的表現進行評估。這些維度包括樂感（Musicality）、基礎動作的扎實程度（Foundation）、原創性（Originality）、動作執行的清晰度（Execution）、技術掌握（Technique）、創造力（Creativity）、地板動作的運用（Floor Work）、表演力（Showmanship）以及服裝（Costume）。其中，樂感和基礎動作是最重要的評分標準。比賽中，兩位舞者各有一分鐘的表演時間，裁判不使用點數系統，而是根據各項標準分別評判，尋找能夠展現所有評判標準的完整舞者（Complete Popper），最終由裁判判定勝負[來源 1][來源 3][來源 4]。

Step 3 — 缺口說明：
資料未涵蓋具體裁判如何在無點數系統下進行勝負判定的詳細流程或標準。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.362（retriever 平均距離: 0.292）
- 擴展查詢: Popping battle 判定勝負的標準與技巧 / Criteria for winning a Popping dance battle / How to judge the winner in a Popping dance competition
- 引用: `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 9 (dense), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense), `popping_style_taxonomy.txt` chunk 6 (dense)...

Step 1 — 列出 Facts：
1. Popping battle 通常是 1 對 1 的對決，每位舞者有 1 分鐘的表演時間，沒有使用積分系統，重點在於音樂感而非競爭分數 [來源 2]。
2. 裁判會根據多個評分維度分別評判，尋找符合所有評判標準的「完整舞者」 [來源 2][來源 3]。
3. 評分維度包括：樂感（Musicality）、基礎（Foundation）、原創性（Originality）、執行（Execution）、技術（Technique）、創造力（Creativity）、地板動作（Floor Work）、表演力（Showmanship）、服裝（Costume）等，其中樂感和基礎最為重要 [來源 3]。
4. Battle 的結果反映舞者是否達到裁判的評判標準，競賽本身是一種學習工具 [來源 2]。

Step 2 — 綜合回答：
判斷一場 Popping battle 的勝負主要依靠裁判根據多項評分標準來決定。每位舞者在 1 分鐘的表演中展現自己的技巧，裁判會分別評估樂感、基礎動作的扎實度、原創性、執行力、技術掌握、創造力、地板動作運用、表演力以及服裝等方面。特別重視樂感和基礎動作的清晰度。雖然沒有使用積分系統，但裁判會綜合這些標準選出最符合「完整舞者」條件的勝者。整個過程強調對音樂的感受和舞者的全面表現，競賽同時也是學習和成長的機會 [來源 2][來源 3]。

Step 3 — 缺口說明：
資料未涵蓋具體裁判如何宣布勝負的程序或細節。
