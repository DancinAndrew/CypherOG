# RAG 3 組實驗比較（E、F、G）

*生成時間: 2026-11-10 01:12 UTC；來源: 3 個 combo_*/results.json*

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
| 產生時間 | 2026-11-10T01:11:29+00:00 | 2026-11-10T01:11:54+00:00 | 2026-11-10T01:12:24+00:00 |

## 與原問題的平均距離

| Combo | 與原問題平均距離（全部題目的 chunks） |
|---|---|
| E | 0.386 |
| F | 0.443 |
| G | 0.474 |

---
## Q1: What is Popping and who created it?

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.355 | 0.355 |
| F | 5 | 0.462 | 0.462 |
| G | 5 | 0.189 | 0.466 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.355（retriever 平均距離: 0.355）
- 引用: `POPPING  societydance.txt` chunk 4 (dense), `History initiative to document the origins of ‘popping’ - Fresno State News.txt` chunk 0 (dense), `POPPING  societydance.txt` chunk 6 (dense)...

Step 1 — 列出 Facts：
1. Popping 是一種街舞，起源於1960年代末至1970年代的加州弗雷斯諾，屬於原始的放克舞蹈風格之一。其舞蹈技巧是快速收縮和放鬆肌肉，造成身體的抽動，稱為pop或hit，並配合音樂節奏和各種動作與姿勢進行 [來源 1]。
2. Popping 是由想在洛杉磯和奧克蘭跳舞時看起來像幫派份子的舞者創造的，特徵是通過身體部分肌肉的彎曲產生抽動動作。舞者在彎曲肌肉時會發出“Pop, Pop”的聲音，因此稱為Popping。當肌肉用力且突然彎曲時稱為Hitting [來源 3]。
3. 歷史學家Robert Farris Thompson描述popping為“軀幹和四肢的節奏性角度動作”，節奏可快可慢。先驅舞者Timothy Solomon（又名Pop’in Pete）定義popping為“用腿、脖子、頭等關節用力擊打” [來源 4]。
4. Popin’ Pete是電波舞團體The Electric Boogaloos的第一代成員，1978年加入該團體並開始學習popping風格。該團體創始人Boogaloo Sam教導成員popping基礎，團體很快取得成功 [來源 5]。

Step 2 — 綜合回答：
Popping是一種起源於1960年代末至1970年代加州弗雷斯諾的街舞風格，屬於原始的放克舞蹈之一。其舞蹈技巧基於快速收縮和放鬆肌肉，造成身體的抽動（稱為pop或hit），並配合音樂節奏和多樣動作與姿勢進行。Popping由想在洛杉磯和奧克蘭跳舞時展現強悍形象的舞者創造，他們在彎曲肌肉時會發出“Pop, Pop”的聲音，因此得名。這種舞蹈強調肌肉的突然用力彎曲（稱為hitting），並融合了模仿海浪動作的waving元素。歷史學家和先驅舞者如Timothy Solomon（Pop’in Pete）對popping有具體定義，強調用關節用力擊打的動作。Popin’ Pete作為The Electric Boogaloos的成員，於1978年開始學習並推廣popping，該團體由Boogaloo Sam創立並教授popping基礎，促使這種舞蹈風格迅速流行起來 [來源 1][來源 3][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋具體創造者的完整名單或單一創始人身份。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.462（retriever 平均距離: 0.462）
- 引用: `POPPING  societydance.txt` chunk 6 (both), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 2 (both), `POPPING  societydance.txt` chunk 4 (dense)...

Step 1 — 列出 Facts：
1. Popping 是一種街舞，起源於1960年代末至1970年代的加州弗雷斯諾，屬於原始的放克舞蹈風格之一，舞蹈技巧是快速收縮和放鬆肌肉，造成身體的抽動，稱為 pop 或 hit，並持續配合音樂節奏和各種動作與姿勢進行 [來源 3]。
2. Popping 是由想要像洛杉磯和奧克蘭的幫派成員一樣看起來強悍的舞者創造的，特徵是身體部分肌肉的抽動，舞者在抽動時會低聲說“Pop, Pop”，因此稱為 Popping，當肌肉更用力且突然收縮時稱為 Hitting [來源 1]。
3. Popping 的根源可追溯到放克音樂時代，先驅者包括 Don Campbell、Boogaloo Sam 和 Popin' Pete，他們將放克音樂與流暢的身體動作融合，奠定了現今 Popping 的基礎 [來源 2]。

Step 2 — 綜合回答：
Popping 是一種起源於1960年代末至1970年代加州弗雷斯諾的街舞，屬於原始的放克舞蹈風格。其舞蹈技巧主要是透過快速收縮和放鬆肌肉，造成身體的抽動（稱為 pop 或 hit），並持續配合音樂節奏和各種動作與姿勢進行。Popping 是由想要像洛杉磯和奧克蘭的幫派成員一樣看起來強悍的舞者所創造，舞者在抽動肌肉時會低聲說“Pop, Pop”，因此得名。當肌肉更用力且突然收縮時，稱為 Hitting。Popping 的發展受到放克音樂時代的影響，先驅者包括 Don Campbell、Boogaloo Sam 和 Popin' Pete，他們將放克音樂與流暢的身體動作融合，奠定了現代 Popping 的基礎 [來源 1][來源 2][來源 3]。

Step 3 — 缺口說明：
資料未涵蓋具體哪一位舞者最早創造了 Popping。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.466（retriever 平均距離: 0.189）
- 擴展查詢: Popping 舞蹈的起源與創始人是誰？ / History and origin of Popping dance style / Who invented Popping and how did it develop in street dance culture?
- 引用: `What Is Popping (Popping Dance)  STEEZY Blog.txt` chunk 0 (dense), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 1 (dense), `What Is Popping (Popping Dance)  STEEZY Blog.txt` chunk 2 (dense)...

Step 1 — 列出 Facts：
1. Popping is a style of dance that started in the late 1960’s and 70’s [來源 1]。
2. Popping is characterized by sudden tensing and releasing of the muscles ("hitting") to the rhythm of beats in music [來源 1]。
3. Popping originated primarily in California in the 1970s [來源 2]。
4. The dance style started among groups of teenage friends in deindustrializing parts of the city, who created moves, practiced them, and showed them off at clubs or dance battles [來源 4]。
5. The term “Popping” (also called “hitting”) describes the collection of moves and techniques invented by the OG Poppers [來源 3]。
6. Boogaloo Sam invented the Boogaloo style, a key move in Popping [來源 3]。

Step 2 — 綜合回答：
Popping is a dance style that originated in the late 1960s and 1970s, primarily in California. It is characterized by the sudden tensing and releasing of muscles, known as "hitting," synchronized to the rhythm of music beats. The dance began among groups of teenage friends in deindustrializing urban areas who created and practiced moves before showcasing them in clubs and dance battles. The term "Popping" refers to the collection of moves and techniques developed by the original pioneers of the style, known as OG Poppers. Among these pioneers, Boogaloo Sam is credited with inventing the Boogaloo style, a fluid and loose movement that is a key component of Popping [來源 1][來源 2][來源 3][來源 4]。

Step 3 — 缺口說明：
資料未涵蓋具體的個人創始者（如是否有單一創始人）或更詳細的創始歷史背景。


---

## Q2: Electric Boogaloos 的成員有哪些？他們各自的貢獻是什麼？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.358 | 0.358 |
| F | 5 | 0.392 | 0.392 |
| G | 5 | 0.192 | 0.453 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.358（retriever 平均距離: 0.358）
- 引用: `Popin' Pete - Wikipedia.txt` chunk 3 (dense), `Popin' Pete - Wikipedia.txt` chunk 6 (dense), `Popin' Pete - Wikipedia.txt` chunk 4 (dense)...

Step 1 — 列出 Facts：
1. Electric Boogaloos 的創始人是 Boogaloo Sam，他編排了一套基本動作並創立了該團體 [來源 1]。
2. Boogaloo Sam 教授團體成員基本動作，團體成員都是經驗豐富且熱情的舞者，團體很快取得成功 [來源 2]。
3. Electric Boogaloos 成員包括 Boogaloo Sam、Popin Pete、Skeeter Rabbit、Suga Pop、Mr. Wiggles、Jazzy J、Creepin Cid、Puppet Boozer、Robot Dane 等 [來源 5]。
4. Electric Boogaloos 推廣了電波舞（electric boogaloo）、popping 及其相關舞蹈風格 [來源 4]。
5. Boogaloo Sam 被認為是 popping 一詞的創造者，因為他在跳舞時會嘴型模仿“pop”這個詞 [來源 5]。
6. Popin Pete 編排了一套基本動作並創立了 Electric Boogaloos [來源 1]。
7. Electric Boogaloos 成員曾在多個美國知名節目中表演，包括 The Midnight Special、Kicks 和 Soul Train [來源 2]。

Step 2 — 綜合回答：
Electric Boogaloos 的成員包括 Boogaloo Sam、Popin Pete、Skeeter Rabbit、Suga Pop、Mr. Wiggles、Jazzy J、Creepin Cid、Puppet Boozer 和 Robot Dane 等。Boogaloo Sam 是團體的創始人，他編排了基本動作並教授成員，且被認為是 popping 一詞的創造者。Popin Pete 也編排了一套基本動作並共同創立了 Electric Boogaloos。團體成員都是經驗豐富且熱情的舞者，他們推廣了電波舞（electric boogaloo）、popping 及相關舞蹈風格，並在多個美國知名節目中表演，促進了這些舞蹈風格的普及 [來源 1][來源 2][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋各成員除 Boogaloo Sam 與 Popin Pete 之外的具體貢獻細節。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.392（retriever 平均距離: 0.392）
- 引用: `Popin' Pete - Wikipedia.txt` chunk 3 (both), `Popin' Pete - Wikipedia.txt` chunk 4 (both), `Electric boogaloo (dance) - Wikipedia.txt` chunk 3 (both)...

Step 1 — 列出 Facts：
1. Popin' Pete 編輯了一套基本動作並創立了 The Electric Boogaloos。[來源 1]
2. Electric Boogaloos 是1970年代中期的舞蹈團體，推廣了 electric boogaloo 舞風及 popping 和相關風格。[來源 3]
3. Mr. Wiggles 與 The Electric Boogaloos 有關聯。[來源 4]
4. Suga Pop 也與 Electric Boogaloos 有關聯。[來源 5]

Step 2 — 綜合回答：
Electric Boogaloos 的成員包括 Popin' Pete，他編輯了一套基本動作並創立了該團體，推廣了 electric boogaloo 舞風及 popping 和相關風格。[來源 1][來源 3] 此外，Mr. Wiggles 和 Suga Pop 也與 Electric Boogaloos 有關聯，分別作為成員或合作舞者，但具體貢獻未詳述。[來源 4][來源 5]

Step 3 — 缺口說明：
資料未涵蓋 Electric Boogaloos 其他成員的完整名單及各自詳細貢獻。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.453（retriever 平均距離: 0.192）
- 擴展查詢: Electric Boogaloos 成員名單及其舞蹈貢獻 / Who are the members of the Electric Boogaloos and what are their individual contributions? / Electric Boogaloos dancers and their roles in the development of popping and boogaloo styles
- 引用: `Boogaloo (funk dance) - Wikipedia.txt` chunk 1 (dense), `Electric boogaloo (dance) - Wikipedia.txt` chunk 3 (dense), `Electric boogaloo (dance) - Wikipedia.txt` chunk 2 (dense)...

Step 1 — 列出 Facts：
1. Boogaloo Sam 是 Electric Boogaloos 團體的創建者，並教導團體成員基本舞步 [來源 4]。
2. Electric Boogaloos 的成員都是非常有經驗且熱情的舞者，團體很快取得成功 [來源 4]。
3. 成員包括 Popin' Pete（從上下文中推測，因為該段落標題為 Popin' Pete，且提及 Electric Boogaloos 成員） [來源 4]。
4. Electric Boogaloos 成員曾在多個美國知名節目中表演，如 The Midnight Special、Kicks 和 Soul Train（首次出現於1980年） [來源 4]。

Step 2 — 綜合回答：
Electric Boogaloos 的成員包括 Boogaloo Sam（團體創建者，負責教導基本舞步）和 Popin' Pete 等有經驗且熱情的舞者。Boogaloo Sam 作為創建者，對團體的舞蹈基礎有重要貢獻，而其他成員則憑藉豐富的舞蹈經驗使團體迅速成功。該團體成員曾多次在美國知名節目如 The Midnight Special、Kicks 和 Soul Train 上表演，推廣他們的舞蹈風格 [來源 4]。

Step 3 — 缺口說明：
資料未涵蓋 Electric Boogaloos 其他具體成員的姓名及其個別貢獻。


---

## Q3: Popping 和 Boogaloo 有什麼差別？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.388 | 0.388 |
| F | 5 | 0.420 | 0.420 |
| G | 5 | 0.183 | 0.423 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.388（retriever 平均距離: 0.388）
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `popping-terminology-snippet.txt` chunk 0 (dense), `POPPING  societydance.txt` chunk 3 (dense)...

Step 1 — 列出 Facts：
1. Boogaloo 是一種流暢、滾動的動作詞彙，強調身體各部位的流動性，包括滾動臀部、膝蓋和頭部，是“機器人舞”和模仿動作的平滑隔離的結合 [來源 2][來源 3]。
2. Popping 是指肌肉快速收縮和放鬆以配合節拍的動作，也稱為“hitting”，是由 OG Poppers 發明的一系列動作和技巧的總稱 [來源 3][來源 4]。
3. Boogaloo Sam 創立了 Electric Boogaloo Lockers（後來稱為 Electric Boogaloos），並且他是將“popping”這個詞用於描述這種舞蹈動作的人 [來源 4][來源 5]。
4. Popin’ Pete 是 Electric Boogaloos 的成員，他先學會了 popping，因為 boogaloo 太難了，顯示兩者是不同但相關的舞蹈風格 [來源 5]。

Step 2 — 綜合回答：
Popping 是一種舞蹈技巧，特點是肌肉快速收縮和放鬆以配合節拍，這種動作也被稱為“hitting”，是由最早的 Poppers 創造的一系列動作和技巧的總稱 [來源 3][來源 4]。Boogaloo 則是一種特定的 popping 風格，強調身體各部位的流暢滾動動作，如滾動臀部、膝蓋和頭部，結合了“機器人舞”和模仿動作的平滑隔離 [來源 2][來源 3]。Boogaloo Sam 創立了 Electric Boogaloo Lockers（後來的 Electric Boogaloos），並且他是將“popping”這個詞用於描述這種舞蹈動作的人 [來源 4][來源 5]。此外，Electric Boogaloos 成員 Popin’ Pete 表示他先學會了 popping，因為 boogaloo 太難，這說明 popping 和 boogaloo 是相關但不同的舞蹈風格 [來源 5]。

Step 3 — 缺口說明：
資料未涵蓋更詳細的技術差異或歷史發展細節。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.420（retriever 平均距離: 0.420）
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (both), `Electric boogaloo (dance) - Wikipedia.txt` chunk 2 (both), `Boogaloo (funk dance) - Wikipedia.txt` chunk 24 (both)...

Step 1 — 列出 Facts：
1. Boogaloo 是一種自由風格的即興街舞，與 popping 舞蹈和 turfing 密切相關，著名動作包括機器人舞（Robot）[來源 5]。
2. Electric boogaloo 是一種舞蹈風格，結合了現代 popping 技巧和早期 boogaloo 形式，是 1970 年代中期舞團 Electric Boogaloos 的代表風格[來源 2]。
3. 1970 年代初，Black Messengers 舞者創新了一種稱為 "Posing Hard" 的 Boogaloo 技巧，這種技巧會在動作結束時用力停頓並保持肌肉震動，影響了現代 popping 技巧[來源 3]。
4. Popping 是指肌肉快速收縮和放鬆以配合節奏的舞蹈技巧，Boogaloo Sam 在 1970 年代中期創立了 Electric Boogaloo Lockers（後稱 Electric Boogaloos），他在跳 popping 時會口型模仿“pop”一詞[來源 4]。

Step 2 — 綜合回答：
Popping 和 Boogaloo 兩者都是街舞中的舞蹈風格，且彼此密切相關。Boogaloo 是一種自由風格的即興街舞，著名動作包括機器人舞，並且與 popping 舞蹈有關聯[來源 5]。在 1970 年代初，Boogaloo 舞者創新了 "Posing Hard" 技巧，這種技巧強調動作結束時的停頓和肌肉震動，這對後來的 popping 技巧產生了影響[來源 3]。Popping 則是一種透過肌肉快速收縮和放鬆來配合節奏的舞蹈技巧，Boogaloo Sam 創立的 Electric Boogaloo Lockers（後來的 Electric Boogaloos）將這種技巧發揚光大，並且 popping 這個詞就是由 Boogaloo Sam 在跳舞時口型模仿“pop”而來[來源 4]。此外，Electric boogaloo 是結合了現代 popping 技巧和早期 boogaloo 形式的舞蹈風格，是 1970 年代中期 Electric Boogaloos 舞團的代表風格[來源 2]。

Step 3 — 缺口說明：
資料未涵蓋兩者在具體舞步或風格細節上的差異描述。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.423（retriever 平均距離: 0.183）
- 擴展查詢: Popping 與 Boogaloo 的風格和動作差異 / Differences between Popping and Boogaloo dance styles / What distinguishes Boogaloo from Popping in street dance techniques
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `Boogaloo (funk dance) - Wikipedia.txt` chunk 2 (dense), `POPPING  societydance.txt` chunk 3 (dense)...

Step 1 — 列出 Facts：
1. Popping 是一種起源於1970年代加州的節奏舞蹈，透過肌肉的收縮與放鬆，配合音樂節拍，產生獨特的「pop」或「hit」效果。[來源 4]
2. Boogaloo 是一種即興的街舞風格，與 popping 舞蹈密切相關，並以機器人舞（Robot）動作聞名，也與後來的 electric boogaloo 舞蹈有關。[來源 5]
3. Boogaloo 舞者使用肌肉限制、停頓、機器人動作和擺動來創造一種充滿靈魂和熱情的街舞形式，並融合了 popping 的基礎技巧，最初稱為「Posing Hard」。[來源 2]
4. Boogaloo 的特點是身體各部位的流暢運用，包括臀部、膝蓋和頭部的滾動，結合了機器人舞和模仿動作的平滑隔離動作。[來源 3]

Step 2 — 綜合回答：
Popping 是一種節奏感強烈的舞蹈風格，起源於1970年代的加州，透過肌肉的快速收縮與放鬆來創造「pop」或「hit」的視覺效果，是一種以節奏為核心的舞蹈表現形式。[來源 4] 而 Boogaloo 則是與 popping 密切相關的即興街舞風格，著重於身體各部位的流暢動作和肌肉限制，並融合了機器人舞的元素，形成一種充滿靈魂和熱情的舞蹈形式。[來源 2][來源 3][來源 5] 簡言之，Boogaloo 是 popping 的一種風格或變體，強調流暢和機器人動作的結合，而 popping 則是一個更廣泛的舞蹈類別，包含多種技巧和風格。

Step 3 — 缺口說明：
資料未涵蓋更詳細的技術差異或歷史發展細節。


---

## Q4: Juste Debout 比賽的規則和歷史？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.474 | 0.474 |
| F | 5 | 0.562 | 0.562 |
| G | 5 | 0.375 | 0.648 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.474（retriever 平均距離: 0.474）
- 引用: `Salah (dancer) - Wikipedia.txt` chunk 66 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 22 (dense), `Salah (dancer) - Wikipedia.txt` chunk 19 (dense)...

Step 1 — 列出 Facts：
- <context> 中未提及 Juste Debout 比賽的規則。
- <context> 中未提及 Juste Debout 比賽的歷史。

Step 2 — 綜合回答：
資料未涵蓋 Juste Debout 比賽的規則和歷史。

Step 3 — 缺口說明：
<question> 中關於 Juste Debout 比賽的規則和歷史的資訊，<context> 未提供相關內容，無法回答。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.562（retriever 平均距離: 0.562）
- 引用: `Juste Debout.txt` chunk 4 (both), `Salah (dancer) - Wikipedia.txt` chunk 66 (dense), `Juste Debout.txt` chunk 0 (bm25)...

Step 1 — 列出 Facts：
1. Juste Debout 是世界上最大的街舞比賽之一，在法國巴黎舉辦，包含 popping 2 對 2 競賽類別，強調創意和原創性，舞者展示技巧並營造熱烈氣氛 [來源 5]。
2. Juste Debout 的決賽於 2025 年 3 月 2 日在巴黎 Accor Arena 舉行，包含 Locking、Popping、Hip-Hop Newstyle、House Dance、Experimental、Junior Dance Tour PLUS Afro、Dancehall、Krump 和 Electro 等舞蹈類別 [來源 1]。
3. Juste Debout 是國際最大的 Hip-Hop 聚會之一，號稱 2026 年第一街舞活動 [來源 3]。
4. 比賽從 2024 年 11 月 30 日到 2025 年 3 月 6 日進行全球預選賽 [來源 1]。
5. 評審制度中提到街舞比賽通常由評審在整場比賽後決定勝負，但有時會因記憶偏差影響判決，因此有 Kozen 評審系統，評審在每回合後選出勝者，並即時顯示分數 [來源 4]。

Step 2 — 綜合回答：
Juste Debout 是一個在法國巴黎舉辦的世界最大街舞比賽之一，涵蓋多種舞蹈類別，包括 Locking、Popping、Hip-Hop Newstyle、House Dance 等，並於 2025 年 3 月 2 日在 Accor Arena 舉行決賽。比賽從 2024 年 11 月 30 日開始進行全球預選賽，持續到 2025 年 3 月 6 日。Juste Debout 強調舞者的創意和原創性，並以 2 對 2 的 popping 競賽形式聞名。評審方面，街舞比賽通常由評審在整場比賽後決定勝負，但為避免記憶偏差，可能採用 Kozen 評審系統，即每回合後評審即時選出勝者並顯示分數，確保判決的公平與透明。Juste Debout 被視為國際最大的 Hip-Hop 聚會和街舞活動之一 [來源 1][來源 3][來源 4][來源 5]。

Step 3 — 缺口說明：
資料中未具體說明 Juste Debout 比賽的詳細規則條文、評分標準細節及歷史起源背景。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.648（retriever 平均距離: 0.375）
- 擴展查詢: Juste Debout competition rules and history / Juste Debout 比賽的起源與規則介紹 / What are the guidelines and background of Juste Debout dance battles?
- 引用: `Popping Dance Battles Near You — EVERYDAY POPPING - Online Popping Dance Tutorials.txt` chunk 1 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 7 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 2 (dense)...

Step 1 — 列出 Facts：
1. Juste Debout 是世界上最大的街舞比賽之一，在法國巴黎舉辦，包含 popping 項目，舞者以 2 對 2 的形式競賽，強調創意和原創性，氣氛熱烈 [來源 1]。
2. 舞蹈比賽通常從資格賽開始，所有參賽者可以展示一次或兩次，評審決定誰能繼續比賽 [來源 2]。
3. 舞蹈比賽有兩種形式：一種是 all-style，涵蓋多種街舞風格；另一種是專注於單一舞蹈風格，音樂也會針對該風格播放 [來源 3]。
4. 舞蹈比賽起源於小型活動，通常在社區中心、公園或夜店舉行，參賽者可現場報名參賽 [來源 4]。

Step 2 — 綜合回答：
Juste Debout 是一個在法國巴黎舉辦的世界大型街舞比賽，包含 popping 項目，採用 2 對 2 的競賽形式，強調舞者的創意和原創性，營造出熱烈的比賽氛圍 [來源 1]。一般舞蹈比賽會從資格賽開始，所有參賽者有機會展示一次或兩次，評審根據表現決定誰能繼續比賽 [來源 2]。舞蹈比賽可分為涵蓋多種街舞風格的 all-style 形式，或專注於單一舞蹈風格的比賽，後者的音樂會特別針對該舞蹈風格播放 [來源 3]。舞蹈比賽的歷史起源於小型的地方活動，通常在社區中心、公園或夜店舉行，參賽者可現場報名參加 [來源 4]。

Step 3 — 缺口說明：
資料未涵蓋 Juste Debout 比賽的詳細規則和完整歷史沿革。


---

## Q5: 如何判斷一場 Popping battle 的勝負？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.355 | 0.355 |
| F | 5 | 0.380 | 0.380 |
| G | 5 | 0.290 | 0.378 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.355（retriever 平均距離: 0.355）
- 引用: `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 0 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 15 (dense)...

Step 1 — 列出 Facts：
1. Popping battle 是兩位舞者面對面進行，每人有一分鐘的表演時間，沒有積分系統，重點在音樂而非競爭 [來源 1]。
2. 評審會針對各項標準分別評分，尋找符合所有標準的「完整舞者（COMPLETE POPPER）」 [來源 1][來源 5]。
3. 比賽結果反映舞者是否符合評審標準，並且比賽被視為學習和成長的工具，重點在於自我提升而非擊敗對手 [來源 1][來源 5]。
4. 評審常關注音樂性、獨創性與完整度 [來源 4]。

Step 2 — 綜合回答：
Popping battle 的勝負判斷並非依靠積分系統，而是由評審根據多項標準分別評分，尋找最符合所有標準的「完整舞者（COMPLETE POPPER）」。評審會特別關注舞者的音樂性、獨創性與完整度。比賽的結果反映舞者是否達到評審的標準，並且比賽本身被視為一個學習和成長的過程，強調與自身潛力的競爭而非與對手的較量 [來源 1][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋具體評審標準的詳細內容及評分細節。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.380（retriever 平均距離: 0.380）
- 引用: `popping-culture-notes.txt` chunk 0 (both), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 2 (both), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 9 (both)...

Step 1 — 列出 Facts：
1. Popping battle 中沒有使用積分系統，評判不以分數決定勝負，而是根據多項標準分別評判 [來源 4]。
2. 評判標準包括音樂性、獨創性、完整度、舞台表現力（娛樂性、表演及與觀眾的連結）、服裝（符合Popping風格）、地板動作等 [來源 1][來源 2]。
3. 評判尋找的是符合所有標準的「完整的Popper」 [來源 4]。
4. 每場battle由兩位poppers對戰，輪流表演1分鐘的回合 [來源 2][來源 4]。

Step 2 — 綜合回答：
判斷一場Popping battle的勝負並非依靠積分系統，而是由評判根據多項標準分別評判，包括音樂性、獨創性、完整度、舞台表現力（如娛樂性、表演及與觀眾的連結）、服裝是否符合Popping風格以及地板動作等。評判會尋找在這些標準中表現最全面、最符合「完整的Popper」形象的參賽者。每場battle由兩位poppers輪流進行1分鐘的表演回合，評判根據這些表現來決定勝負 [來源 1][來源 2][來源 4]。

Step 3 — 缺口說明：
資料未涵蓋具體評判勝負的程序細節，例如是否有投票、多數決或評判討論等過程。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.378（retriever 平均距離: 0.290）
- 擴展查詢: Popping battle 判定勝負的標準和技巧 / Criteria for judging the winner in a Popping dance battle / How to evaluate performance and decide the victor in a Popping competition
- 引用: `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 9 (dense), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 4 (dense)...

Step 1 — 列出 Facts：
1. Popping battle 是兩位舞者面對面進行，每人有一分鐘的表演時間，沒有使用積分系統判分，重點在於音樂和表現而非競爭本身。[來源 2]
2. 評審會根據多項標準分別評判，尋找符合所有標準的「完整Popper」(COMPLETE POPPER)。[來源 2]
3. 競賽結果反映舞者是否達到評審的標準，競賽被視為學習和成長的工具。[來源 3]

Step 2 — 綜合回答：
判斷一場 Popping battle 的勝負並非透過積分系統，而是由評審根據多項標準分別評判，尋找最符合所有評判標準的「完整Popper」。評審會關注舞者如何配合音樂及展現技巧，並以此決定勝負。整個競賽過程強調學習與成長，結果反映舞者是否達到評審的要求。[來源 2][來源 3]

Step 3 — 缺口說明：
資料未涵蓋具體的評判標準細節及評審如何做出最終勝負判定的具體流程。
