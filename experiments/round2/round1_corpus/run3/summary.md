# RAG 3 組實驗比較（E、F、G）

*生成時間: 2026-11-10 01:13 UTC；來源: 3 個 combo_*/results.json*

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
| 產生時間 | 2026-11-10T01:13:00+00:00 | 2026-11-10T01:13:26+00:00 | 2026-11-10T01:13:53+00:00 |

## 與原問題的平均距離

| Combo | 與原問題平均距離（全部題目的 chunks） |
|---|---|
| E | 0.386 |
| F | 0.443 |
| G | 0.468 |

---
## Q1: What is Popping and who created it?

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.355 | 0.355 |
| F | 5 | 0.462 | 0.462 |
| G | 5 | 0.187 | 0.466 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.355（retriever 平均距離: 0.355）
- 引用: `POPPING  societydance.txt` chunk 4 (dense), `History initiative to document the origins of ‘popping’ - Fresno State News.txt` chunk 0 (dense), `POPPING  societydance.txt` chunk 6 (dense)...

Step 1 — 列出 Facts：
1. Popping 是一種街舞，起源於1960年代末至1970年代的加州弗雷斯諾，屬於原始的放克舞蹈風格之一。舞蹈技巧基於快速收縮和放鬆肌肉，使舞者身體產生抽動，稱為pop或hit，並持續配合音樂節奏和各種動作與姿勢進行。[來源 1]
2. Popping 是由想在洛杉磯和奧克蘭跳舞時看起來像幫派分子的舞者創造的。其特徵是通過身體部分肌肉的抽動產生顫動動作。舞者在抽動時會輕聲說“Pop, Pop”，因此稱為Popping。Hitting 指的是更用力且突然的肌肉抽動，像是在嚇唬別人。[來源 3]
3. 歷史學家Robert Farris Thompson描述Popping為“軀幹和四肢的節奏性角度動作”，速度可快可慢。先驅舞者Timothy Solomon（又名Pop’in Pete）定義Popping為“用腿、脖子、頭部等關節用力擊打”。[來源 4]
4. Popin’ Pete是The Electric Boogaloos舞團的第一代成員，1978年加入該團並開始學習Popping風格。該團創始人Boogaloo Sam教導成員Popping的基礎動作，團體很快取得成功。[來源 5]

Step 2 — 綜合回答：
Popping是一種起源於1960年代末至1970年代加州弗雷斯諾的街舞，屬於原始的放克舞蹈風格。其舞蹈技巧基於快速收縮和放鬆肌肉，造成身體的抽動（稱為pop或hit），並配合音樂節奏和多樣動作進行。Popping由想在洛杉磯和奧克蘭跳舞時展現強悍形象的舞者創造，特徵是身體部分肌肉的顫動動作，舞者在抽動時會輕聲說“Pop, Pop”。Hitting則是更用力且突然的肌肉抽動。歷史學家和先驅舞者也對Popping有具體定義，強調節奏性和關節用力擊打的動作。Popin’ Pete是著名的先驅舞者之一，並是The Electric Boogaloos舞團的成員，該團由Boogaloo Sam創立並教授Popping基礎，推動了這種舞蹈風格的發展與成功。[來源 1][來源 3][來源 4][來源 5]

Step 3 — 缺口說明：
資料未涵蓋Popping的具體創始人姓名，但提及Boogaloo Sam為The Electric Boogaloos創始人並教授Popping基礎，暗示其在Popping發展中的重要角色。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.462（retriever 平均距離: 0.462）
- 引用: `POPPING  societydance.txt` chunk 6 (both), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 2 (both), `POPPING  societydance.txt` chunk 4 (dense)...

Step 1 — 列出 Facts：
1. Popping 是一種街舞，起源於1960年代末至1970年代的加州弗雷斯諾，屬於原始的放克舞蹈風格之一，舞蹈技巧是快速收縮和放鬆肌肉，造成身體的抽動，稱為pop或hit，並配合音樂節奏和各種動作與姿勢進行 [來源 3]。
2. Popping 是由想要像洛杉磯和奧克蘭的幫派分子一樣看起來強悍的舞者創造的，特徵是身體部分肌肉的抽動，舞者在抽動時會發出“Pop, Pop”的聲音，因此得名Popping [來源 1]。
3. Popping的根源可追溯到放克音樂時代，先驅者包括Don Campbell、Boogaloo Sam和Popin' Pete，他們融合了放克音樂與流暢的身體動作，奠定了現代Popping的基礎 [來源 2]。

Step 2 — 綜合回答：
Popping是一種起源於1960年代末至1970年代加州弗雷斯諾的街舞，屬於原始的放克舞蹈風格之一。其舞蹈技巧是透過快速收縮和放鬆肌肉，造成身體的抽動（稱為pop或hit），並配合音樂節奏和各種動作與姿勢進行。Popping由想要模仿洛杉磯和奧克蘭幫派分子強悍形象的舞者所創造，舞者在抽動肌肉時會發出“Pop, Pop”的聲音，因此得名。Popping的發展受到放克音樂時代的影響，先驅者如Don Campbell、Boogaloo Sam和Popin' Pete融合放克音樂與流暢的身體動作，奠定了現代Popping的基礎[來源 1][來源 2][來源 3]。

Step 3 — 缺口說明：
資料未涵蓋其他可能的創始人或更詳細的歷史背景。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.466（retriever 平均距離: 0.187）
- 擴展查詢: Popping 舞蹈的起源與創始人是誰？ / History and origin of Popping dance style / Who invented Popping and what defines this dance technique?
- 引用: `What Is Popping (Popping Dance)  STEEZY Blog.txt` chunk 0 (dense), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 1 (dense), `What Is Popping (Popping Dance)  STEEZY Blog.txt` chunk 2 (dense)...

Step 1 — 列出 Facts：
1. Popping is a style of dance that started in the late 1960’s and 70’s [來源 1]。
2. Popping is characterized by sudden tensing and releasing of the muscles ("hitting") to the rhythm of beats in music [來源 1]。
3. Popping originated primarily in California in the 1970s [來源 2]。
4. The dance started among groups of teenage friends in deindustrializing parts of the city, who created moves, practiced them, and showed them off at clubs or dance battles [來源 4]。
5. The term “Popping” (also called “hitting”) describes the collection of moves and techniques invented by the OG Poppers [來源 3]。
6. Boogaloo Sam invented the Boogaloo style, a part of Popping [來源 3]。

Step 2 — 綜合回答：
Popping is a dance style that emerged in the late 1960s and 1970s, primarily in California. It is characterized by the sudden tensing and releasing of muscles, known as "hitting," synchronized to the rhythm of music beats. The dance originated among groups of teenage friends in deindustrializing urban areas who created and practiced moves before showcasing them in clubs and dance battles. The term "Popping" refers to the collection of moves and techniques developed by the original Poppers (OG Poppers). One notable creator within this dance culture is Boogaloo Sam, who invented the Boogaloo style, a fluid and loose motion within Popping [來源 1][來源 2][來源 3][來源 4]。

Step 3 — 缺口說明：
資料未涵蓋具體的個人或團體被認定為整體Popping舞蹈的唯一創始者。


---

## Q2: Electric Boogaloos 的成員有哪些？他們各自的貢獻是什麼？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.358 | 0.358 |
| F | 5 | 0.392 | 0.392 |
| G | 5 | 0.197 | 0.442 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.358（retriever 平均距離: 0.358）
- 引用: `Popin' Pete - Wikipedia.txt` chunk 3 (dense), `Popin' Pete - Wikipedia.txt` chunk 6 (dense), `Popin' Pete - Wikipedia.txt` chunk 4 (dense)...

Step 1 — 列出 Facts：
1. Boogaloo Sam 創立了 The Electric Boogaloos，並教導成員基本動作 [來源 2][來源 5]。
2. Popin' Pete 是 The Electric Boogaloos 的成員之一 [來源 5]。
3. 其他成員包括 Skeeter Rabbit、Suga Pop、Mr. Wiggles、Jazzy J、Creepin Cid、Puppet Boozer、Robot Dane [來源 5]。
4. The Electric Boogaloos 推廣了 popping 和相關舞蹈風格，並在多個知名節目中表演 [來源 2][來源 4]。

Step 2 — 綜合回答：
Electric Boogaloos 的成員包括 Boogaloo Sam、Popin' Pete、Skeeter Rabbit、Suga Pop、Mr. Wiggles、Jazzy J、Creepin Cid、Puppet Boozer 和 Robot Dane。其中，Boogaloo Sam 是該團體的創始人，負責編寫基本動作並教導成員，對團體的成立和風格形成有關鍵貢獻 [來源 2][來源 5]。Popin' Pete 作為成員之一，與團體共同推廣 popping 舞蹈風格 [來源 5]。整個團體因其對 popping 及相關舞蹈風格的普及，以及在多個美國知名節目如 The Midnight Special、Kicks 和 Soul Train 的表演而聞名 [來源 2][來源 4]。

Step 3 — 缺口說明：
資料中未具體說明除 Boogaloo Sam 和 Popin' Pete 外其他成員的個別貢獻。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.392（retriever 平均距離: 0.392）
- 引用: `Popin' Pete - Wikipedia.txt` chunk 3 (both), `Popin' Pete - Wikipedia.txt` chunk 4 (both), `Electric boogaloo (dance) - Wikipedia.txt` chunk 3 (both)...

Step 1 — 列出 Facts：
1. Popin' Pete 編輯了一套基本動作並創立了 The Electric Boogaloos [來源 1]。
2. Electric Boogaloos 是1970年代中期的舞蹈團體，推廣了 electric boogaloo 舞風及 popping 和相關風格 [來源 3]。
3. Mr. Wiggles 與 The Electric Boogaloos 有關聯 [來源 4]。
4. Suga Pop 曾與 Electric Boogaloos 有關聯 [來源 5]。

Step 2 — 綜合回答：
Electric Boogaloos 是一個1970年代中期成立的舞蹈團體，創始人之一是 Popin' Pete，他編輯了一套基本動作並創立了該團體，推廣了 electric boogaloo 舞風及 popping 和相關風格 [來源 1][來源 3]。此外，成員中還包括與該團體有關聯的 Mr. Wiggles 和 Suga Pop，他們也在推廣和表演這些舞蹈風格中扮演角色 [來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋 Electric Boogaloos 的完整成員名單及各自具體貢獻的詳細說明。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.442（retriever 平均距離: 0.197）
- 擴展查詢: Electric Boogaloos 成員名單及其舞蹈風格貢獻 / Who are the members of the Electric Boogaloos and what are their individual contributions to popping and boogaloo? / Electric Boogaloos key dancers and their impact on the development of Fresno and popping techniques
- 引用: `Boogaloo (funk dance) - Wikipedia.txt` chunk 1 (dense), `Electric boogaloo (dance) - Wikipedia.txt` chunk 3 (dense), `Electric boogaloo (dance) - Wikipedia.txt` chunk 2 (dense)...

Step 1 — 列出 Facts：
1. Boogaloo Sam 是 Electric Boogaloos 的創始人，並教導成員基本舞步 [來源 4]。
2. Electric Boogaloos 成員皆為經驗豐富且熱情的舞者，團體很快獲得成功 [來源 4]。
3. 成員包括 Popin' Pete（提及於來源 4，與其他成員一同出現在多個美國知名節目） [來源 4]。
4. Electric Boogaloos 推廣了 electric boogaloo 舞蹈風格，並普及了 popping 及相關舞蹈風格 [來源 2]。

Step 2 — 綜合回答：
Electric Boogaloos 的成員包括創始人 Boogaloo Sam，他負責教導團體成員基本舞步，為團體奠定基礎。此外，成員如 Popin' Pete 等都是經驗豐富且熱情的舞者，這使得團體迅速獲得成功。Electric Boogaloos 不僅成為 mid-1970s 電子 boogaloo 舞蹈風格的代表團體，也推廣了 popping 舞蹈及其相關風格，並多次出現在美國知名節目如 The Midnight Special、Kicks 和 Soul Train 中 [來源 2][來源 4]。

Step 3 — 缺口說明：
資料未涵蓋 Electric Boogaloos 其他具體成員的完整名單及各自詳細貢獻。


---

## Q3: Popping 和 Boogaloo 有什麼差別？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.388 | 0.388 |
| F | 5 | 0.420 | 0.420 |
| G | 5 | 0.197 | 0.423 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.388（retriever 平均距離: 0.388）
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `popping-terminology-snippet.txt` chunk 0 (dense), `POPPING  societydance.txt` chunk 3 (dense)...

Step 1 — 列出 Facts：
1. Popping 是一種街舞風格，包含肌肉快速收縮和放鬆以配合節拍的動作，並且是一系列由原始 Poppers 發明的動作和技巧的總稱。[來源 3][來源 4]
2. Boogaloo 是 Popping 中的一種風格，特徵是身體各部位的流暢滾動動作，包括滾動臀部、膝蓋和頭部，結合了「機器人舞」和模仿表演的平滑隔離動作。[來源 2][來源 3]
3. Boogaloo Sam 創立了 Electric Boogaloo Lockers（後來稱為 Electric Boogaloos），並且他是 Boogaloo 風格的主要推廣者。[來源 4][來源 5]
4. Popping 也被稱為 hitting，是一種包含多種動作和技巧的舞蹈風格，而 Boogaloo 是其中一種特定的流派，強調流暢的滾動動作。[來源 3][來源 4]

Step 2 — 綜合回答：
Popping 是一種街舞風格，主要特點是肌肉快速收縮和放鬆以配合節拍，並包含多種動作和技巧的集合，這些動作由最早的 Poppers 創造並發展起來。[來源 3][來源 4] Boogaloo 則是 Popping 中的一種特定風格，強調身體各部位的流暢滾動動作，如滾動臀部、膝蓋和頭部，並融合了機器人舞和平滑隔離動作的元素。[來源 2][來源 3] Boogaloo 風格由 Boogaloo Sam 創立並推廣，他同時創立了著名的舞團 Electric Boogaloos。[來源 4][來源 5] 因此，Popping 是一個包含多種風格和技巧的總稱，而 Boogaloo 是其中一種以流暢滾動動作為特色的風格。[來源 3][來源 4]

Step 3 — 缺口說明：
資料未涵蓋其他可能的差異細節，如音樂選擇或文化背景的不同。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.420（retriever 平均距離: 0.420）
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (both), `Electric boogaloo (dance) - Wikipedia.txt` chunk 2 (both), `Boogaloo (funk dance) - Wikipedia.txt` chunk 24 (both)...

Step 1 — 列出 Facts：
1. Boogaloo 是一種即興的街舞，與 popping 舞蹈密切相關，並且以機器人舞（the Robot）動作聞名，也與後來的 electric boogaloo 舞蹈相關。[來源 5]
2. Electric boogaloo 是一種舞蹈風格，與早期在奧克蘭表演的 Boogaloo 街舞及 popping 密切相關，結合了現代 popping 技巧和早期 boogaloo 形式。[來源 2]
3. 在1970年代初，Black Messengers 團體的舞者創新了一種稱為 "Posing Hard" 的 Boogaloo 技巧，這種技巧會在動作結束時硬停並凍結姿勢，影響了現代 popping 技巧。[來源 3]
4. Popping 是指肌肉快速收縮和放鬆以配合節拍的舞蹈技巧，Boogaloo Sam 在1970年代中期創立了 Electric Boogaloo Lockers（後稱 Electric Boogaloos），並由他口中發出的“pop”一詞命名了 popping。[來源 4]

Step 2 — 綜合回答：
Popping 和 Boogaloo 兩者都是街舞風格且彼此密切相關，但有其不同之處。Boogaloo 是一種即興的街舞風格，著重於流暢的動作和機器人舞（the Robot）等特定動作，並且是 electric boogaloo 舞蹈的前身。[來源 5][來源 2] Popping 則是一種舞蹈技巧，強調肌肉的快速收縮與放鬆以配合節拍，這種技巧部分源自於 Boogaloo 中的 "Posing Hard" 技巧，即在動作結束時硬停並凍結姿勢。[來源 3][來源 4] Electric boogaloo 則是結合了 Boogaloo 與 popping 技巧的舞蹈風格，代表了兩者的融合。[來源 2]

Step 3 — 缺口說明：
資料未涵蓋兩者在舞蹈風格上更細節的技術差異或表演風格上的具體區別。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.423（retriever 平均距離: 0.197）
- 擴展查詢: Popping 與 Boogaloo 的主要區別是什麼 / Differences between Popping and Boogaloo dance styles / How does Boogaloo differ from Popping in street dance techniques?
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 1 (dense), `Boogaloo (funk dance) - Wikipedia.txt` chunk 2 (dense)...

Step 1 — 列出 Facts：
1. Popping 是一種起源於1970年代加州的街舞風格，透過節奏性地收縮和放鬆肌肉，創造出獨特的「pop」或「hit」效果 [來源 2]。
2. Boogaloo 是一種即興的街舞，與 popping 密切相關，並以機器人舞（Robot）動作聞名 [來源 5]。
3. Boogaloo 舞者利用肌肉限制、停止動作、機器人動作和擺動，創造出充滿靈魂和激情的街舞形式，並融合了 popping 的基礎技巧 [來源 3]。
4. Boogaloo 的特點是身體各部位的流暢運用，包括臀部、膝蓋和頭部的滾動，結合了機器人舞和模仿表演的平滑隔離動作 [來源 4]。
5. Popping 包含多種動作和技巧的集合，Boogaloo 是其中一種風格，且 Boogaloo 與 popping 及 turfing 等街舞風格密切相關 [來源 5]。

Step 2 — 綜合回答：
Popping 是一種起源於1970年代加州的街舞風格，透過節奏性地收縮和放鬆肌肉，創造出獨特的「pop」或「hit」效果，屬於一個包含多種動作和技巧的舞蹈類型 [來源 2][來源 4]。Boogaloo 則是 popping 的一種即興街舞風格，著重於身體各部位的流暢運用，如臀部、膝蓋和頭部的滾動，並融合了機器人舞（Robot）和平滑的隔離動作，形成一種充滿靈魂和激情的舞蹈形式 [來源 3][來源 4][來源 5]。簡言之，Boogaloo 是 popping 這個大類別下的一種具體風格，兩者密切相關但 Boogaloo 更強調流暢性和表演性 [來源 3][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋兩者在技術細節或歷史發展上的更深入差異。


---

## Q4: Juste Debout 比賽的規則和歷史？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.474 | 0.474 |
| F | 5 | 0.562 | 0.562 |
| G | 5 | 0.370 | 0.638 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.474（retriever 平均距離: 0.474）
- 引用: `Salah (dancer) - Wikipedia.txt` chunk 66 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 22 (dense), `Salah (dancer) - Wikipedia.txt` chunk 19 (dense)...

Step 1 — 列出 Facts：
- <context> 中未提及 Juste Debout 比賽的規則。
- <context> 中未提及 Juste Debout 比賽的歷史。

Step 2 — 綜合回答：
資料未涵蓋 Juste Debout 比賽的規則和歷史。

Step 3 — 缺口說明：
本次提供的 <context> 中沒有包含任何關於 Juste Debout 比賽的規則或歷史資訊，無法回答此問題。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.562（retriever 平均距離: 0.562）
- 引用: `Juste Debout.txt` chunk 4 (both), `Salah (dancer) - Wikipedia.txt` chunk 66 (dense), `Juste Debout.txt` chunk 0 (bm25)...

Step 1 — 列出 Facts：
1. Juste Debout 是世界上最大的街舞比賽之一，在法國巴黎舉辦，包含 popping 項目，舞者以 2 對 2 形式競賽，強調創意和原創性，氣氛熱烈 [來源 5]。
2. Juste Debout 於 2025 年 3 月 2 日在巴黎 Accor Arena 舉行決賽，項目包括 Locking、Popping、Hip-Hop Newstyle、House Dance、Experimental、Junior Dance Tour PLUS Afro、Dancehall、Krump 和 Electro [來源 1]。
3. Juste Debout 是國際最大的 Hip-Hop 聚會之一，被稱為 2026 年的 #1 街舞活動 [來源 3]。
4. 比賽從 2025 年 11 月 30 日到 2025 年 3 月 6 日進行全球預選賽 [來源 1]。
5. 關於評審制度，雖然未明確指出 Juste Debout 的具體評分規則，但一般街舞比賽中，評審會在每回合後投票選出勝者，並可能有加賽回合（tiebreak round）以決定勝負，Kozen 評分系統會在每回合後即時顯示分數 [來源 4]。

Step 2 — 綜合回答：
Juste Debout 是世界上最大的街舞比賽之一，於法國巴黎舉辦，包含多種舞蹈項目如 Locking、Popping、Hip-Hop Newstyle、House Dance 等，並於 2025 年 3 月 2 日在巴黎 Accor Arena 舉行決賽。比賽強調創意與原創性，並以 2 對 2 的形式進行 popping 項目競賽。整個比賽從 2025 年 11 月 30 日開始，持續到 2025 年 3 月 6 日進行全球預選賽。Juste Debout 被視為國際最大的 Hip-Hop 聚會之一，是 2026 年最重要的街舞活動。關於評審規則，雖未明確說明 Juste Debout 的具體評分方式，但一般街舞比賽會在每回合後由評審投票決定勝者，並可能設有加賽回合以打破平手，Kozen 評分系統則會在每回合後即時顯示分數，這些評審方式可能被採用於此類比賽中 [來源 1][來源 3][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋 Juste Debout 比賽具體的詳細規則和評分標準。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.638（retriever 平均距離: 0.370）
- 擴展查詢: Juste Debout competition rules and history / Juste Debout 比賽的起源與規則介紹 / What are the origins and judging criteria of Juste Debout dance battles?
- 引用: `Guarato.Dance.Articulated.txt` chunk 29 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 6 (dense), `Guarato.Dance.Articulated.txt` chunk 64 (dense)...

Step 1 — 列出 Facts：
1. Juste Debout 是世界上最大的街舞比賽之一，在法國巴黎舉辦，包含 popping 類別，舞者以 2 對 2 形式競賽，強調創意和原創性，舞者在充滿活力的氛圍中展示技能。[來源 4]

Step 2 — 綜合回答：
Juste Debout 是一個在法國巴黎舉辦的世界大型街舞比賽，包含 popping 類別，舞者以 2 對 2 的形式競賽。該比賽強調舞者的創意和原創性，參賽者在充滿活力的環境中展示他們的舞蹈技巧。[來源 4]

Step 3 — 缺口說明：
資料未涵蓋 Juste Debout 比賽的詳細規則和歷史背景。


---

## Q5: 如何判斷一場 Popping battle 的勝負？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.355 | 0.355 |
| F | 5 | 0.380 | 0.380 |
| G | 5 | 0.284 | 0.372 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.355（retriever 平均距離: 0.355）
- 引用: `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 0 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 15 (dense)...

Step 1 — 列出 Facts：
1. Popping battle 中兩位選手各有一分鐘的表演時間，並且沒有使用積分系統來決定勝負，重點放在音樂性而非競爭本身。[來源 1]
2. 評審會針對每個評分標準分別進行評判，尋找符合所有標準的「完整的 Popping 選手」(COMPLETE POPPER) 。[來源 1][來源 5]
3. 競賽的目的是作為學習工具，結果反映選手是否達到評審的標準，強調自我成長與理解 Popping 的過程，而非單純擊敗對手。[來源 1][來源 5]
4. Popping 賽事中評審常關注音樂性、獨創性與完整度等要素。[來源 4]

Step 2 — 綜合回答：
在 Popping battle 中，勝負並非透過積分系統來判定，而是由評審根據多項標準分別評判，尋找能夠展現音樂性、獨創性與完整度的「完整的 Popping 選手」。評審會根據這些標準來決定誰更符合「完整的 Popping 選手」的條件，並以此作為勝負的依據。整個競賽強調的是選手對 Popping 的理解與自我成長，競爭的本質是與自己的潛力較量，而非單純擊敗對手。[來源 1][來源 4][來源 5]

Step 3 — 缺口說明：
資料未涵蓋具體評審如何在沒有積分系統下做出勝負判定的詳細流程或標準細節。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.380（retriever 平均距離: 0.380）
- 引用: `popping-culture-notes.txt` chunk 0 (both), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 2 (both), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 9 (both)...

Step 1 — 列出 Facts：
1. Popping battle 通常是兩位舞者面對面進行，每人有一分鐘的回合時間 [來源 2][來源 4]。
2. 評判不使用積分制，而是根據多項標準分別評分，尋找符合所有標準的「完整的 Popper」 [來源 4]。
3. 評判標準包括音樂性、獨創性、完整度、舞台表現力、創意、服裝風格等 [來源 1][來源 2]。
4. 評判重點是舞者如何與音樂互動，而非單純的動作或錯誤 [來源 2][來源 4]。

Step 2 — 綜合回答：
判斷一場 Popping battle 的勝負，評判會根據多項標準分別評分，這些標準包括音樂性、獨創性、完整度、舞台表現力、創意以及服裝風格。比賽中兩位舞者會輪流在一分鐘的回合中展現表演，評判會尋找最能涵蓋所有評分標準的「完整的 Popper」。比賽不採用積分制，重點在於舞者如何與音樂互動，而非僅看動作或錯誤。最終勝負由評判根據上述標準綜合判定 [來源 2][來源 4][來源 1]。

Step 3 — 缺口說明：
資料未涵蓋具體評判標準的詳細分數分配或評判流程細節。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.372（retriever 平均距離: 0.284）
- 擴展查詢: Popping battle 判定標準與勝負評分方式 / Criteria for judging winners in a Popping dance battle / How to decide the outcome of a Popping dance competition
- 引用: `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 9 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 12 (dense)...

Step 1 — 列出 Facts：
1. Popping battle 中兩位舞者各有一分鐘的表演時間，沒有使用積分系統判定勝負，重點在於音樂性而非競爭本身。[來源 1]
2. 評審會針對各項評分標準分別評判，尋找符合所有標準的「完整的 Popping 舞者」作為勝出者。[來源 1]
3. 評審結果反映舞者是否達到評審標準，競賽本身也被視為學習工具。[來源 1]
4. Popping battle 常見的賽制包括 battle 與 cypher，評審關注音樂性、獨創性與完整度。[來源 4]

Step 2 — 綜合回答：
在 Popping battle 中，勝負並非透過積分系統決定，而是由評審根據多項標準分別評判，重點在於舞者對音樂的詮釋與展現。評審會尋找同時符合音樂性、獨創性與完整度等標準的「完整的 Popping 舞者」，並以此決定勝負。整場比賽強調的是舞者與音樂的互動，而非直接競爭的分數高低，且比賽結果同時也是舞者學習與成長的依據。[來源 1][來源 4]

Step 3 — 缺口說明：
資料未涵蓋具體評分標準的詳細內容及評審如何具體判斷勝負的流程。
