# RAG 3 組實驗比較（E、F、G）

*生成時間: 2026-11-10 01:10 UTC；來源: 3 個 combo_*/results.json*

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
| 產生時間 | 2026-11-10T01:10:00+00:00 | 2026-11-10T01:10:27+00:00 | 2026-11-10T01:10:53+00:00 |

## 與原問題的平均距離

| Combo | 與原問題平均距離（全部題目的 chunks） |
|---|---|
| E | 0.386 |
| F | 0.443 |
| G | 0.466 |

---
## Q1: What is Popping and who created it?

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.355 | 0.355 |
| F | 5 | 0.462 | 0.462 |
| G | 5 | 0.207 | 0.409 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.355（retriever 平均距離: 0.355）
- 引用: `POPPING  societydance.txt` chunk 4 (dense), `History initiative to document the origins of ‘popping’ - Fresno State News.txt` chunk 0 (dense), `POPPING  societydance.txt` chunk 6 (dense)...

Step 1 — 列出 Facts：
1. Popping 是一種街舞，起源於1960年代末至1970年代的加州弗雷斯諾，屬於原始的放克舞蹈風格之一。其舞蹈技巧基於快速收縮和放鬆肌肉，造成身體的抽動，稱為pop或hit，並配合音樂節奏和各種動作及姿勢進行 [來源 1]。
2. Popping 是由想要像洛杉磯和奧克蘭的幫派成員一樣看起來強悍的舞者創造的。其特徵是由於身體部分肌肉的屈伸而產生的抽動動作。舞者在屈伸肌肉時會低聲說“Pop, Pop”，因此稱為Popping。Hitting指的是更用力且突然的肌肉屈伸，像是在嚇唬別人 [來源 3]。
3. 歷史學家Robert Farris Thompson描述Popping為“軀幹和四肢的節奏性角度變化”，速度可快可慢。先驅舞者Timothy Solomon（Pop’in Pete）定義Popping為“用腿、脖子、頭部等關節用力擊打” [來源 4]。
4. Popin’ Pete是The Electric Boogaloos舞團的第一代成員，1978年加入該團體並開始學習Popping。Boogaloo Sam是該團體的創始人，教導成員Popping的基本技巧 [來源 5]。

Step 2 — 綜合回答：
Popping是一種起源於1960年代末至1970年代加州弗雷斯諾的街舞風格，屬於原始的放克舞蹈之一。其舞蹈技巧基於快速收縮和放鬆肌肉，造成身體的抽動（稱為pop或hit），並配合音樂節奏和各種動作及姿勢進行。Popping由想要像洛杉磯和奧克蘭幫派成員一樣強悍的舞者創造，舞者在屈伸肌肉時會低聲說“Pop, Pop”，因此得名。Hitting則是更用力且突然的肌肉屈伸動作。歷史學家和先驅舞者也對Popping有具體定義，強調其節奏性和關節用力擊打的特點。Popin’ Pete是著名的先驅舞者，並在1978年加入The Electric Boogaloos舞團，該團體由Boogaloo Sam創立並教授Popping基本技巧 [來源 1][來源 3][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋具體個人創造者的完整名單或詳細創作過程。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.462（retriever 平均距離: 0.462）
- 引用: `POPPING  societydance.txt` chunk 6 (both), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 2 (both), `POPPING  societydance.txt` chunk 4 (dense)...

Step 1 — 列出 Facts：
1. Popping 是一種街舞，起源於1960年代末至1970年代的加州弗雷斯諾，屬於原始的放克舞蹈風格之一，舞蹈技巧是快速收縮和放鬆肌肉，造成身體的抽動，稱為 pop 或 hit，並配合音樂節奏和各種動作與姿勢持續進行 [來源 3]。
2. Popping 是由想要像洛杉磯和奧克蘭的幫派分子一樣看起來強硬的舞者創造的，特點是身體部分肌肉的抽動動作，舞者在抽動時會喃喃自語“Pop, Pop”，因此得名 Popping。Hitting 是指更用力且突然的肌肉抽動，像是在嚇唬別人 [來源 1]。
3. Popping 的根源可追溯到放克音樂時代，先驅者包括 Don Campbell、Boogaloo Sam 和 Popin' Pete，他們將放克音樂與流暢的身體動作融合，奠定了現代 Popping 的基礎 [來源 2]。

Step 2 — 綜合回答：
Popping 是一種起源於1960年代末至1970年代加州弗雷斯諾的街舞，屬於原始的放克舞蹈風格。這種舞蹈的特點是透過快速收縮和放鬆肌肉，造成身體的抽動（稱為 pop 或 hit），並配合音樂節奏和各種動作與姿勢持續進行。Popping 是由想要模仿洛杉磯和奧克蘭幫派分子強硬形象的舞者創造的，舞者在抽動肌肉時會喃喃自語“Pop, Pop”，因此得名。這種舞蹈的根源可追溯到放克音樂時代，先驅者包括 Don Campbell、Boogaloo Sam 和 Popin' Pete，他們將放克音樂與流暢的身體動作融合，奠定了現代 Popping 的基礎 [來源 1][來源 2][來源 3]。

Step 3 — 缺口說明：
資料未涵蓋具體單一創始人或創造者的詳細身份。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.409（retriever 平均距離: 0.207）
- 擴展查詢: Popping 舞蹈的起源和創始人是誰 / History and key figures behind the dance style Popping / Who invented Popping and what defines this dance technique?
- 引用: `What Is Popping (Popping Dance)  STEEZY Blog.txt` chunk 2 (dense), `What Is Popping (Popping Dance)  STEEZY Blog.txt` chunk 0 (dense), `What Is Popping (Popping Dance)  STEEZY Blog.txt` chunk 1 (dense)...

Step 1 — 列出 Facts：
1. Popping is a style of dance that started in the late 1960’s and 70’s in the U.S. [來源 2][來源 5]
2. This dance style is characterized by sudden tensing and releasing of the muscles ("hitting") to the rhythm of beats in music. [來源 2][來源 5]
3. Popping originated among groups of teenage friends in deindustrializing parts of the city, who created moves, practiced, and showcased them in clubs or dance battles. [來源 3]
4. Boogaloo Sam was a key figure in the creation of Popping and founded the group ‘The Electric Boogaloos’ in 1977. [來源 5]
5. Boogaloo Sam invented the Boogaloo, a loose, fluid motion that is part of Popping dance. [來源 1][來源 5]

Step 2 — 綜合回答：
Popping is a dance style that originated in the United States during the late 1960s and 1970s. It is characterized by the sudden tensing and releasing of muscles, known as "hitting," performed rhythmically to music beats. The dance began among groups of teenage friends in deindustrializing urban areas, who developed and practiced moves before performing them in clubs or dance battles. A central figure in the creation of Popping is Boogaloo Sam, who founded the influential dance group ‘The Electric Boogaloos’ in 1977 and invented the Boogaloo style, which is a fluid, bone-less body motion incorporated into Popping. [來源 2][來源 3][來源 5][來源 1]

Step 3 — 缺口說明：
資料未涵蓋其他創始人或更詳細的起源故事。


---

## Q2: Electric Boogaloos 的成員有哪些？他們各自的貢獻是什麼？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.358 | 0.358 |
| F | 5 | 0.392 | 0.392 |
| G | 5 | 0.188 | 0.520 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.358（retriever 平均距離: 0.358）
- 引用: `Popin' Pete - Wikipedia.txt` chunk 3 (dense), `Popin' Pete - Wikipedia.txt` chunk 6 (dense), `Popin' Pete - Wikipedia.txt` chunk 4 (dense)...

Step 1 — 列出 Facts：
1. Electric Boogaloos 的創始人是 Boogaloo Sam，他編制了一套基本動作並教導成員 [來源 1][來源 2][來源 5]。
2. Electric Boogaloos 的成員包括 Boogaloo Sam、Popin Pete、Skeeter Rabbit、Suga Pop、Mr. Wiggles、Jazzy J、Creepin Cid、Puppet Boozer、Robot Dane [來源 5]。
3. Boogaloo Sam 是該團體的創建者，並且他創造了“popping”這個詞彙，因為他在跳舞時會嘴型模仿“pop” [來源 5]。
4. Popin Pete 編制了一套基本動作並創立了 Electric Boogaloos [來源 1]。
5. Electric Boogaloos 成員都是經驗豐富且充滿熱情的舞者，團體很快獲得成功，並曾在多個美國知名節目中表演，包括 The Midnight Special、Kicks 和 Soul Train [來源 2]。
6. Electric Boogaloos 推廣了 electric boogaloo 舞風以及 popping 和相關風格 [來源 4]。

Step 2 — 綜合回答：
Electric Boogaloos 的成員包括 Boogaloo Sam、Popin Pete、Skeeter Rabbit、Suga Pop、Mr. Wiggles、Jazzy J、Creepin Cid、Puppet Boozer 和 Robot Dane。Boogaloo Sam 是該團體的創始人，他編制了基本動作並教導成員，同時他創造了“popping”這個詞彙。Popin Pete 也編制了一套基本動作並創立了 Electric Boogaloos。其他成員都是經驗豐富且充滿熱情的舞者，團體很快獲得成功，並在多個知名節目中表演。Electric Boogaloos 推廣了 electric boogaloo 舞風以及 popping 和相關風格 [來源 1][來源 2][來源 4][來源 5]。

Step 3 — 缺口說明：
其他成員如 Skeeter Rabbit、Suga Pop、Mr. Wiggles 等的具體貢獻未在資料中詳細說明，資料未涵蓋。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.392（retriever 平均距離: 0.392）
- 引用: `Popin' Pete - Wikipedia.txt` chunk 3 (both), `Popin' Pete - Wikipedia.txt` chunk 4 (both), `Electric boogaloo (dance) - Wikipedia.txt` chunk 3 (both)...

Step 1 — 列出 Facts：
1. Popin' Pete 編輯並創立了 The Electric Boogaloos [來源 1]。
2. The Electric Boogaloos 是1970年代中期的舞蹈團體，推廣了 electric boogaloo 舞風和 popping 及其相關風格 [來源 3]。
3. Mr. Wiggles 與 The Electric Boogaloos 有關聯 [來源 4]。
4. Suga Pop 也與 Electric Boogaloos 有關聯 [來源 5]。

Step 2 — 綜合回答：
Electric Boogaloos 的成員包括 Popin' Pete，他編輯並創立了該團體，並推廣了 electric boogaloo 舞風及 popping 等相關舞蹈風格 [來源 1][來源 3]。此外，Mr. Wiggles 和 Suga Pop 也與 Electric Boogaloos 有關聯，暗示他們是成員之一或與團體有合作關係，但具體貢獻未詳述 [來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋 Electric Boogaloos 其他成員的完整名單及各自具體貢獻。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.520（retriever 平均距離: 0.188）
- 擴展查詢: Electric Boogaloos 成員名單及其舞蹈風格貢獻 / Who were the members of the Electric Boogaloos and what were their individual contributions? / Electric Boogaloos key dancers and their roles in popularizing Popping and Boogaloo styles
- 引用: `Boogaloo (funk dance) - Wikipedia.txt` chunk 1 (dense), `Electric boogaloo (dance) - Wikipedia.txt` chunk 3 (dense), `POPPING  societydance.txt` chunk 3 (dense)...

Step 1 — 列出 Facts：
- Electric Boogaloos 是一支1970年代中期的舞蹈團體，代表風格為 electric boogaloo 舞蹈，並且推廣 popping 及相關舞蹈風格。[來源 2][來源 5]

Step 2 — 綜合回答：
Electric Boogaloos 是1970年代中期一支著名的舞蹈團體，他們的代表風格是 electric boogaloo 舞蹈，並且在推廣 popping 舞蹈及其相關風格方面有重要貢獻。[來源 2][來源 5]

Step 3 — 缺口說明：
資料未涵蓋 Electric Boogaloos 的具體成員名單及各自的貢獻。


---

## Q3: Popping 和 Boogaloo 有什麼差別？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.388 | 0.388 |
| F | 5 | 0.420 | 0.420 |
| G | 5 | 0.208 | 0.420 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.388（retriever 平均距離: 0.388）
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `popping-terminology-snippet.txt` chunk 0 (dense), `POPPING  societydance.txt` chunk 3 (dense)...

Step 1 — 列出 Facts：
1. Boogaloo 是一種以流暢、滾動動作為特徵的舞蹈風格，包含身體各部位的流動運用，如滾動臀部、膝蓋和頭部，結合了「機器人舞」和模仿動作的平滑隔離技巧 [來源 2][來源 3]。
2. Popping 是一種肌肉快速收縮和放鬆以配合節奏的舞蹈技術，通常稱為「hitting」，是由原始 Poppers 創造的一系列動作和技巧的總稱 [來源 3][來源 4]。
3. Boogaloo Sam 創立了 Electric Boogaloo Lockers（後稱 Electric Boogaloos）團體，並且他創造了「popping」這個詞，描述他在跳舞時嘴裡模仿「pop」的聲音 [來源 4]。
4. Popin’ Pete 是 Electric Boogaloos 的成員，他先學會 popping，因為 boogaloo 太難，Boogaloo Sam 教導團體成員 boogaloo 的基礎 [來源 5]。

Step 2 — 綜合回答：
Popping 是一種舞蹈技術，主要特點是肌肉快速收縮和放鬆以配合音樂節奏，這種技術被稱為「hitting」，是由早期的 Poppers 創造的一系列動作和技巧的總稱。而 Boogaloo 則是一種具體的舞蹈風格，強調身體各部位的流暢滾動動作，如滾動臀部、膝蓋和頭部，結合了機器人舞和平滑的隔離動作。Boogaloo 是 Electric Boogaloos 團體的創始人 Boogaloo Sam 所創，並且他也創造了「popping」這個詞。簡言之，Boogaloo 是 Popping 技術中的一種風格，兩者的差別在於 Boogaloo 更注重流暢的滾動動作，而 Popping 是指整體的肌肉收縮放鬆技術和相關動作 [來源 2][來源 3][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋更詳細的技術差異或歷史演變細節。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.420（retriever 平均距離: 0.420）
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (both), `Electric boogaloo (dance) - Wikipedia.txt` chunk 2 (both), `Boogaloo (funk dance) - Wikipedia.txt` chunk 24 (both)...

Step 1 — 列出 Facts：
1. Electric boogaloo 是一種舞蹈風格，與早期在奧克蘭表演的 Boogaloo 街舞和 popping 密切相關，結合了現代 popping 技巧和早期 boogaloo 形式 [來源 2]。
2. Boogaloo 是一種自由風格的即興街舞，與 popping 舞蹈和 turfing 密切相關，以機器人舞步聞名，也與後來的 electric boogaloo 舞蹈有關 [來源 5]。
3. 在1970年代初，Black Messengers 團體的舞者創新了一種稱為 "Posing Hard" 的 Boogaloo 技巧，這種技巧會在動作結束時硬停並凍結姿勢，這影響了現代 popping 舞蹈中的 popping 技巧 [來源 3]。
4. popping 舞蹈風格融合了肌肉快速收縮和放鬆以配合節拍的動作，而 Boogaloo 則包含更多的擺動和滾動動作（如臀部和腿部） [來源 4]。
5. Boogaloo Sam 在70年代中期成立了 Electric Boogaloo Lockers（後改名為 Electric Boogaloos），並且 popping 這個詞是由 Boogaloo Sam 在跳舞時口中發出的“pop”聲而命名的 [來源 4]。

Step 2 — 綜合回答：
Popping 和 Boogaloo 是兩種密切相關但有不同特點的街舞風格。Boogaloo 是一種自由即興的街舞，著重於身體的擺動和滾動動作，如臀部和腿部的動作，並以機器人舞步聞名。Popping 則強調肌肉的快速收縮和放鬆，配合節拍產生“pop”效果。Boogaloo 舞者在1970年代初期創新了“Posing Hard”技巧，即在動作結束時凍結姿勢，這對後來 popping 的技巧產生了影響。Electric Boogaloo 是結合了現代 popping 技巧和早期 Boogaloo 形式的舞蹈風格，並由 Boogaloo Sam 和 Electric Boogaloos 舞團推廣。popping 這個詞據說是由 Boogaloo Sam 在跳舞時發出的聲音而得名 [來源 2][來源 3][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋更詳細的技術差異或歷史演變細節。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.420（retriever 平均距離: 0.208）
- 擴展查詢: Popping 與 Boogaloo 的主要區別是什麼 / Differences between Popping and Boogaloo dance styles / How does Fresno differ from Popping and Boogaloo in street dance?
- 引用: `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 0 (dense), `Exploring Different Popping Styles Boogaloo, Waving, and More – Panic 39.txt` chunk 1 (dense), `POPPING  societydance.txt` chunk 3 (dense)...

Step 1 — 列出 Facts：
1. Popping 是一種起源於1970年代加州的街舞風格，透過節奏性地收縮和放鬆肌肉，創造出獨特的「pop」或「hit」效果。[來源 2]
2. Boogaloo 是一種結合「The Robot」和模仿動作（mime）平滑隔離動作的舞蹈風格，強調身體各部位的流暢運用，如臀部、膝蓋和頭部的滾動。[來源 3]
3. Boogaloo 舞者使用幻覺效果、肌肉限制、停頓、機械人動作和擺動，創造出充滿靈魂和激情的街舞形式，並融合了基礎的 popping 技巧，這些技巧最初被稱為「Posing Hard」。[來源 5]
4. Electric Boogaloo 是一種與 Boogaloo 和 popping 密切相關的舞蹈風格，結合了現代 popping 技巧和早期 Boogaloo 形式。[來源 4]

Step 2 — 綜合回答：
Popping 是一種起源於1970年代加州的街舞風格，透過節奏性地收縮和放鬆肌肉來創造「pop」或「hit」的效果，強調肌肉的節奏性動作。[來源 2] Boogaloo 則是一種結合了「The Robot」和模仿動作的舞蹈風格，強調身體各部位的流暢運用，如臀部、膝蓋和頭部的滾動，並且使用幻覺效果、肌肉限制、停頓和機械人動作，融合了 popping 的基礎技巧，形成一種充滿靈魂和激情的街舞形式。[來源 3][來源 5] 此外，Electric Boogaloo 是 Boogaloo 和 popping 的結合，融合了現代 popping 技巧和早期 Boogaloo 風格。[來源 4]

Step 3 — 缺口說明：
資料未涵蓋 Popping 和 Boogaloo 在音樂選擇、文化背景或具體舞步細節上的差異。


---

## Q4: Juste Debout 比賽的規則和歷史？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.474 | 0.474 |
| F | 5 | 0.562 | 0.562 |
| G | 5 | 0.377 | 0.600 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.474（retriever 平均距離: 0.474）
- 引用: `Salah (dancer) - Wikipedia.txt` chunk 66 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 22 (dense), `Salah (dancer) - Wikipedia.txt` chunk 19 (dense)...

Step 1 — 列出 Facts：
- <context> 中未提及 Juste Debout 比賽的規則。
- <context> 中未提及 Juste Debout 比賽的歷史。

Step 2 — 綜合回答：
資料未涵蓋 Juste Debout 比賽的規則和歷史。

Step 3 — 缺口說明：
本次提供的 <context> 中沒有包含任何關於 Juste Debout 比賽的規則或歷史的資訊，因此無法回答此問題。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.562（retriever 平均距離: 0.562）
- 引用: `Juste Debout.txt` chunk 4 (both), `Salah (dancer) - Wikipedia.txt` chunk 66 (dense), `Juste Debout.txt` chunk 0 (bm25)...

Step 1 — 列出 Facts：
1. Juste Debout 是世界上最大的街舞比賽之一，於法國巴黎舉辦，包含 popping 項目，舞者以兩人一組（2 on 2）形式競賽，強調創意和原創性，氣氛熱烈 [來源 5]。
2. Juste Debout 是國際性的 Hip-Hop 舞蹈盛會，2025 年 3 月 2 日在巴黎 Accor Arena 舉行決賽，涵蓋 Locking、Popping、Hip-Hop Newstyle、House Dance、Experimental、Junior Dance Tour PLUS Afro、Dancehall、Krump 和 Electro 等多種舞蹈類別 [來源 1][來源 3]。
3. 比賽從 2024 年 11 月 30 日到 2025 年 3 月 6 日進行全球預選賽，最後決賽在巴黎 Accor Arena 舉行 [來源 1]。
4. 評審制度中提到街舞比賽通常由評審在整場比賽後決定勝負，但有時會因記憶偏差而影響判決，因此有 Kozen 評分系統，評審在每回合後即選出勝者，並有即時計分板顯示分數 [來源 4]。

Step 2 — 綜合回答：
Juste Debout 是一個於法國巴黎舉辦的國際大型街舞比賽，涵蓋多種舞蹈類別，包括 Locking、Popping、Hip-Hop Newstyle、House Dance 等。比賽強調舞者的創意與原創性，特別是在 popping 項目中，舞者以兩人一組的形式競賽。比賽流程包括從 2024 年 11 月 30 日開始的全球預選賽，最終決賽於 2025 年 3 月 2 日在巴黎 Accor Arena 舉行。評審制度可能採用 Kozen 評分系統，即每回合結束後評審即時選出勝者並更新分數，以避免因記憶偏差影響判決，確保比賽公平公正 [來源 1][來源 3][來源 4][來源 5]。

Step 3 — 缺口說明：
資料中未具體說明 Juste Debout 比賽的詳細規則條文及歷史起源背景。資料未涵蓋。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.600（retriever 平均距離: 0.377）
- 擴展查詢: Juste Debout competition rules and history / Juste Debout 比賽的起源與比賽規則介紹 / What are the judging criteria and background of Juste Debout dance battles?
- 引用: `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 12 (dense), `Popping Dance Battles Near You — EVERYDAY POPPING - Online Popping Dance Tutorials.txt` chunk 1 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 18 (dense)...

Step 1 — 列出 Facts：
1. Juste Debout 是世界上最大的街舞比賽之一，在法國巴黎舉辦。[來源 2]
2. Juste Debout 包含 popping 類別，舞者以兩人一組（2 on 2）形式競賽。[來源 2]
3. Juste Debout 強調創意和原創性，舞者在充滿活力的氛圍中展現技巧。[來源 2]

Step 2 — 綜合回答：
Juste Debout 是世界上最大的街舞比賽之一，於法國巴黎舉辦。比賽中包含 popping 類別，舞者以兩人一組（2 on 2）的形式進行競賽。該比賽強調舞者的創意和原創性，並在充滿活力的氛圍中展現他們的舞蹈技巧。[來源 2]

Step 3 — 缺口說明：
資料未涵蓋 Juste Debout 比賽的詳細規則和歷史背景。


---

## Q5: 如何判斷一場 Popping battle 的勝負？

| Combo | chunks | 平均距離（retriever） | 與原問題平均距離 |
|---|---|---|---|
| E | 5 | 0.355 | 0.355 |
| F | 5 | 0.380 | 0.380 |
| G | 5 | 0.302 | 0.382 |

### Combo E — Combo E — CoT + 結構化 context
- 取回 chunks: 5，與原問題平均距離: 0.355（retriever 平均距離: 0.355）
- 引用: `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 0 (dense), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 15 (dense)...

Step 1 — 列出 Facts：
1. Popping battle 是兩位舞者面對面進行，每人有一分鐘的表演時間，沒有積分系統，重點在音樂而非競爭 [來源 1]。
2. 評審會根據多項標準分別評分，尋找符合所有標準的「完整的 Popping 舞者」 [來源 1]。
3. 比賽的目的是學習和提升，結果反映是否符合評審標準 [來源 1][來源 5]。
4. Popping 賽事文化中，評判常關注音樂性、獨創性與完整度 [來源 4]。
5. 比賽設計是為了幫助舞者提升和理解 Popping，強調與自身潛力的競爭而非對手 [來源 5]。

Step 2 — 綜合回答：
Popping battle 的勝負判斷並非依靠積分系統，而是由評審根據多項標準分別評分，重點在於舞者是否展現出完整的 Popping 技巧，包括音樂性、獨創性與整體表現的完整度。評審會尋找符合所有評分標準的「完整的 Popping 舞者」，並以此決定勝負。比賽的核心在於學習與自我提升，強調與自身潛力的競爭，而非直接與對手較量 [來源 1][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋具體評分標準的詳細內容及評審如何在沒有積分系統下具體決定勝負的流程。

### Combo F — Combo F — CoT + Hybrid Search（BM25+Vector+RRF）
- 取回 chunks: 5，與原問題平均距離: 0.380（retriever 平均距離: 0.380）
- 引用: `popping-culture-notes.txt` chunk 0 (both), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 2 (both), `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 9 (both)...

Step 1 — 列出 Facts：
1. Popping battle 中，評判常關注音樂性、獨創性與完整度 [來源 1]。
2. 評判標準包含：FLOOR WORK（地板動作）、STAGE PRESENCE（舞台表現力/娛樂性/與觀眾的連結）、CREATIVITY（創意）、ATTIRE（服裝符合Popping風格）等 [來源 2]。
3. 每場 battle 由兩位舞者對戰，輪流表演一分鐘的回合 [來源 2][來源 4]。
4. 不採用點數制，評判會針對每個標準分別評分，尋找符合所有標準的「完整Popper」 [來源 4]。
5. 評判的結果反映舞者是否符合評判標準，競賽本身是學習工具 [來源 4]。

Step 2 — 綜合回答：
判斷一場 Popping battle 的勝負，評判會根據多項標準來評估舞者的表現，包括音樂性、獨創性、完整度、地板動作（FLOOR WORK）、舞台表現力（STAGE PRESENCE）、創意（CREATIVITY）以及服裝是否符合Popping風格（ATTIRE）[來源 1][來源 2]。每場 battle 由兩位舞者輪流表演一分鐘的回合[來源 2][來源 4]，評判不使用點數制，而是分別針對各項標準進行評分，尋找最能展現所有評判標準的「完整Popper」[來源 4]。最終勝負由評判根據這些標準綜合判定，並反映舞者是否達到評判的要求[來源 4]。

Step 3 — 缺口說明：
資料未涵蓋具體勝負判定的細節程序或評判如何決定勝者（例如是否有投票、多數決等）。

### Combo G — Combo G — CoT + Query Expansion（多角度子查詢）
- 取回 chunks: 5，與原問題平均距離: 0.382（retriever 平均距離: 0.302）
- 擴展查詢: Popping battle 判定勝負的標準有哪些？ / Criteria for winning a Popping battle / How to judge the winner in a Popping dance competition
- 引用: `Dance Battles How Do They Work (Ultimate Guide).txt` chunk 9 (dense), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 3 (dense), `Official Rules & Regulations-Popping  HIP HOP INTERNATIONAL.txt` chunk 0 (dense)...

Step 1 — 列出 Facts：
1. Popping battle 通常是 1 對 1 的對決，每位舞者有 1 分鐘的表演時間，無積分系統，重點在音樂與表現 [來源 2]。
2. 評審會根據多項標準分別評分，尋找符合所有標準的「完整舞者」 [來源 2]。
3. 在淘汰賽中，主持人倒數後，評審會用手臂指向他們認為表現較佳的舞者，得票多者勝出 [來源 4]。
4. 若評審無法決定勝負，會用交叉手臂表示平手，進入加賽決勝 [來源 4]。
5. 評審關注的標準包括音樂性、獨創性與完整度 [來源 5]。

Step 2 — 綜合回答：
判斷一場 Popping battle 的勝負，首先是兩位舞者各有 1 分鐘的表演時間，評審不採用積分制，而是根據多項標準分別評分，尋找最符合所有標準的完整舞者。比賽中，主持人倒數後，評審會用手臂指向他們認為表現較佳的舞者，得票多者即為勝者。如果評審無法決定勝負，會以交叉手臂表示平手，進入加賽決勝。評審主要關注的標準包括音樂性、獨創性與完整度 [來源 2][來源 4][來源 5]。

Step 3 — 缺口說明：
資料未涵蓋具體評審標準的詳細項目及評分細節。
