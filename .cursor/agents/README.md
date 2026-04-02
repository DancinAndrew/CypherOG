# `.cursor/agents/` 說明

此目錄為 **ECC 子代理定義**：每個 `*.md` 是一份 **角色／審查清單**（內含 YAML `name` / `description`）。

在 **Cursor** 裡**沒有**像 Claude Code 那樣「真的開子進程」的按鈕；實際用法是請模型 **依某 agent 的準則行事**，或 **`@` 該檔**。不會 25 個自動全載入。

---

## 本專案（CypherOG / Python RAG）優先

| 檔案 | 時機 |
|------|------|
| **planner.md** | 功能大、步驟多、要拆階段 |
| **python-reviewer.md** | 改動 Python 後做審查 |
| **tdd-guide.md** | 新功能／修 bug，要測試先行 |
| **code-reviewer.md** | 任何語言改完後通用審查 |
| **build-error-resolver.md** | 建置或型別錯誤要快速修綠 |
| **security-reviewer.md** | 輸入、金鑰、注入、敏感路徑 |
| **doc-updater.md** | 同步 README、codemap、文件結構 |

---

## 完整列表（25 個）

| 檔案 | 角色／用途（精簡） |
|------|-------------------|
| **architect.md** | 系統設計、擴展性、重大技術決策 |
| **build-error-resolver.md** | 建置／TS 型別錯誤，最小 diff 修到過 |
| **chief-of-staff.md** | 多渠道訊息分類、草稿回覆（通訊工作流，非一般寫碼） |
| **code-reviewer.md** | 品質、安全、可維護性；改碼後應審 |
| **database-reviewer.md** | PostgreSQL、遷移、查詢效能、schema（含 Supabase 慣例） |
| **doc-updater.md** | 更新文件、codemap、`docs/CODEMAPS` 等 |
| **docs-lookup.md** | 查官方文件／API 範例（常搭配 Context7 MCP） |
| **e2e-runner.md** | E2E：Agent Browser／Playwright、維護測試旅程 |
| **gan-evaluator.md** | GAN harness：對跑起来的應用評分、給生成端回饋 |
| **gan-generator.md** | GAN harness：依規格實作並迭代到門檻 |
| **gan-planner.md** | GAN harness：一行需求展開成完整規格與評測方向 |
| **harness-optimizer.md** | 調整本機 harness（可靠度、成本、吞吐） |
| **healthcare-reviewer.md** | 醫療資訊系統：臨床安全、PHI、CDSS（僅相關專案） |
| **loop-operator.md** | 監控自主迴圈、卡關時安全介入 |
| **opensource-forker.md** | 開源化第一步：去機敏、`.env.example`、可選清 git 歷史 |
| **opensource-packager.md** | 開源第三步：LICENSE、CONTRIBUTING、issue 模板等 |
| **opensource-sanitizer.md** | 開源第二步：掃外洩、PII、內部字串，產報告 |
| **performance-optimizer.md** | 效能瓶頸、記憶體、bundle、演算法 |
| **planner.md** | 複雜功能與重構的實作計畫 |
| **python-reviewer.md** | Python：PEP 8、型別、安全、慣例（Python 專案必用） |
| **pytorch-build-resolver.md** | PyTorch 訓練／推理錯誤（張量、device、DataLoader 等） |
| **refactor-cleaner.md** | 死碼、重複、knip/depcheck 等工具輔助移除 |
| **security-reviewer.md** | OWASP、秘密、SSRF、注入、加密使用 |
| **tdd-guide.md** | 強制先測試、覆蓋率思維 |
| **typescript-reviewer.md** | TS/JS 審查：型別、async、Node／web 安全 |

---

## 與 `skills/` 的差異（簡短）

| | **agents** | **skills** |
|---|------------|------------|
| 內容 | 我是誰、要檢查什麼 | 流程怎麼跑、領域知識步驟 |
| 典型長度 | 較短 | 常較長（完整 workflow） |

詳見 `../skills/README.md`。
