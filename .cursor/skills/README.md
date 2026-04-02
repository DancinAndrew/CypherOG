# `.cursor/skills/` 說明

此目錄為 **Everything Claude Code (ECC)** 與專案額外套件中的 **Skill**：每個子資料夾內有 `SKILL.md`，內容是**可重複使用的流程／領域手冊**。

在 Cursor 中 **不會**自動把每個 SKILL 全文載入；需要時請 **`@` 該 `SKILL.md`** 或在提示中說「照某某 skill」。

---

## 本專案（CypherOG / HW3）優先會用到的

| 資料夾 | 用途（精簡） |
|--------|----------------|
| **python-patterns** | Python 慣例、PEP 8、型別、可維護性 |
| **python-testing** | pytest、fixtures、mock、覆蓋率、TDD |
| **api-design** | 若作業或延伸做出 REST API：命名、狀態碼、分頁、錯誤格式 |
| **backend-patterns** | 後端模組邊界、快取、錯誤處理（Node 範例多，概念可類推到 Python） |
| **tdd-workflow** | 紅綠重構、單元／整合／E2E 與覆蓋率思維 |
| **verification-loop** | 建置、測試、lint、一連串驗證迴圈 |

---

## 完整列表（依資料夾名）

| Skill | 在做什麼 |
|-------|-----------|
| **ai-regression-testing** | AI 寫碼場景的回歸測試：沙盒 API、抓「同一個模型又寫又審」的盲點 |
| **api-design** | REST 設計：資源命名、狀態碼、分頁、錯誤、版本、限流 |
| **backend-patterns** | 後端架構、API、DB、快取（Node/Express/Next API 為主） |
| **coding-standards** | TS/JS/React/Node 通用寫作與慣例 |
| **configure-ecc** | 互動式安裝／調整 ECC（選 skill、路徑、簡單驗證） |
| **continuous-learning** | 從 session 抽可重用模式並存成「學到的技能」 |
| **continuous-learning-v2** | Hook 觀察 session → instinct（信心分）→ 演化進 skill/command |
| **django-patterns** | Django／DRF、ORM、快取、middleware 等 |
| **django-tdd** | pytest-django、factory、DRF API 測試、TDD |
| **django-verification** | Django 發版／PR 前：migration、lint、測試、安全掃描 |
| **e2e-testing** | Playwright、POM、CI、artifact、flaky 處理 |
| **eval-harness** | Eval-driven development：對話／任務的正式評測框架 |
| **frontend-patterns** | React、Next、狀態、效能、UI 慣例 |
| **frontend-slides** | HTML 簡報、PPTX 轉網頁、動畫與版型探索 |
| **iterative-retrieval** | 子代理／多輪時逐步縮小檢索、補齊 context |
| **mcp-server-patterns** | 用 Node/TS 寫 MCP server（tools、Zod、stdio／HTTP） |
| **perl-patterns** | 現代 Perl 5.36+ 慣例 |
| **perl-testing** | Test2、prove、mock、Devel::Cover、TDD |
| **plankton-code-quality** | 搭配 Plankton：每次編輯後 format／lint／AI 修（偏 hook） |
| **project-guidelines-example** | 專案專用 skill 的範本（真實產品風格） |
| **python-patterns** | Pythonic、PEP 8、型別、最佳實踐 |
| **python-testing** | pytest、TDD、fixtures、mock、覆蓋率 |
| **skill-stocktake** | 盤點 skills/commands 品質（快掃／全掃） |
| **strategic-compact** | 建議在任務段落手動 compact，而非只靠自動壓縮 |
| **tdd-workflow** | 新功能／修 bug／重構時強制 TDD 與高覆蓋率思維 |
| **ui-ux-pro-max** | UI/UX 設計指南與可搜尋素材庫（風格、色票、字型、UX 準則等）；做前端介面美化時可 `@` |
| **verification-loop** | 完整驗證管線（建置、測試、靜態檢查等） |

---

## 與 `agents/` 的差異（簡短）

- **skills**：偏「流程與領域怎麼做」（手冊、步驟長）。
- **agents**（見 `../agents/README.md`）：偏「扮演哪種角色、審什麼」（角色卡較短）。

兩者在 Cursor 裡都需 **`@` 或明示** 才會整份進 context；常駐行為請靠根目錄 `AGENTS.md` 與 `.cursor/rules/`。
