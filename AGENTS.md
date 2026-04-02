# CypherOG — Agent 與協作說明

本專案在根目錄沿用 **Everything Claude Code (ECC)** 的工作流；實際打包內容以 **`.cursor/`** 為準（**25** 個 agent、**約 27** 個 skill、**約 55** 個 slash command）。

**索引（請優先閱讀）：**

| 路徑 | 內容 |
|------|------|
| `.cursor/commands/README.md` | 全部 `/` 指令對照表 |
| `.cursor/skills/README.md` | 各 skill 在做什麼、HW3 優先建議 |
| `.cursor/agents/README.md` | 各 agent 角色與使用時機 |

**ECC 參考版本：** 1.9.0

## 本專案重點（HW3）

- **主題**：課程作業「個人知識 RAG」— 資料收集 → chunk → embedding → 向量庫 → `rag_query` / LiteLLM → `skill_builder` → `skill.md`。
- **技術棧**：以 **Python（≥3.10）** 為主；詳細必繳與評分請對齊 `HW3_Describtion.MD` 與 `README.md`。
- **優先委派**：與 Python、資料管線、測試、文件、資安相關時，優先依下表與 `.cursor/rules/`（尤其 `python-*.md`）。

## Core Principles

1. **Agent-First** — 複雜或跨領域工作交給對應 agent 定義（`.cursor/agents/*.md`）
2. **Test-Driven** — 新功能／修 bug 以測試優先；覆蓋率目標依作業與 `README` 聲明（ECC 預設常見為 80%+）
3. **Security-First** — API Key 僅能出現在 `.env`，勿提交金鑰；輸入與檔案路徑需驗證
4. **Immutability** — 偏好不可變與純函數式更新，避免隱性共享狀態
5. **Plan Before Execute** — 大改動先規劃再實作

## Available Agents（25，檔案於 `.cursor/agents/`）

| Agent | Purpose | When to Use |
|-------|---------|-------------|
| planner | 功能拆解、重構規劃 | 複雜需求、多階段實作 |
| architect | 系統設計、擴展性 | 架構決策、技術選型 |
| tdd-guide | TDD、覆蓋率 | 新功能、修 bug、重構 |
| code-reviewer | 品質、可維護性 | 寫完或改完程式碼後 |
| security-reviewer | 漏洞與設定風險 | 認證、輸入、API、敏感資料 |
| build-error-resolver | 建置／型別錯誤 | 編譯或型別失敗時 |
| e2e-runner | E2E（Playwright 等） | 有需要 UI／瀏覽器流程時 |
| refactor-cleaner | 死碼、重複、精簡 | 維護、瘦身 |
| doc-updater | 文件、codemap | README、架構圖更新 |
| docs-lookup | 官方文件／API 範例 | 查庫用法（常搭配 Context7 MCP） |
| database-reviewer | PostgreSQL／遷移／查詢 | 向量庫以外的 schema、SQL |
| python-reviewer | Python 風格、型別、安全 | **本專案主語言** |
| typescript-reviewer | TS/JS 審查 | 若專案內有前端／Node 腳本 |
| pytorch-build-resolver | PyTorch 訓練／推理錯誤 | 僅在用到 PyTorch 時 |
| performance-optimizer | 效能、記憶體、瓶頸 | 明顯慢或資源問題 |
| loop-operator | 自主迴圈監控與介入 | 長程自動化任務 |
| harness-optimizer | Agent 設定、成本、吞吐 | 調整 harness／規則時 |
| gan-planner | GAN：一行需求展開成規格 | 實驗性產品／設計迭代 |
| gan-generator | GAN：依規格實作並迭代 | 搭配 gan-evaluator |
| gan-evaluator | GAN：Playwright 等評分回饋 | 搭配 gan-generator |
| opensource-forker | 開源 fork 去機敏、模板化 | 開源流程第一階段 |
| opensource-packager | 開源 README、LICENSE、模板 | 開源流程第三階段 |
| opensource-sanitizer | 開源前掃描外洩與風險 | 公開 release 前 |
| healthcare-reviewer | 醫療資訊系統合規 | 僅醫療相關程式碼 |
| chief-of-staff | 多渠道訊息分類與草稿 | 個人通訊工作流（非一般寫碼） |

## Agent Orchestration

可主動委派（無需使用者逐句指定）：

- 複雜功能 → **planner**
- 剛改完程式 → **code-reviewer**
- 新功能或修 bug → **tdd-guide**
- 架構決策 → **architect**
- 敏感路徑 → **security-reviewer**
- 長程 loop → **loop-operator**
- Harness 調参 → **harness-optimizer**

獨立子任務可並行（多個 agent 讀不同檔案時）。

## Security Guidelines

**Commit 前：**

- 禁止硬編碼金鑰；僅用環境變數或秘密管理
- 使用者與檔案輸入需驗證；資料庫參數化查詢
- 錯誤訊息不洩漏內部路徑或金鑰

**若發現外洩：** 停手 → **security-reviewer** 思路修復 → 旋轉金鑰 → 搜尋同類問題。

## Coding Style

**Immutability：** 優先回傳新物件，避免就地修改共享結構。

**檔案：** 依功能／模組切分，單檔不宜過大；與 `HW3` 必繳結構一致。

**錯誤：** 邊界捕捉、記錄足夠上下文；不要靜默吞掉例外。

## Testing

- 以 **pytest** 與作業要求為主；模組與整合測試優先。
- **E2E**：僅在確實有瀏覽器／端到端需求時再動用 **e2e-runner**（本 HW 核心多為 CLI／管線）。

**TDD 迴圈（通用）：** 先失敗測試 → 最小實作通過 → 重構。

## Development Workflow

1. **Plan** — planner：依賴與風險、分階段
2. **TDD** — tdd-guide：測試先行
3. **Review** — code-reviewer / python-reviewer：處理 HIGH／CRITICAL
4. **知識存放** — 架構與操作以 `README.md`、程式註解為主；避免重複開頂層說明檔，除非使用者要求

## Workflow Surface

- **`/.cursor/skills/`**：流程與領域手冊；詳見 **`/.cursor/skills/README.md`**。需要長流程時 `@` 對應 `SKILL.md`。
- **`/.cursor/agents/`**：角色／審查清單；詳見 **`/.cursor/agents/README.md`**。用 `@` 或請模型依該角色執行。
- **`/.cursor/commands/`**：斜線指令；詳見 **`/.cursor/commands/README.md`**。與 skill 重複時以 skill 為準。

## Git

**Commit：** `<type>: <description>`（feat, fix, refactor, docs, test, chore, perf, ci）

**PR：** 變更要旨、測試方式、如何重現。

## Architecture（依專案類型取捨）

- 若有 **REST／HTTP**：一致錯誤格式、分頁、版本化可參考通用後端模式。
- **本 repo 以 CLI、資料管線、RAG、LiteLLM 為主**時：以模組邊界、可重跑腳本、`data_update` 冪等等為架構核心，不必硬套網頁 API 信封。

## Performance

- 長上下文任務避免在視窗末端做超大重構。
- 建置失敗 → **build-error-resolver**，小步修正並重跑。

## Project Structure（本專案與 ECC）

```
（作業必繳） data/, data_update.py, rag_query.py, skill_builder.py, skill.md, README.md, requirements.txt, .env.example
.cursor/
  agents/README.md — agent 總覽
  agents/*.md      — 25 個 agent 定義
  skills/README.md — skill 總覽
  skills/*/SKILL.md— 領域與流程手冊
  commands/README.md — `/` 指令總覽
  commands/*.md  — 斜線指令本體（約 55 個）
  rules/         — 專案規則（common + python 等）
  hooks/         — 事件掛鉤（需在 Cursor 啟用）
  mcp-configs/   — MCP 範本（需自行合併與填 key）
```

## Success Metrics

- 作業階段要求與 CI 通過（見 `HW3_Describtion.MD`）
- 無金鑰外洩；測試與文件可復現
- 程式可讀、模組責任清楚
