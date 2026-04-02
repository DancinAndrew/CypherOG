# Cursor 自訂指令（`/檔名`）

在聊天輸入 **`/`** 會出現專案指令；下列對應本目錄 `*.md`（**不含副檔名**）。

**相關索引：**

- 領域流程手冊 → `../skills/README.md`
- 角色／審查 → `../agents/README.md`
- 專案總說明 → _repo 根目錄_ `AGENTS.md`

---

## 規劃與實作

| 指令 | 說明 |
|------|------|
| `/plan` | 釐清需求與步驟；先確認再改碼 |
| `/code-review` | 審查本地變更或指定 PR |
| `/build-fix` | 偵測建置並逐步修錯 |
| `/python-review` | Python：PEP8、型別、安全、慣例 |
| `/refactor-clean` | 用靜態工具找並移除 dead code |

---

## 測試與品質

| 指令 | 說明 |
|------|------|
| `/tdd` | TDD 流程（對應 tdd-workflow skill） |
| `/e2e` | E2E／Playwright（對應 e2e-testing skill） |
| `/test-coverage` | 覆蓋率分析並補測 |
| `/verify` | 驗證迴圈（對應 verification-loop skill） |
| `/eval` | 評測框架（對應 eval-harness skill） |
| `/quality-gate` | 跑 ECC 品質管線（可 `--fix`、`--strict`） |
| `/checkpoint` | 建立／驗證／列出 checkpoint |

---

## 文件與架構圖

| 指令 | 說明 |
|------|------|
| `/update-docs` | 依專案真相來源同步文件 |
| `/update-codemaps` | 產生／更新 CODEMAPS 架構說明 |

---

## 學習／Instinct／Skill（偏 ECC 生態）

| 指令 | 說明 |
|------|------|
| `/learn` | 從當次對話抽可重用模式 |
| `/learn-eval` | 抽模式前先自評再決定存哪 |
| `/instinct-status` | 查看已學 instinct 與信心分 |
| `/instinct-import` | 匯入 instinct |
| `/instinct-export` | 匯出 instinct |
| `/promote` | 專案級 instinct 升全域 |
| `/prune` | 刪 30 天未升級的 pending instinct |
| `/projects` | 專案與 instinct 統計 |
| `/evolve` | 分析並演化 instinct 結構 |
| `/skill-create` | 從 git 歷史產 SKILL.md |
| `/skill-health` | Skill 組合健康儀表板 |

---

## Session（Claude Code 路徑）

| 指令 | 說明 |
|------|------|
| `/sessions` | Session 歷史與別名 |
| `/save-session` | 存到 `~/.claude/session-data/` |
| `/resume-session` | 載入最近一次 session |

---

## 多模型／PRP／部署

| 指令 | 說明 |
|------|------|
| `/multi-plan` | 多模型協作只寫計畫（`.claude/plan/`） |
| `/multi-execute` | 依計畫多模型執行 |
| `/multi-backend` | 後端為主流程 |
| `/multi-frontend` | 前端為主流程 |
| `/multi-workflow` | 完整多模型 workflow |
| `/prp-prd` | 互動式 PRD |
| `/prp-plan` | 功能實作計畫 |
| `/prp-implement` | 依計畫實作與驗證 |
| `/prp-commit` | 自然語言指定要 commit 的檔 |
| `/prp-pr` | 從分支開 GitHub PR |
| `/pm2` | 掃專案並產 PM2 指令／設定 |

---

## Harness／迴圈／模型

| 指令 | 說明 |
|------|------|
| `/harness-audit` | Repo harness 稽核打分 |
| `/loop-start` | 啟動受控自動迴圈 |
| `/loop-status` | 查看迴圈狀態 |
| `/model-route` | 依任務建議模型層級 |
| `/orchestrate` | 編排（舊入口，優先改用 skills） |
| `/santa-loop` | 雙 reviewer 都過才放行 |

---

## 設計實驗（GAN）

| 指令 | 說明 |
|------|------|
| `/gan-design` | 前端設計迭代（Generator + Evaluator） |
| `/gan-build` | GAN 式實作迴圈（可接 Playwright 等） |

---

## 其他 shim（優先改用對應 skill）

| 指令 | 說明 |
|------|------|
| `/prompt-optimize` | prompt 優化 |
| `/rules-distill` | 規則精煉 |
| `/context-budget` | 上下文預算 |
| `/docs` | 文件查詢 |
| `/devfleet` | devfleet |
| `/claw` | nanoclaw REPL |
| `/aside` | 旁支問題不打斷主線 |

---

## 工具

| 指令 | 說明 |
|------|------|
| `/setup-pm` | 設定 npm/pnpm/yarn/bun |

---

**提示**：多數 Legacy shim 在檔案開頭會寫「請改用 `skills/...`」；需要長流程時也可直接 `@ ../skills/某資料夾/SKILL.md`。
