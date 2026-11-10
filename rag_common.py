"""共用設定：路徑、embedding、Chroma 集合、BM25 索引（供 data_update / rag_query / skill_builder 使用）。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

# 只載入本 repo 的 .env：不帶路徑的 load_dotenv() 會往上層目錄搜尋，可能誤用其他專案的 proxy 設定
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MANIFEST_PATH = ROOT / "data" / ".ingest_manifest.json"
# data/processed 的鎖定指紋：round-1 還原與 round-2 實驗都必須使用這份已提交的文字
CORPUS_PIN_PATH = ROOT / "experiments" / "corpus_pin.json"
# round-1 索引實際收錄的 39 份 data/raw 文件（experiment_runner --corpus round1 使用）
CORPUS_ROUND1_PATH = ROOT / "experiments" / "corpus_round1.json"

# 找不到或無法使用索引時的建議指令（不建議會改寫已提交 data/processed 的指令）
REBUILD_HINT = (
    "預設 collection 請執行 python data_update.py --rebuild --reuse-processed；"
    "實驗 collection 請重跑 experiment_runner.py（不加 --skip-rebuild）"
)


def _resolve_project_dir(value: str) -> str:
    """相對路徑一律相對於專案根目錄解析（與 .env.example 的說明一致，不受執行時 cwd 影響）。"""
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else ROOT / p)


CHROMA_PERSIST_DIR = _resolve_project_dir(
    os.environ.get("CHROMA_PERSIST_DIR", str(ROOT / "chroma_db"))
)
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "cypher_og")

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)

# 未設定 LITELLM_MODEL 時的預設模型（第一輪實驗經 OpenAI-compatible proxy 呼叫）
DEFAULT_LLM_MODEL = "gemini-2.5-flash"

# chunk 預設（可被 CLI 覆寫時請改 data_update 內常數）
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

_st_model = None

# Chroma collection 名稱只允許英數與 . _ -，且首尾為英數（同時避免索引檔路徑跳脫）
_COLLECTION_NAME_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_BM25_TOKEN_RE = re.compile(r"[a-z0-9']+|[一-鿿]")


def _warn(message: str) -> None:
    print(message, file=sys.stderr)


def display_path(path: str | os.PathLike[str]) -> str:
    """輸出用路徑：repo 內顯示相對路徑，repo 外只顯示最後兩層（不洩漏本機絕對路徑）。"""
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return "/".join(resolved.parts[-2:])


def describe_exception(exc: BaseException) -> str:
    """例外 → 「型別: 說明」，不含本機絕對路徑。

    OSError 只顯示 strerror 與 display_path(filename)；其他例外訊息中的 repo 路徑改為相對路徑、
    家目錄改為 ~。LLM 供應商的例外不應傳到這裡（呼叫端只記錄型別）。
    """
    name = type(exc).__name__
    if isinstance(exc, OSError):
        target = f" {display_path(exc.filename)}" if exc.filename else ""
        return f"{name}: {exc.strerror or 'OS error'}{target}"
    message = (
        str(exc)
        .replace(str(ROOT) + os.sep, "")
        .replace(str(ROOT), ".")
        .replace(str(Path.home()), "~")
    )
    return f"{name}: {message}" if message else name


# ---------------------------------------------------------------------------
# 語料指紋：data/processed/*.txt 的 sha256，與 experiments/corpus_pin.json 比對
# ---------------------------------------------------------------------------


def text_sha256(text: str) -> str:
    """文字（UTF-8）的 sha256；data/processed 檔案不含 \\r，與檔案位元組的 sha256 相同。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def processed_corpus_hashes(processed_dir: Path | None = None) -> dict[str, str]:
    """{檔名: sha256(檔案位元組)}，涵蓋 processed_dir（預設 data/processed）下所有 *.txt，依檔名排序。"""
    base = processed_dir if processed_dir is not None else DATA_PROCESSED
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(base.glob("*.txt"))
        if p.is_file()
    }


def corpus_digest(hashes: Mapping[str, str]) -> str:
    """整份語料的單一指紋：sha256 of 依檔名排序的「檔名\\tsha256\\n」。"""
    lines = "".join(f"{name}\t{hashes[name]}\n" for name in sorted(hashes))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def load_corpus_pin(path: Path | None = None) -> dict[str, str]:
    """讀取 corpus_pin.json 的 {檔名: sha256}；結構不符或 digest 對不上時拋 ValueError。"""
    data = json.loads((path if path is not None else CORPUS_PIN_PATH).read_text(encoding="utf-8"))
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, dict) or not files:
        raise ValueError("corpus_pin.json 缺少 files")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items()):
        raise ValueError("corpus_pin.json 的 files 必須是 {檔名: sha256}")
    if data.get("digest") != corpus_digest(files):
        raise ValueError("corpus_pin.json 的 digest 與 files 不符")
    return dict(files)


def corpus_mismatches(actual: Mapping[str, str], pinned: Mapping[str, str]) -> list[str]:
    """回傳缺少、多出或內容不同的檔名（排序後）；空 list 代表與鎖定版本完全一致。"""
    return sorted(name for name in set(actual) | set(pinned) if actual.get(name) != pinned.get(name))


# ---------------------------------------------------------------------------
# 語料選擇：以 data/raw 檔名列出要建索引的文件（例如 experiments/corpus_round1.json）
# ---------------------------------------------------------------------------

_CORPUS_NAME_RE = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


def processed_name_for(raw_name: str) -> str:
    """data/raw 檔名 → data/processed 檔名（<stem>.txt，與 data_update 的規則相同）。"""
    return f"{Path(raw_name).stem}.txt"


def document_list_digest(names: Iterable[str]) -> str:
    """文件清單的指紋：sha256 of 依名稱排序的「名稱\\n」（只看檔名，不看內容；重複名稱只算一次）。"""
    lines = "".join(f"{name}\n" for name in sorted(set(names)))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _file_name_list(value: Any, field: str) -> list[str]:
    """驗證「不重複的單層檔名 list」，回傳排序後的新 list；不符時拋 ValueError。"""
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{field} 必須是非空字串的 list")
    if len(set(value)) != len(value):
        raise ValueError(f"{field} 有重複的檔名")
    if any("/" in v or "\\" in v or v in {".", ".."} for v in value):
        raise ValueError(f"{field} 只能列 data/raw 內的檔名，不可含路徑")
    return sorted(value)


def load_corpus_definition(path: Path | None = None) -> dict[str, Any]:
    """讀取語料定義檔（預設 experiments/corpus_round1.json），回傳 {name, documents, excluded}（皆為新物件）。

    documents：要建索引的 data/raw 檔名（排序、不可為空）；excluded：刻意排除的 data/raw 檔名。
    name 會成為 Chroma collection 名稱的一部分，只允許小寫英數與底線。結構不符時拋 ValueError。
    """
    data = json.loads((path if path is not None else CORPUS_ROUND1_PATH).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("語料定義檔不是 JSON 物件")
    name = data.get("name")
    if not isinstance(name, str) or not _CORPUS_NAME_RE.fullmatch(name):
        raise ValueError(f"語料名稱不合法: {name!r}")
    documents = _file_name_list(data.get("documents"), "documents")
    if not documents:
        raise ValueError("documents 不可為空")
    excluded = _file_name_list(data.get("excluded", []), "excluded")
    overlap = sorted(set(documents) & set(excluded))
    if overlap:
        raise ValueError(f"documents 與 excluded 重疊: {overlap[:3]}")
    return {"name": name, "documents": documents, "excluded": excluded}


def pinned_digest_for(documents: Iterable[str], pinned: Mapping[str, str]) -> str | None:
    """所選 data/raw 文件在 corpus_pin.json 中的文字指紋（與 rebuild 回傳值的 corpus_digest 可比）；
    任一文件未被鎖定時回傳 None。"""
    names = {processed_name_for(d) for d in documents}
    if not names or not names <= set(pinned):
        return None
    return corpus_digest({n: pinned[n] for n in names})


def resolve_collection_name(collection_name: str | None = None) -> str:
    """回傳實際使用的 collection 名稱（呼叫時才讀取模組層 CHROMA_COLLECTION）。"""
    name = collection_name or CHROMA_COLLECTION
    if not _COLLECTION_NAME_RE.fullmatch(name) or ".." in name:
        raise ValueError(f"不合法的 collection 名稱: {name!r}")
    return name


# ---------------------------------------------------------------------------
# BM25 Index — 混合搜尋（Hybrid Search）支援
# 對專有名詞（人名、組織名）的精確匹配優於純向量搜尋。
# 每個 Chroma collection 各有一份 JSON 索引：<CHROMA_PERSIST_DIR>/bm25_<collection>.json
# ---------------------------------------------------------------------------


def tokenize_for_bm25(text: str) -> list[str]:
    """BM25 共用分詞：轉小寫；英文/數字按詞、中文按單字；去除標點。永不回傳空 list。"""
    tokens = _BM25_TOKEN_RE.findall(text.lower())
    return tokens if tokens else [""]


def bm25_index_path(collection_name: str | None = None) -> Path:
    """指定 collection 的 BM25 索引路徑（呼叫時才讀取 CHROMA_PERSIST_DIR / CHROMA_COLLECTION）。"""
    return Path(CHROMA_PERSIST_DIR) / f"bm25_{resolve_collection_name(collection_name)}.json"


def save_bm25_index(
    docs: list[str], ids: list[str], collection_name: str | None = None
) -> Path:
    """將 collection 的全部 chunk 寫成 BM25 索引（UTF-8 JSON，原子替換），回傳檔案路徑。"""
    if len(docs) != len(ids):
        raise ValueError(f"docs 與 ids 數量不一致: {len(docs)} != {len(ids)}")
    name = resolve_collection_name(collection_name)
    path = bm25_index_path(name)
    payload = {
        "collection": name,
        "ids": list(ids),
        "docs": list(docs),
        "tokenized": [tokenize_for_bm25(d) for d in docs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _parse_bm25_payload(data: Any) -> tuple[list[str], list[str], list[list[str]]]:
    """驗證索引 JSON 結構，回傳 (ids, docs, tokenized)；結構不符時拋 ValueError。"""
    if not isinstance(data, dict):
        raise ValueError("索引內容不是 JSON 物件")
    ids, docs, tokenized = data.get("ids"), data.get("docs"), data.get("tokenized")
    if not (isinstance(ids, list) and isinstance(docs, list) and isinstance(tokenized, list)):
        raise ValueError("索引缺少 ids / docs / tokenized")
    if not (len(ids) == len(docs) == len(tokenized)):
        raise ValueError("ids / docs / tokenized 長度不一致")
    if not all(isinstance(i, str) for i in ids) or not all(isinstance(d, str) for d in docs):
        raise ValueError("ids / docs 必須是字串")
    if not all(isinstance(t, list) and all(isinstance(w, str) for w in t) for t in tokenized):
        raise ValueError("tokenized 必須是字串 list 的 list")
    return list(ids), list(docs), [list(t) for t in tokenized]


def load_bm25_index(collection_name: str | None = None):
    """載入指定 collection 的 BM25 索引，回傳 (BM25Okapi 或 None, ids, docs)。

    - 檔案不存在、或語料為空：靜默回傳 (None, [], [])。
    - 檔案損毀/結構不符、或未安裝 rank_bm25：在 stderr 警告（含 collection 與例外類型）後回傳 (None, [], [])。
    """
    name = resolve_collection_name(collection_name)
    path = bm25_index_path(name)
    if not path.is_file():
        return None, [], []
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        _warn(f"[rag_common] 警告：無法載入 collection {name!r} 的 BM25 索引（{type(exc).__name__}：未安裝 rank_bm25）。")
        return None, [], []
    try:
        ids, docs, tokenized = _parse_bm25_payload(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:  # JSONDecodeError / UnicodeDecodeError 皆為 ValueError 子類
        _warn(f"[rag_common] 警告：collection {name!r} 的 BM25 索引無法讀取（{type(exc).__name__}）；{REBUILD_HINT}。")
        return None, [], []
    if not ids:
        return None, [], []
    return BM25Okapi(tokenized), ids, docs


def refresh_bm25_index(collection_name: str | None = None) -> Path:
    """依 Chroma collection 目前的全部 documents 重建 BM25 索引（增量更新後呼叫，避免索引過期）。"""
    coll = get_chroma_collection(collection_name)
    got = coll.get(include=["documents"])
    ids = list(got.get("ids") or [])
    raw_docs = got.get("documents")
    docs = [d or "" for d in (raw_docs if raw_docs is not None else [])]
    return save_bm25_index(docs, ids, collection_name)


# ---------------------------------------------------------------------------
# Embedding / Chroma
# ---------------------------------------------------------------------------


def get_sentence_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer(EMBEDDING_MODEL)
    return _st_model


def loaded_embedding_max_seq_length() -> int | None:
    """若 embedding 模型已載入，回傳其 max_seq_length（token 上限）；未載入時不觸發載入，回傳 None。"""
    if _st_model is None:
        return None
    value = getattr(_st_model, "max_seq_length", None)
    return int(value) if value is not None else None


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_sentence_model()
    arr = model.encode(texts, convert_to_numpy=True)
    return arr.tolist()


def get_chroma_collection(collection_name: str | None = None):
    import chromadb
    from chromadb.config import Settings

    name = collection_name or CHROMA_COLLECTION
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# LiteLLM
# ---------------------------------------------------------------------------


def effective_llm_model(model: str | None = None) -> str:
    """實際使用的 LLM 模型名稱：明確指定 > 環境變數 LITELLM_MODEL > DEFAULT_LLM_MODEL。"""
    return model or os.environ.get("LITELLM_MODEL") or DEFAULT_LLM_MODEL


def llm_proxy_configured() -> bool:
    """是否同時設定了 LITELLM_BASE_URL 與 LITELLM_API_KEY（走 OpenAI-compatible proxy）。不回傳任何值本身。"""
    base = (os.environ.get("LITELLM_BASE_URL") or "").strip()
    key = (os.environ.get("LITELLM_API_KEY") or "").strip()
    return bool(base and key)


def _proxy_openai_model_name(model: str) -> str:
    """OpenAI-compatible proxy：須用 openai/ 前綴，否則 gemini-* 會被導向 Vertex。"""
    if "/" in model:
        return model
    return f"openai/{model}"


def litellm_chat_completion(
    *,
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.2,
    **kwargs,
):
    """若設定了 LITELLM_BASE_URL + LITELLM_API_KEY，走 OpenAI-compatible 代理；否則沿用 LiteLLM 預設路由。"""
    from litellm import completion

    m = effective_llm_model(model)
    if llm_proxy_configured():
        return completion(
            model=_proxy_openai_model_name(m),
            api_base=os.environ["LITELLM_BASE_URL"].strip(),
            api_key=os.environ["LITELLM_API_KEY"].strip(),
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
    return completion(model=m, messages=messages, temperature=temperature, **kwargs)
