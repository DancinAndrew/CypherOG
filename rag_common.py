"""共用設定：路徑、embedding、Chroma 集合、BM25 索引（供 data_update / rag_query / skill_builder 使用）。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MANIFEST_PATH = ROOT / "data" / ".ingest_manifest.json"

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", str(ROOT / "chroma_db"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "hw3_rag")

# BM25 索引的持久化路徑（pickle 格式）
BM25_INDEX_PATH = ROOT / "chroma_db" / "bm25_index.pkl"

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)

# chunk 預設（可被 CLI 覆寫時請改 data_update 內常數）
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

_st_model = None


# ---------------------------------------------------------------------------
# BM25 Index — 混合搜尋（Hybrid Search）支援
# 對專有名詞（人名、組織名）的精確匹配優於純向量搜尋
# ---------------------------------------------------------------------------

def save_bm25_index(docs: list[str], ids: list[str]) -> None:
    """儲存 BM25 索引（pickle）。docs 和 ids 對應 Chroma 中的所有 chunk。"""
    try:
        import pickle
        import re

        def _tokenize(text: str) -> list[str]:
            """簡易分詞：英文按空白切，中文按字切，去除標點。"""
            text = text.lower()
            # 保留英文單字、數字、中文字
            tokens = re.findall(r"[a-z0-9']+|[\u4e00-\u9fff]", text)
            return tokens if tokens else [""]

        tokenized = [_tokenize(d) for d in docs]
        BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump({"tokenized": tokenized, "ids": ids, "docs": docs}, f)
    except ImportError:
        pass  # rank_bm25 未安裝時靜默跳過


def load_bm25_index():
    """載入 BM25 索引，回傳 (bm25_obj, ids, docs) 或 (None, [], [])。"""
    try:
        import pickle
        from rank_bm25 import BM25Okapi

        if not BM25_INDEX_PATH.is_file():
            return None, [], []
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        bm25 = BM25Okapi(data["tokenized"])
        return bm25, data["ids"], data["docs"]
    except (ImportError, Exception):
        return None, [], []


def get_sentence_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer(EMBEDDING_MODEL)
    return _st_model


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


def _proxy_openai_model_name(model: str) -> str:
    """助教 OpenAI-compatible proxy：須用 openai/ 前綴，否則 gemini-* 會被導向 Vertex。"""
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

    base = (os.environ.get("LITELLM_BASE_URL") or "").strip()
    key = (os.environ.get("LITELLM_API_KEY") or "").strip()
    m = model or os.environ.get("LITELLM_MODEL", "gemini-2.5-flash")
    if base and key:
        return completion(
            model=_proxy_openai_model_name(m),
            api_base=base,
            api_key=key,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
    return completion(model=m, messages=messages, temperature=temperature, **kwargs)
