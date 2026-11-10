#!/usr/bin/env python3
"""資料清理、chunk、embedding、寫入向量庫，並維護該 collection 的 BM25 索引。

支援 --rebuild 全量重建與增量更新（檔案 hash）；兩者結束時 BM25 索引都會與向量庫同步。
--rebuild 會從 data/raw 重新抽取並改寫已提交的 data/processed；加 --reuse-processed 則直接以
data/processed 的文字建索引，不改動 data/（與實驗及 experiments/corpus_pin.json 使用同一份文字）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections.abc import Callable, Collection
from pathlib import Path

from rag_common import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PROCESSED,
    DATA_RAW,
    MANIFEST_PATH,
    describe_exception,
    embed_texts,
    get_chroma_collection,
    refresh_bm25_index,
    resolve_collection_name,
    save_bm25_index,
    text_sha256,
)

SUPPORTED_SUFFIXES = frozenset({".md", ".txt", ".markdown", ".pdf"})


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict[str, str]:
    if not MANIFEST_PATH.is_file():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(
            f"[data_update] 警告：ingest manifest 無法解析（{type(exc).__name__}），視為空白並重新處理所有檔案。",
            file=sys.stderr,
        )
        return {}


def _save_manifest(m: dict[str, str]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def _strip_noise(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return _strip_noise("\n".join(parts))


def _read_textish(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "big5", "cp950", "latin-1"):
        try:
            return _strip_noise(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    return _strip_noise(raw.decode("utf-8", errors="replace"))


def extract_clean_text(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return _read_pdf(path)
    if suf in {".md", ".txt", ".markdown"}:
        return _read_textish(path)
    raise ValueError(f"不支援的副檔名: {suf}")


def naive_chunk_text(text: str, size: int) -> list[str]:
    """純字元切割（無段落感知、無 overlap）—— Combo A baseline 用。"""
    if not text:
        return []
    return [
        text[i : i + size].strip()
        for i in range(0, len(text), size)
        if text[i : i + size].strip()
    ]


def _split_by_separators(text: str, separators: list[str], max_chars: int) -> list[str]:
    """遞迴式分割：依 separators 優先序切，超出 max_chars 時往下一層 separator。"""
    if not separators:
        # 字元層最終兜底
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars) if text[i : i + max_chars].strip()]
    sep = separators[0]
    parts = text.split(sep) if sep else list(text)
    result: list[str] = []
    buf = ""
    for part in parts:
        candidate = (buf + sep + part).lstrip(sep) if buf else part
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf.strip():
                result.append(buf.strip())
            if len(part) > max_chars:
                # 這一片段自身太長，往更細的 separator 遞迴
                result.extend(_split_by_separators(part, separators[1:], max_chars))
                buf = ""
            else:
                buf = part
    if buf.strip():
        result.append(buf.strip())
    return result


def _extract_section_title(text_before: str) -> str:
    """從 chunk 之前的文字中取最近的 Markdown heading（## 或 #）。"""
    headings = re.findall(r"^#{1,3}\s+(.+)$", text_before, re.MULTILINE)
    return headings[-1].strip() if headings else ""


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """段落感知切割：優先在段落邊界切，超過 size 時遞迴細分，最後加 overlap 拼接。"""
    if not text:
        return []
    separators = ["\n\n", "\n", "。", ".", " ", ""]
    raw_chunks = _split_by_separators(text, separators, size)
    if not raw_chunks:
        return []
    # 加 overlap：每個 chunk 前附上前一 chunk 的最後 overlap 個字元
    result: list[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0 or overlap <= 0:
            result.append(chunk)
        else:
            prefix = raw_chunks[i - 1][-overlap:]
            merged = (prefix + "\n" + chunk).strip()
            result.append(merged)
    return result


def _processed_path_for(raw_path: Path) -> Path:
    return DATA_PROCESSED / f"{raw_path.stem}.txt"


def list_raw_documents() -> list[Path]:
    """data/raw 內所有支援格式的原始文件（依路徑排序）。"""
    return sorted(
        p for p in DATA_RAW.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def select_raw_documents(include: Collection[str] | None = None) -> list[Path]:
    """list_raw_documents() 中檔名在 include 內的文件（保持排序）；include 為 None 時回傳全部。

    include 是單一字串時拋 TypeError（避免被當成字元集合）；為空、或含 data/raw 中不存在
    （或副檔名不支援）的檔名時拋 ValueError，不會默默略過。
    """
    documents = list_raw_documents()
    if include is None:
        return documents
    if isinstance(include, str):
        raise TypeError("include 必須是 data/raw 檔名的集合，不可是單一字串")
    wanted = set(include)
    if not wanted:
        raise ValueError("include 不可為空（None 代表全部文件）")
    missing = sorted(wanted - {p.name for p in documents})
    if missing:
        shown = "、".join(missing[:3]) + ("…" if len(missing) > 3 else "")
        raise ValueError(f"include 有 {len(missing)} 個檔名不在 data/raw 的支援文件中（{shown}）")
    return [p for p in documents if p.name in wanted]


def _load_document_text(raw: Path, reuse_processed: bool) -> str:
    """取得文件清理後的文字：reuse_processed 時優先讀 data/processed/<stem>.txt（不寫入任何檔案）。"""
    if reuse_processed:
        processed = _processed_path_for(raw)
        if processed.is_file():
            return processed.read_text(encoding="utf-8")
    return extract_clean_text(raw)


def _clear_collection(coll, collection_name: str | None) -> None:
    """刪除 collection 內所有 chunk；失敗時拋 RuntimeError（chromadb 對重複 id 的 add 不會覆寫舊文字）。"""
    try:
        ids = coll.get(include=[]).get("ids") or []
        if ids:
            coll.delete(ids=ids)
    except Exception as exc:
        name = resolve_collection_name(collection_name)
        raise RuntimeError(
            f"清空 collection {name!r} 失敗（{type(exc).__name__}）；中止重建，避免新舊 chunk 混雜。"
        ) from exc


def rebuild_processed_and_index(
    chunk_fn: Callable[[str], list[str]] | None = None,
    collection_name: str | None = None,
    reuse_processed: bool = False,
    write_processed: bool = True,
    include: Collection[str] | None = None,
) -> dict[str, str]:
    """全量重建：清空 Chroma collection、重新 chunk + embed + 寫入，並重建該 collection 的 BM25 索引。

    chunk_fn: 自訂切割函式 (text -> list[str])；為 None 時使用預設 chunk_text。
    collection_name: 指定 Chroma collection；為 None 時使用 rag_common.CHROMA_COLLECTION。
    reuse_processed: True 時優先讀取既有的 data/processed/<stem>.txt（不存在才從 data/raw 抽取，僅存於記憶體），
        不刪除、不寫入 data/processed，也不動 ingest manifest；False 時一律從 data/raw 重新抽取。
    write_processed: 只在 reuse_processed=False 時有作用。True（預設，CLI --rebuild）時清空 data/processed
        與 ingest manifest 並寫回新抽取的文字；False 時新抽取的文字只存於記憶體，不動 data/ 下任何檔案
        （experiment_runner 的 --re-extract 如此使用）。
    include: 只為這些 data/raw 檔名建索引（例如 experiments/corpus_round1.json 的 39 份）；None 為全部文件。
        會寫回 data/processed 的模式（write_processed=True 且 reuse_processed=False）不接受 include，
        否則 data/processed 只會剩下部分文件。
    include 不合法時拋 TypeError / ValueError，清空 collection 失敗時拋 RuntimeError；兩者都發生在
    動到 collection 或 data/processed 之前。
    回傳 {processed 檔名: 本次建索引所用文字的 sha256}（只含所選文件），供實驗記錄語料指紋。
    """
    persist = write_processed and not reuse_processed
    if include is not None and persist:
        raise ValueError(
            "include 不能搭配會寫回 data/processed 的重建（請用 reuse_processed=True 或 write_processed=False）"
        )
    raw_documents = select_raw_documents(include)
    coll = get_chroma_collection(collection_name)
    _clear_collection(coll, collection_name)

    if persist:
        if DATA_PROCESSED.exists():
            shutil.rmtree(DATA_PROCESSED)
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        if MANIFEST_PATH.is_file():
            MANIFEST_PATH.unlink()

    _chunk = chunk_fn or (lambda t: chunk_text(t, CHUNK_SIZE, CHUNK_OVERLAP))

    manifest: dict[str, str] = {}
    text_hashes: dict[str, str] = {}
    all_ids: list[str] = []
    all_docs: list[str] = []
    all_metas: list[dict] = []

    for raw in raw_documents:
        rel = raw.relative_to(DATA_RAW)
        text = _load_document_text(raw, reuse_processed)
        out = _processed_path_for(raw)
        if persist:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
        text_hashes[out.name] = text_sha256(text)
        digest = _sha256_file(raw)
        manifest[str(rel)] = digest

        chunks = _chunk(text)
        base = out.name
        for i, ch in enumerate(chunks):
            cid = f"{base}::{i}"
            char_offset = text.find(ch[:40]) if len(ch) >= 40 else text.find(ch)
            section = _extract_section_title(text[:char_offset] if char_offset > 0 else "")
            all_ids.append(cid)
            all_docs.append(ch)
            all_metas.append(
                {
                    "source_file": base,
                    "chunk_index": i,
                    "raw_relpath": str(rel),
                    "sha256": digest,
                    "section_title": section,
                    "char_offset": max(char_offset, 0),
                }
            )

    if all_docs:
        embs = embed_texts(all_docs)
        coll.add(ids=all_ids, documents=all_docs, metadatas=all_metas, embeddings=embs)
    # 每個 collection 一份 BM25 索引；語料為空時也覆寫，避免留下舊索引
    save_bm25_index(all_docs, all_ids, collection_name)

    if persist:
        _save_manifest(manifest)
    return text_hashes


def incremental_update(
    chunk_fn: Callable[[str], list[str]] | None = None,
    collection_name: str | None = None,
) -> None:
    """增量更新：只重新處理 hash 變動的檔案，最後依 collection 現況重建 BM25 索引。

    ingest manifest 是全域的（不分 collection / persist 目錄）；collection 為空時 manifest 不適用，
    視為所有檔案都已變動。刪除舊 chunk 失敗時拋 RuntimeError（避免新舊文字混雜）。
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    coll = get_chroma_collection(collection_name)
    if manifest and coll.count() == 0:
        print(
            f"[data_update] 注意：collection {resolve_collection_name(collection_name)!r} 是空的，"
            "ingest manifest 不適用，全部文件重新處理。",
            file=sys.stderr,
        )
        manifest = {}
    _chunk = chunk_fn or (lambda t: chunk_text(t, CHUNK_SIZE, CHUNK_OVERLAP))

    for raw in list_raw_documents():
        rel = raw.relative_to(DATA_RAW)
        rel_s = str(rel)
        digest = _sha256_file(raw)
        if manifest.get(rel_s) == digest:
            continue

        text = extract_clean_text(raw)
        out = _processed_path_for(raw)
        out.write_text(text, encoding="utf-8")

        base = out.name
        try:
            existing = coll.get(where={"source_file": base})
            if existing and existing.get("ids"):
                coll.delete(ids=existing["ids"])
        except Exception as exc:
            raise RuntimeError(
                f"刪除 {base} 的舊 chunk 失敗（{type(exc).__name__}）；中止更新，避免新舊 chunk 混雜。"
            ) from exc

        chunks = _chunk(text)
        if not chunks:
            manifest[rel_s] = digest
            continue

        ids = [f"{base}::{i}" for i in range(len(chunks))]
        metas = []
        for i, ch in enumerate(chunks):
            char_offset = text.find(ch[:40]) if len(ch) >= 40 else text.find(ch)
            section = _extract_section_title(text[:char_offset] if char_offset > 0 else "")
            metas.append(
                {
                    "source_file": base,
                    "chunk_index": i,
                    "raw_relpath": rel_s,
                    "sha256": digest,
                    "section_title": section,
                    "char_offset": max(char_offset, 0),
                }
            )
        embs = embed_texts(chunks)
        coll.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embs)
        manifest[rel_s] = digest

    _save_manifest(manifest)
    # 詞彙索引與向量庫同步，避免 Hybrid Search 用到過期的 BM25 索引
    refresh_bm25_index(collection_name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="從 data/raw 清理、chunk、embedding 並寫入 Chroma（可 --rebuild 全量重建）",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="清空 data/processed、重設向量集合並全量重建索引",
    )
    p.add_argument(
        "--reuse-processed",
        action="store_true",
        help="搭配 --rebuild：以已提交的 data/processed 文字建索引，不刪除、不改寫 data/processed 與 manifest",
    )
    args = p.parse_args(argv)
    if args.reuse_processed and not args.rebuild:
        p.error("--reuse-processed 只能搭配 --rebuild 使用")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.rebuild:
            rebuild_processed_and_index(reuse_processed=args.reuse_processed)
        else:
            incremental_update()
    except Exception as e:
        print(f"[data_update] 錯誤（{describe_exception(e)}）", file=sys.stderr)
        return 1
    print("data_update 完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
