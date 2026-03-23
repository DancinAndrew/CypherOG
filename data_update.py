#!/usr/bin/env python3
"""資料清理、chunk、embedding、寫入向量庫。支援 --rebuild 與增量更新（檔案 hash）。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from rag_common import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PROCESSED,
    DATA_RAW,
    MANIFEST_PATH,
    embed_texts,
    get_chroma_collection,
    save_bm25_index,
)


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
    except json.JSONDecodeError:
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


def rebuild_processed_and_index(
    chunk_fn: Callable[[str], list[str]] | None = None,
    collection_name: str | None = None,
) -> None:
    """全量重建：清空 processed/、清空 Chroma collection、重新 chunk + embed + 寫入。
    chunk_fn: 自訂切割函式 (text -> list[str])；為 None 時使用預設 chunk_text。
    collection_name: 指定 Chroma collection；為 None 時使用 rag_common.CHROMA_COLLECTION。
    """
    if DATA_PROCESSED.exists():
        shutil.rmtree(DATA_PROCESSED)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.is_file():
        MANIFEST_PATH.unlink()

    coll = get_chroma_collection(collection_name)
    try:
        existing = coll.get(include=[])
        ids = existing.get("ids") or []
        if ids:
            coll.delete(ids=ids)
    except Exception:
        pass

    _chunk = chunk_fn or (lambda t: chunk_text(t, CHUNK_SIZE, CHUNK_OVERLAP))

    manifest: dict[str, str] = {}
    raws = sorted(
        p
        for p in DATA_RAW.iterdir()
        if p.is_file() and p.suffix.lower() in {".md", ".txt", ".markdown", ".pdf"}
    )
    all_ids: list[str] = []
    all_docs: list[str] = []
    all_metas: list[dict] = []

    for raw in raws:
        rel = raw.relative_to(DATA_RAW)
        text = extract_clean_text(raw)
        out = _processed_path_for(raw)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
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
        # 同步建立 BM25 索引（混合搜尋支援）
        save_bm25_index(all_docs, all_ids)

    _save_manifest(manifest)


def incremental_update(
    chunk_fn: Callable[[str], list[str]] | None = None,
    collection_name: str | None = None,
) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    coll = get_chroma_collection(collection_name)
    _chunk = chunk_fn or (lambda t: chunk_text(t, CHUNK_SIZE, CHUNK_OVERLAP))

    raws = sorted(
        p
        for p in DATA_RAW.iterdir()
        if p.is_file() and p.suffix.lower() in {".md", ".txt", ".markdown", ".pdf"}
    )

    for raw in raws:
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
        except Exception:
            pass

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="從 data/raw 清理、chunk、embedding 並寫入 Chroma（可 --rebuild 全量重建）",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="清空 data/processed、重設向量集合並全量重建索引",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.rebuild:
            rebuild_processed_and_index()
        else:
            incremental_update()
    except Exception as e:
        print(f"[data_update] 錯誤: {e}", file=sys.stderr)
        return 1
    print("data_update 完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
