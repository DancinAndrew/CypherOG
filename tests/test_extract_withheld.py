"""data_update.py --extract-withheld：自備的 withheld 副本只有在文字與鎖定指紋相同時才寫入 data/processed。"""

from __future__ import annotations

import pytest

import data_update
from rag_common import text_sha256

PINNED_TEXT = "# Chapter\n\nThe pinned text."


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    monkeypatch.setattr(data_update, "DATA_RAW", raw)
    monkeypatch.setattr(data_update, "DATA_PROCESSED", processed)
    return raw, processed


def withheld(sha256: str = text_sha256(PINNED_TEXT)) -> dict:
    return {"chapter.txt": {"raw": "chapter.md", "processed": "chapter.txt", "sha256": sha256, "reason": "test"}}


def test_nothing_is_written_when_no_copy_was_supplied(dirs):
    raw, processed = dirs
    assert data_update.extract_withheld(withheld()) == {"chapter.txt": "not supplied"}
    assert list(processed.iterdir()) == []


def test_a_matching_copy_is_written_once(dirs):
    raw, processed = dirs
    (raw / "chapter.md").write_text(PINNED_TEXT + "\n\n\n", encoding="utf-8")  # 清理後與鎖定文字相同
    assert data_update.extract_withheld(withheld()) == {"chapter.txt": "written"}
    assert (processed / "chapter.txt").read_text(encoding="utf-8") == PINNED_TEXT
    assert data_update.extract_withheld(withheld()) == {"chapter.txt": "already present"}


def test_a_copy_whose_text_differs_from_the_pin_is_refused(dirs):
    raw, processed = dirs
    (raw / "chapter.md").write_text("another edition", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        data_update.extract_withheld(withheld())
    assert list(processed.iterdir()) == []


def test_existing_processed_text_that_differs_is_never_overwritten(dirs):
    raw, processed = dirs
    (raw / "chapter.md").write_text(PINNED_TEXT, encoding="utf-8")
    (processed / "chapter.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="chapter.txt"):
        data_update.extract_withheld(withheld())
    assert (processed / "chapter.txt").read_text(encoding="utf-8") == "stale"


def test_the_flag_cannot_be_combined_with_a_rebuild():
    with pytest.raises(SystemExit):
        data_update.parse_args(["--extract-withheld", "--rebuild"])
    assert data_update.parse_args(["--extract-withheld"]).extract_withheld is True
