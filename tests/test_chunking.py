"""Characterization tests for the chunkers used by every experiment combo."""

from data_update import chunk_text, naive_chunk_text


def test_paragraphs_that_fit_stay_in_one_chunk():
    text = "First paragraph.\n\nSecond paragraph."
    assert chunk_text(text, 600, 0) == [text]


def test_split_happens_at_paragraph_boundaries_first():
    p1, p2 = "a" * 400, "b" * 400
    assert chunk_text(f"{p1}\n\n{p2}", 600, 0) == [p1, p2]


def test_overlap_prefixes_the_previous_chunk_tail():
    p1, p2 = "a" * 390 + "XYZ", "b" * 400
    chunks = chunk_text(f"{p1}\n\n{p2}", 600, 3)
    assert chunks == [p1, "XYZ\n" + p2]


def test_chunks_without_overlap_never_exceed_the_size():
    words = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(words, 300, 0)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_naive_chunking_cuts_fixed_width_windows():
    assert naive_chunk_text("abcdefgh", 3) == ["abc", "def", "gh"]


def test_empty_text_yields_no_chunks():
    assert chunk_text("", 600, 100) == []
    assert naive_chunk_text("", 500) == []


# --- extensions -------------------------------------------------------------


def test_overlap_can_grow_a_chunk_by_at_most_overlap_plus_newline():
    # The size limit applies before the overlap prefix is added, so a chunk may
    # reach size + overlap + 1 characters (the "+1" is the joining newline).
    words = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(words, 300, 50)
    assert max(len(c) for c in chunks) <= 300 + 50 + 1
    assert any(len(c) > 300 for c in chunks)


def test_long_chinese_paragraph_splits_at_sentence_marks():
    # Cuts land on "。"; the mark at each cut is consumed by the split.
    body = "震感舞源自加州弗雷斯諾" * 5
    text = (body + "。") * 20
    chunks = chunk_text(text, 200, 0)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert all(c.startswith(body) for c in chunks)
    assert all(c.endswith(body) for c in chunks[:-1])


def test_naive_chunking_strips_windows_and_drops_blank_ones():
    assert naive_chunk_text("ab   cd", 2) == ["ab", "c", "d"]
