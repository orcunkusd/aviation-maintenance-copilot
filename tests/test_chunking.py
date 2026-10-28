import pytest
from chunking import chunk_text, CHUNK_CHARS, OVERLAP


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []


def test_short_text_is_one_chunk():
    assert chunk_text("hello") == ["hello"]


def test_chunk_sizes_and_count():
    chunks = chunk_text("x" * 2500)
    assert [len(c) for c in chunks] == [1000, 1000, 800]


def test_consecutive_chunks_overlap():
    text = "".join(chr(97 + i % 26) for i in range(3000))
    chunks = chunk_text(text)
    assert chunks[0][-OVERLAP:] == chunks[1][:OVERLAP]


def test_no_text_is_lost():
    text = "".join(chr(97 + i % 26) for i in range(2345))
    chunks = chunk_text(text)
    step = CHUNK_CHARS - OVERLAP
    rebuilt = "".join(c[:step] for c in chunks[:-1]) + chunks[-1]
    assert rebuilt == text


def test_bad_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)
