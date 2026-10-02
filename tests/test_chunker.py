from __future__ import annotations

import pytest
from app.rag.chunker import chunk_document, chunk_text, normalize, split_blocks

SENTENCE = "Week 3 covers eigenvalues and diagonalisation of square matrices. "


def test_normalize_collapses_whitespace_but_keeps_paragraph_breaks() -> None:
    assert normalize("a   b\r\nc\n\n\n  d  ") == "a b\nc\n\nd"


def test_split_blocks_drops_empty_paragraphs() -> None:
    assert split_blocks("one\n\n\n\ntwo\n\n") == ["one", "two"]


def test_empty_text_produces_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_paragraph_is_a_single_chunk() -> None:
    chunks = chunk_text("Office hours are on Fridays.", max_chars=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == "Office hours are on Fridays."
    assert chunks[0].ordinal == 0


def test_chunks_never_exceed_the_limit() -> None:
    text = SENTENCE * 40
    chunks = chunk_text(text, max_chars=200, overlap=40)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 for chunk in chunks)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_oversized_paragraph_is_split_on_sentence_boundaries() -> None:
    text = "Alpha beta gamma. " * 30
    chunks = chunk_text(text, max_chars=120, overlap=0)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    for chunk in chunks:
        assert not chunk.text.startswith(" gamma")


def test_overlap_carries_the_previous_sentence_forward() -> None:
    chunks = chunk_text(SENTENCE * 6, max_chars=180, overlap=90)

    assert len(chunks) >= 2
    first_tail = chunks[0].text.split(" ")[-1]
    assert first_tail in chunks[1].text


def test_a_single_giant_sentence_is_hard_split() -> None:
    chunks = chunk_text("x" * 500, max_chars=120, overlap=0)

    assert [len(chunk.text) for chunk in chunks] == [120, 120, 120, 120, 20]


def test_repeated_runs_are_identical() -> None:
    text = "\n\n".join(SENTENCE * 3 for _ in range(6))
    first = chunk_text(text, max_chars=300, overlap=50)
    second = chunk_text(text, max_chars=300, overlap=50)
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]


def test_no_content_is_lost_across_chunks() -> None:
    words = [f"w{index}" for index in range(60)]
    chunks = chunk_text(" ".join(words), max_chars=200, overlap=0)

    recovered = " ".join(chunk.text for chunk in chunks).split()
    assert recovered == words


def test_chunk_document_attaches_metadata() -> None:
    payload = chunk_document(
        "First lecture covers vectors. Second covers matrices.",
        title="Linear Algebra",
        source="notes/week-01.md",
        max_chars=60,
        overlap=10,
    )

    assert payload
    for entry in payload:
        assert entry["title"] == "Linear Algebra"
        assert entry["source"] == "notes/week-01.md"
        assert isinstance(entry["ordinal"], int)
        assert entry["text"]


def test_chunk_document_defaults_metadata_to_empty_strings() -> None:
    payload = chunk_document("A short note.", max_chars=100, overlap=10)
    assert payload == [{"text": "A short note.", "ordinal": 0, "title": "", "source": ""}]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_chars": 0}, "max_chars"),
        ({"max_chars": 100, "overlap": 100}, "overlap"),
        ({"max_chars": 100, "overlap": -1}, "overlap"),
    ],
)
def test_invalid_limits_are_rejected(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        chunk_text("text", **kwargs)
