from __future__ import annotations

import re
from dataclasses import dataclass

# Sentences end at punctuation followed by whitespace. Kept simple and explicit: a heavier
# splitter would need a model download, and determinism matters more than perfection here
# because chunk boundaries decide what a student can retrieve.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RUN = re.compile(r"[ \t]+")
BLANK_LINES = re.compile(r"\n\s*\n+")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One embeddable slice of a document."""

    text: str
    ordinal: int

    def __len__(self) -> int:
        return len(self.text)


def normalize(text: str) -> str:
    """Collapse horizontal whitespace and blank lines, keeping paragraph breaks.

    Line endings are normalised first so CRLF documents, which arrive straight out of a
    Windows editor or a Markdown file, chunk the same way as LF ones.
    """
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = []
    for block in BLANK_LINES.split(cleaned):
        lines = [WHITESPACE_RUN.sub(" ", line).strip() for line in block.splitlines()]
        joined = "\n".join(line for line in lines if line)
        if joined:
            paragraphs.append(joined)
    return "\n\n".join(paragraphs)


def split_sentences(block: str) -> list[str]:
    return [part.strip() for part in SENTENCE_BOUNDARY.split(block) if part.strip()]


def split_blocks(text: str) -> list[str]:
    """Paragraphs, with blank lines collapsed. Oversized paragraphs are split later, once the
    limit is known, so this stays a pure text normalisation step."""
    return [block for block in normalize(text).split("\n\n") if block]


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Break over-long text at word boundaries.

    Used for a single sentence longer than the limit, such as a pasted table row. Splitting on
    whitespace keeps chunks readable; falling back to a blind slice only when there is no
    space at all, so a chunk never begins mid-word in normal prose.
    """
    pieces: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        cut = remaining.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _units(text: str, max_chars: int) -> list[str]:
    """Retrieval units: whole paragraphs when they fit, sentences when they do not."""
    units: list[str] = []
    for block in split_blocks(text):
        if len(block) <= max_chars:
            units.append(block)
            continue
        for sentence in split_sentences(block):
            if len(sentence) <= max_chars:
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, max_chars))
    return units


def _carry_overlap(sentences: list[str], overlap: int) -> tuple[list[str], int]:
    """Keep the trailing sentences of the previous chunk so context carries forward."""
    if overlap <= 0:
        return [], 0
    kept: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        candidate = total + (2 if kept else 0) + len(sentence)
        if candidate > overlap:
            break
        kept.append(sentence)
        total = candidate
    kept.reverse()
    return kept, total


def chunk_text(text: str, *, max_chars: int = 900, overlap: int = 150) -> list[Chunk]:
    """Split text into overlapping chunks of at most ``max_chars`` characters.

    Units are packed greedily and joined with single spaces; paragraph layout is flattened on
    purpose, because the chunker decides what a student can retrieve, not how it is typeset.
    Overlap keeps a boundary-straddling sentence whole in one chunk. Chunks come back in
    document order numbered from zero, so the same input always yields the same ordinals and
    therefore the same point ids.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")
    if not text.strip():
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in _units(text, max_chars):
        if not current:
            current, current_len = [unit], len(unit)
            continue
        if current_len + 2 + len(unit) <= max_chars:
            current.append(unit)
            current_len += 2 + len(unit)
            continue
        chunks.append(" ".join(current))
        carry, carry_len = _carry_overlap(current, overlap)
        if carry_len + 2 + len(unit) > max_chars:
            carry, carry_len = [], 0
        current = [*carry, unit]
        current_len = carry_len + (2 if carry else 0) + len(unit)
    if current:
        chunks.append(" ".join(current))

    return [Chunk(text=chunk.strip(), ordinal=ordinal) for ordinal, chunk in enumerate(chunks)]


def chunk_document(
    text: str,
    *,
    title: str | None = None,
    source: str | None = None,
    max_chars: int = 900,
    overlap: int = 150,
) -> list[dict[str, object]]:
    """Chunks plus the metadata that travels with each of them into the vector store."""
    return [
        {
            "text": chunk.text,
            "ordinal": chunk.ordinal,
            "title": title or "",
            "source": source or "",
        }
        for chunk in chunk_text(text, max_chars=max_chars, overlap=overlap)
    ]
