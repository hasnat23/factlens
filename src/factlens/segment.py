"""Sentence segmentation with character offsets.

A small rule-based splitter is used instead of a heavy NLP dependency. It keeps
character offsets so that scores can be mapped back onto the original text
(for highlighting, or for aligning with span-level hallucination labels).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Abbreviations that end with a period but do not end a sentence.
_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "mt",
    "vs",
    "etc",
    "inc",
    "ltd",
    "co",
    "corp",
    "no",
    "fig",
    "approx",
    "dept",
    "est",
    "gen",
    "gov",
    "sen",
    "rep",
    "lt",
    "col",
    "sgt",
    "capt",
    "u.s",
    "u.k",
    "e.g",
    "i.e",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}

# Candidate boundary: terminal punctuation, optional closing quote/bracket, whitespace.
_BOUNDARY = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s+)")


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int


def _is_abbreviation(text: str, dot_index: int) -> bool:
    """True if the period at ``dot_index`` terminates a known abbreviation or initial."""
    match = re.search(r"([A-Za-z.]+)$", text[:dot_index])
    if not match:
        return False
    token = match.group(1).lower().rstrip(".")
    if token in _ABBREVIATIONS:
        return True
    # Single-letter initials such as "J. K. Rowling".
    return len(token) == 1 and token.isalpha()


def split_sentences(text: str, min_chars: int = 3) -> list[Sentence]:
    """Split ``text`` into sentences, preserving character offsets.

    Newlines that separate paragraphs or list items are treated as boundaries.
    Fragments shorter than ``min_chars`` are merged into the previous sentence.
    """
    boundaries: list[int] = []
    for match in _BOUNDARY.finditer(text):
        end = match.end()
        if text[match.start()] == "." and _is_abbreviation(text, match.start()):
            continue
        # Next non-space character should start a new sentence (upper-case, digit, quote).
        rest = text[end:].lstrip()
        if rest and not (rest[0].isupper() or rest[0].isdigit() or rest[0] in "\"'“(["):
            continue
        boundaries.append(end)
    for match in re.finditer(r"\n\s*\n|\n(?=\s*(?:[-*•]|\d+[.)])\s)", text):
        boundaries.append(match.start())

    sentences: list[Sentence] = []
    cursor = 0
    for boundary in sorted(set(boundaries)) + [len(text)]:
        _append(text, cursor, boundary, sentences, min_chars)
        cursor = boundary
    return sentences


def _append(text: str, start: int, end: int, out: list[Sentence], min_chars: int) -> None:
    chunk = text[start:end]
    stripped = chunk.strip()
    if not stripped:
        return
    start += len(chunk) - len(chunk.lstrip())
    end = start + len(stripped)
    if out and len(stripped) < min_chars:
        prev = out[-1]
        out[-1] = Sentence(text[prev.start : end], prev.start, end)
        return
    out.append(Sentence(stripped, start, end))


def chunk_sentences(sentences: list[Sentence], size: int, stride: int) -> list[str]:
    """Group consecutive sentences into overlapping windows of ``size`` sentences."""
    if not sentences:
        return []
    if len(sentences) <= size:
        return [" ".join(s.text for s in sentences)]
    chunks = []
    for i in range(0, len(sentences) - size + stride, stride):
        window = sentences[i : i + size]
        if window:
            chunks.append(" ".join(s.text for s in window))
        if i + size >= len(sentences):
            break
    return chunks
