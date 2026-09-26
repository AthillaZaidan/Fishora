"""Semantic chunking of approved knowledge records.

One candidate record is one semantic unit (species, source, category): it is
kept whole when at or below ``max_tokens`` even when far below ``min_tokens``
— short units are never merged to reach a target size. Longer content splits
on sentence boundaries; a single sentence above ``max_tokens`` is cut into
windows of at most ``max_tokens`` tokens with exactly one ``overlap_tokens``
overlap between consecutive windows, each window decoded independently so
partial tokens at the cut stay safe. Overlap is applied exactly once per
split: window units already carry their overlap and never receive an
additional carry, and multi-sentence chunks carry at most ``overlap_tokens``
tokens of the previous chunk. Every emitted chunk is re-encoded and asserted
to stay within ``max_tokens`` tokens. Encode/decode always run without
special tokens, so no [CLS]/[SEP]/<s>/</s> text can appear in chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from apps.main_api.ports import Tokenizer
from apps.main_api.services.corpus import CandidateChunk, Category

# Sentences end at ., ! or ? followed by whitespace; newlines also split.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|[\r\n]+")


@dataclass(frozen=True)
class ChunkPayload:
    id: str
    species_label: str
    source_id: str
    category: Category
    content: str


def _sentences(text: str) -> list[str]:
    return [part for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]


def chunk_candidate(
    candidate: CandidateChunk,
    tokenizer: Tokenizer,
    min_tokens: int = 300,
    max_tokens: int = 480,  # E5 truncates at 512 incl. prefix and CLS/SEP (W10)
    overlap_tokens: int = 50,
) -> list[ChunkPayload]:
    """Split one candidate into semantic chunks of at most ``max_tokens`` tokens.

    ``min_tokens`` is advisory only: a section below it is never split, and
    sections are never merged to reach it (the candidate is already one
    semantic unit). Chunks below ``max_tokens`` keep whole sentences; a single
    sentence above ``max_tokens`` is cut at token-ID boundaries into windows
    that overlap the previous window by exactly ``overlap_tokens`` tokens.
    When splitting, every chunk after the first carries the previous chunk's
    tail as leading context — except window units, whose head already is the
    overlap, so they never receive a second carry.
    """
    def payload(content: str) -> ChunkPayload:
        return ChunkPayload(
            id=candidate.id,
            species_label=candidate.species_label,
            source_id=candidate.source_id,
            category=candidate.category,
            content=content,
        )

    def token_count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    if token_count(candidate.content) <= max_tokens:
        return [payload(candidate.content)]

    units: list[list[int]] = []
    is_window: list[bool] = []
    for sentence in _sentences(candidate.content):
        ids = tokenizer.encode(sentence, add_special_tokens=False)
        if not ids:
            continue
        if len(ids) <= max_tokens:
            units.append(ids)
            is_window.append(False)
            continue
        # One sentence exceeds max_tokens: slice its token IDs into windows
        # of at most max_tokens tokens; consecutive windows overlap by
        # exactly overlap_tokens tokens.
        start = 0
        while start < len(ids):
            units.append(ids[start : start + max_tokens])
            is_window.append(True)
            if start + max_tokens >= len(ids):
                break
            start += max_tokens - overlap_tokens

    chunks: list[list[int]] = []
    current: list[int] = []
    for unit, window in zip(units, is_window):
        if window:
            # The window head already overlaps the previous window; a carry
            # would duplicate it inside this chunk.
            if current:
                chunks.append(current)
            current = list(unit)
        elif not current or len(current) + len(unit) <= max_tokens:
            current.extend(unit)
        else:
            chunks.append(current)
            # Overlap only when splitting: lead the next chunk with the
            # previous chunk's tail, capped so the chunk stays within
            # max_tokens (a >=551-token sentence cannot carry a full overlap).
            carry = min(overlap_tokens, max_tokens - len(unit))
            current = list(current[-carry:]) + unit if carry else list(unit)
    if current:
        chunks.append(current)

    emitted: list[ChunkPayload] = []
    for ids in chunks:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        if token_count(text) > max_tokens:
            raise ValueError(
                f"chunk for {candidate.id} re-encodes to {token_count(text)} "
                f"tokens; expected at most {max_tokens}"
            )
        emitted.append(payload(text))
    return emitted