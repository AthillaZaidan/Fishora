import inspect

from apps.main_api.services import chunking
from apps.main_api.services.chunking import chunk_candidate
from apps.main_api.services.corpus import CandidateChunk


def _candidate(content: str) -> CandidateChunk:
    return CandidateChunk(
        id="chunk_nila_taste_001",
        species_label="nila",
        source_id="fao_en_niletilapia",
        category="taste_texture",
        content=content,
        source_quote="quote",
        stage="knowledge_editor",
        verification_status="candidate",
    )


def test_short_record_is_kept_whole(word_tokenizer):
    chunks = chunk_candidate(_candidate("Mild flavour. Firm flesh."), word_tokenizer)
    assert [c.content for c in chunks] == ["Mild flavour. Firm flesh."]


def test_long_record_splits_within_limit_on_sentences(word_tokenizer):
    sentence = " ".join(f"w{i}" for i in range(30)) + "."
    chunks = chunk_candidate(_candidate(" ".join([sentence] * 10)), word_tokenizer,
                             max_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    assert all(len(word_tokenizer.encode(c.content)) <= 100 for c in chunks)


def test_oversized_sentence_is_windowed_with_overlap(word_tokenizer):
    text = " ".join(f"t{i}" for i in range(250))
    chunks = chunk_candidate(_candidate(text), word_tokenizer, max_tokens=100, overlap_tokens=10)
    first, second = (c.content.split() for c in chunks[:2])
    assert first[-10:] == second[:10]


def test_default_chunk_limit_fits_the_e5_window():
    """E5 truncates at 512 tokens; the 'passage: ' prefix and CLS/SEP need room."""
    limit = inspect.signature(chunking.chunk_candidate).parameters["max_tokens"].default
    assert limit + 4 <= 512
