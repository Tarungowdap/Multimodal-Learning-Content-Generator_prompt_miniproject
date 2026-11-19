"""Load course notes (PDF / DOCX / TXT / MD) and split into overlapping chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable


@dataclass
class Chunk:
    id: str
    source: str
    text: str


def _read_pdf(f) -> str:
    from pypdf import PdfReader

    return "\n".join((p.extract_text() or "") for p in PdfReader(f).pages)


def _read_docx(f) -> str:
    import docx

    return "\n".join(p.text for p in docx.Document(f).paragraphs)


def read_document(name: str, data: bytes | BinaryIO) -> str:
    import io

    raw = data if isinstance(data, (bytes, bytearray)) else data.read()
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return _read_pdf(io.BytesIO(raw))
    if ext == ".docx":
        return _read_docx(io.BytesIO(raw))
    return raw.decode("utf-8", errors="ignore")


def clean_text(t: str) -> str:
    """Drop NULs/odd control chars that PDF extraction sometimes leaves behind."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", t)


def read_path(path: str | Path) -> str:
    p = Path(path)
    return read_document(p.name, p.read_bytes())


def split_sentences(text: str) -> list[str]:
    # markdown headings / bullets become their own sentence instead of gluing onto the next one
    text = re.sub(r"^\s*#{1,6}\s*(.+?)\s*$", lambda m: m.group(1).rstrip(".") + ".", text, flags=re.M)
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text) if s.strip()]


def chunk_text(source: str, text: str, max_words: int = 110, overlap: int = 1) -> list[Chunk]:
    """Sentence-aware chunking with `overlap` sentences carried between chunks."""
    sents = split_sentences(text)
    chunks: list[Chunk] = []
    cur: list[str] = []
    words = 0
    for s in sents:
        w = len(s.split())
        if cur and words + w > max_words:
            chunks.append(Chunk(f"{source}#{len(chunks)}", source, " ".join(cur)))
            cur = cur[-overlap:] if overlap else []
            words = sum(len(x.split()) for x in cur)
        cur.append(s)
        words += w
    if cur:
        chunks.append(Chunk(f"{source}#{len(chunks)}", source, " ".join(cur)))
    return chunks


def build_chunks(docs: Iterable[tuple[str, str]]) -> list[Chunk]:
    out: list[Chunk] = []
    for name, text in docs:
        out.extend(chunk_text(name, text))
    return out
