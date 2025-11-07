from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


class IngestError(RuntimeError):
    """Raised when PDF ingestion fails."""


@dataclass(frozen=True)
class IngestStats:
    filename: str
    pages: int
    chunks: int
    collection: str


def ingest_pdf_into_qdrant(pdf_path: Path) -> IngestStats:
    """Read a PDF, chunk its contents, embed with OpenAI, and upsert into Qdrant."""
    if not pdf_path.exists():
        raise IngestError(f"PDF file not found: {pdf_path}")

    pages = _extract_pdf_pages(pdf_path)
    if not pages:
        raise IngestError("No readable text found in the uploaded PDF.")

    chunk_size = int(os.getenv("QDRANT_CHUNK_SIZE", "1200"))
    chunk_overlap = int(os.getenv("QDRANT_CHUNK_OVERLAP", "200"))
    chunks = list(_chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    if not chunks:
        raise IngestError("Unable to build text chunks from the uploaded PDF.")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise IngestError("OPENAI_API_KEY must be set to embed documents.")

    embed_model = os.getenv("QDRANT_EMBED_MODEL", "text-embedding-3-small")
    batch_size = int(os.getenv("QDRANT_EMBED_BATCH", "64"))

    client = OpenAI(api_key=openai_api_key)
    vectors = _embed_chunks(client, [chunk["text"] for chunk in chunks], model=embed_model, batch_size=batch_size)
    if len(vectors) != len(chunks):
        raise IngestError("Embedding service returned an unexpected number of vectors.")

    collection_name = os.getenv("QDRANT_COLLECTION", "course_material_docs")
    qdrant_client = build_qdrant_client()

    first_vector_size = len(vectors[0])
    _ensure_collection(qdrant_client, collection_name, first_vector_size)

    points = [
        qmodels.PointStruct(
            id=uuid.uuid4().hex,
            vector=vector,
            payload={
                "text": chunk["text"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "source": pdf_path.name,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    qdrant_client.upsert(collection_name=collection_name, points=points)

    return IngestStats(
        filename=pdf_path.name,
        pages=len(pages),
        chunks=len(chunks),
        collection=collection_name,
    )


def _extract_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract text per page, returning 1-indexed page numbers."""
    reader = PdfReader(str(pdf_path))
    pages: List[Tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover
            raise IngestError(f"Failed to extract text from page {index}: {exc}") from exc
        normalized = "\n".join(line.strip() for line in text.splitlines())
        normalized = normalized.strip()
        if normalized:
            pages.append((index, normalized))
    return pages


def _chunk_pages(
    pages: Sequence[Tuple[int, str]],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> Iterable[dict]:
    """Greedy chunking with character overlap across sequential pages."""
    buffer: List[str] = []
    buffer_length = 0
    start_page = None
    last_page = None

    for page_number, page_text in pages:
        paragraphs = [para.strip() for para in page_text.split("\n") if para.strip()]
        for paragraph in paragraphs:
            if not start_page:
                start_page = page_number
            buffer.append(paragraph)
            buffer_length += len(paragraph) + 1  # include space
            last_page = page_number

            if buffer_length >= chunk_size:
                joined = " ".join(buffer).strip()
                if joined:
                    yield {
                        "text": joined,
                        "page_start": start_page,
                        "page_end": last_page,
                    }
                if chunk_overlap > 0 and buffer_length > chunk_overlap:
                    overlap_text = joined[-chunk_overlap:]
                    buffer = [overlap_text]
                    buffer_length = len(overlap_text)
                    start_page = last_page
                else:
                    buffer = []
                    buffer_length = 0
                    start_page = None

    if buffer:
        joined = " ".join(buffer).strip()
        if joined:
            yield {
                "text": joined,
                "page_start": start_page or pages[0][0],
                "page_end": last_page or pages[-1][0],
            }


def _embed_chunks(client: OpenAI, texts: Sequence[str], *, model: str, batch_size: int) -> List[List[float]]:
    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def build_qdrant_client() -> QdrantClient:
    api_key = os.getenv("QDRANT_API_KEY") or None
    url = os.getenv("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=api_key)

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(url=f"http://{host}:{port}", api_key=api_key)


def _ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    if client.collection_exists(collection):
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
    )
