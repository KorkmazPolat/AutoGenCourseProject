from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

from rag_ingest import build_qdrant_client


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


class RagRetrieverError(RuntimeError):
    """Raised when retrieval from Qdrant fails or cannot be performed."""


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    page_start: Optional[int]
    page_end: Optional[int]
    source: str
    score: float


def _get_openai_embedding_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RagRetrieverError("OPENAI_API_KEY must be set to retrieve supporting context.")
    return OpenAI(api_key=api_key)


def _get_qdrant_client() -> QdrantClient:
    try:
        return build_qdrant_client()
    except Exception as exc:  # pragma: no cover
        raise RagRetrieverError(f"Unable to connect to Qdrant: {exc}") from exc


def retrieve_chunks(
    query: str,
    *,
    top_k: int,
    min_score: Optional[float] = None,
) -> List[RetrievedChunk]:
    """Return the most relevant document chunks for the supplied query."""
    normalized_query = (query or "").strip()
    if not normalized_query:
        return []

    embed_model = os.getenv("QDRANT_EMBED_MODEL", "text-embedding-3-small")
    client = _get_openai_embedding_client()
    try:
        embedding_response = client.embeddings.create(model=embed_model, input=[normalized_query])
    except Exception as exc:  # pragma: no cover
        raise RagRetrieverError(f"Embedding request failed: {exc}") from exc

    vector = embedding_response.data[0].embedding

    collection_name = os.getenv("QDRANT_COLLECTION", "course_material_docs")
    qdrant_client = _get_qdrant_client()
    try:
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:  # pragma: no cover
        raise RagRetrieverError(f"Qdrant search failed: {exc}") from exc

    chunks: List[RetrievedChunk] = []
    for result in results:
        payload = result.payload or {}
        text = (payload.get("text") or "").strip()
        if not text:
            continue
        score = getattr(result, "score", None)
        if min_score is not None and score is not None and score < min_score:
            continue
        chunks.append(
            RetrievedChunk(
                text=text,
                page_start=payload.get("page_start"),
                page_end=payload.get("page_end"),
                source=payload.get("source", collection_name),
                score=float(score) if score is not None else 0.0,
            )
        )
    return chunks


def format_chunks(chunks: Sequence[RetrievedChunk], *, max_chars: int) -> str:
    """Format retrieved chunks into a compact context string."""
    if not chunks:
        return ""

    remaining = max(max_chars, 0)
    formatted_parts: List[str] = []

    for chunk in chunks:
        if remaining <= 0:
            break
        pages = ""
        if chunk.page_start is not None and chunk.page_end is not None:
            if chunk.page_start == chunk.page_end:
                pages = f"page {chunk.page_start}"
            else:
                pages = f"pages {chunk.page_start}-{chunk.page_end}"
        elif chunk.page_start is not None:
            pages = f"page {chunk.page_start}"
        header = f"[Source: {chunk.source}"
        if pages:
            header += f", {pages}"
        header += "]"

        body = chunk.text.strip()
        segment = f"{header}\n{body}"

        if len(segment) > remaining:
            segment = segment[: remaining].rstrip()

        formatted_parts.append(segment)
        remaining -= len(segment) + 2  # account for separator

    return "\n\n".join(formatted_parts)


def build_context(
    query: str,
    *,
    top_k: int,
    max_chars: int,
    min_score: Optional[float] = None,
) -> str:
    """Retrieve and format supporting context text for the supplied query."""
    chunks = retrieve_chunks(query, top_k=top_k, min_score=min_score)
    return format_chunks(chunks, max_chars=max_chars)
