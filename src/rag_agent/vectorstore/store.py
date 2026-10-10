"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        import chromadb
        Path(self._settings.chroma_db_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._settings.chroma_db_path)
        self._collection = self._client.get_or_create_collection(
            name=self._settings.chroma_collection_name, metadata={"hnsw:space": "cosine"})

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        return hashlib.sha256(f"{source}::{chunk_text}".encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        return bool(self._collection.get(ids=[chunk_id])["ids"])

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        result = IngestionResult()
        pending = []
        seen = set()
        for chunk in chunks:
            if chunk.chunk_id in seen or self.check_duplicate(chunk.chunk_id):
                result.skipped += 1
            else:
                pending.append(chunk)
                seen.add(chunk.chunk_id)
        for offset in range(0, len(pending), 100):
            batch = pending[offset:offset + 100]
            try:
                embeddings = self._embeddings.embed_documents([c.chunk_text for c in batch])
                self._collection.upsert(ids=[c.chunk_id for c in batch],
                    documents=[c.chunk_text for c in batch],
                    embeddings=embeddings, metadatas=[c.metadata.to_dict() for c in batch])
                result.ingested += len(batch)
                result.document_ids.extend(sorted({c.metadata.source for c in batch if c.metadata.source not in result.document_ids}))
            except Exception as exc:
                logger.exception("Ingestion failed")
                result.errors.extend([f"{c.metadata.source}: {exc}" for c in batch])
        return result

    def query(self, query_text: str, k: int | None = None,
              topic_filter: str | None = None,
              difficulty_filter: str | None = None) -> list[RetrievedChunk]:
        if not query_text.strip() or self._collection.count() == 0:
            return []
        filters = {}
        if topic_filter:
            filters["topic"] = topic_filter
        if difficulty_filter:
            filters["difficulty"] = difficulty_filter
        where = ({"$and": [{key: value} for key, value in filters.items()]}
                 if len(filters) > 1 else filters or None)
        result = self._collection.query(query_embeddings=[self._embeddings.embed_query(query_text)],
            n_results=min(k or self._settings.retrieval_k, self._collection.count()),
            where=where, include=["documents", "metadatas", "distances"])
        chunks = []
        for chunk_id, content, metadata, distance in zip(result["ids"][0],
                result["documents"][0], result["metadatas"][0], result["distances"][0]):
            score = max(0.0, min(1.0, 1.0 - distance))
            if score >= self._settings.similarity_threshold:
                chunks.append(RetrievedChunk(chunk_id=chunk_id, chunk_text=content,
                    metadata=ChunkMetadata.from_dict(metadata), score=score))
        return sorted(chunks, key=lambda chunk: chunk.score, reverse=True)

    def list_documents(self) -> list[dict]:
        records = self._collection.get(include=["metadatas"])
        grouped = {}
        for meta in records["metadatas"]:
            source = meta["source"]
            if source not in grouped:
                grouped[source] = {"source": source, "topic": meta["topic"], "chunk_count": 0}
            grouped[source]["chunk_count"] += 1
        return sorted(grouped.values(), key=lambda row: row["source"])

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        records = self._collection.get(where={"source": source}, include=["documents", "metadatas"])
        return [DocumentChunk(chunk_id=id_, chunk_text=body,
                metadata=ChunkMetadata.from_dict(meta))
                for id_, body, meta in zip(records["ids"], records["documents"], records["metadatas"])]

    def get_collection_stats(self) -> dict:
        docs = self.list_documents()
        topics = sorted({doc["topic"] for doc in docs})
        return {"total_chunks": self._collection.count(), "topics": topics,
                "sources": [doc["source"] for doc in docs],
                "bonus_topics_present": bool(set(topics) & {"SOM", "BoltzmannMachine", "GAN"})}

    def delete_document(self, source: str) -> int:
        ids = self._collection.get(where={"source": source})["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)
