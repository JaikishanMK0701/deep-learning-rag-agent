"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.

    Parameters
    ----------
    settings : Settings, optional
        Application settings.

    Example
    -------
    >>> chunker = DocumentChunker()
    >>> chunks = chunker.chunk_file(
    ...     Path("data/corpus/lstm.md"),
    ...     metadata_overrides={"topic": "LSTM", "difficulty": "intermediate"}
    ... )
    >>> print(f"Produced {len(chunks)} chunks")
    """

    # Default chunking parameters — justify these in your architecture diagram.
    # chunk_size: 512 tokens balances context richness with retrieval precision.
    # chunk_overlap: 50 tokens prevents concepts that span chunk boundaries
    # from being lost entirely. A common interview question.
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(self, file_path: Path, metadata_overrides: dict | None = None,
                   chunk_size: int = DEFAULT_CHUNK_SIZE,
                   chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[DocumentChunk]:
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Invalid chunk size or overlap")
        if file_path.suffix.lower() != ".md":
            raise ValueError("This workshop supports Markdown (.md) files only")
        raw = self._chunk_markdown(file_path, chunk_size, chunk_overlap)
        metadata = self._infer_metadata(file_path, metadata_overrides)
        return [DocumentChunk(chunk_id=VectorStoreManager.generate_chunk_id(metadata.source, entry["text"]),
                chunk_text=entry["text"], metadata=metadata) for entry in raw]

    def chunk_files(self, file_paths: list[Path], metadata_overrides: dict | None = None) -> list[DocumentChunk]:
        chunks = []
        for path in file_paths:
            try:
                chunks.extend(self.chunk_file(path, metadata_overrides))
            except Exception:
                logger.exception("Unable to chunk {}", path)
                raise
        return chunks

    def _chunk_markdown(self, file_path: Path, chunk_size: int, chunk_overlap: int) -> list[dict]:
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        content = file_path.read_text(encoding="utf-8")
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "title"),
            ("##", "section"), ("###", "subsection")], strip_headers=False)
        sections = header_splitter.split_text(content)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        result = []
        for section in sections:
            for text in splitter.split_text(section.page_content):
                if text.strip():
                    result.append({"text": text, "header": section.metadata.get("section", "")})
        return result

    def _infer_metadata(self, file_path: Path, overrides: dict | None = None) -> ChunkMetadata:
        stem = file_path.stem
        pieces = stem.rsplit("_", 1)
        difficulty = pieces[-1].lower() if len(pieces) == 2 and pieces[-1].lower() in {"beginner", "intermediate", "advanced"} else "beginner"
        topic = pieces[0] if len(pieces) == 2 and difficulty == pieces[-1].lower() else stem
        canonical = {"lstm": "LSTM", "cnn": "CNN", "ann": "ANN", "rnn": "RNN",
            "seq2seq": "Seq2Seq", "autoencoder": "Autoencoder", "som": "SOM",
            "gan": "GAN", "boltzmannmachine": "BoltzmannMachine"}
        topic = canonical.get(topic.lower(), topic)
        data = {"topic": topic, "difficulty": difficulty, "type": "concept_explanation",
                "source": file_path.name, "related_topics": [],
                "is_bonus": topic in {"SOM", "GAN", "BoltzmannMachine"}}
        data.update(overrides or {})
        return ChunkMetadata(**data)
