# -*- coding: utf-8 -*-
"""
ORMAS KNOWLEDGE BASE - Modular Chunking Strategies
Enables swapping and customizing document splitting algorithms seamlessly.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    @abstractmethod
    def split(self, documents: List[Document]) -> List[Document]:
        """Split a list of documents into chunks."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """User-friendly description of this chunking strategy."""
        pass


class RecursiveCharChunker(BaseChunker):
    """
    Standard hierarchical chunking using character separators:
    ["\n\n", "\n", " ", ""]. Keeps paragraphs and sentences intact.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, **kwargs: Any):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunking_strategy"] = "recursive"
        return chunks

    @property
    def description(self) -> str:
        return f"Hierarchical recursive character splitter (size={self.chunk_size}, overlap={self.chunk_overlap})"


class SentenceChunker(BaseChunker):
    """
    Sentence-boundary aware chunker that avoids cutting mid-sentence.
    Groups full sentences until chunk_size is reached.
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 80, **kwargs: Any):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[".\n", "?\n", "!\n", ". ", "? ", "! ", "\n"]
        )

    def split(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunking_strategy"] = "sentence"
        return chunks

    @property
    def description(self) -> str:
        return f"Sentence-boundary splitter (size={self.chunk_size}, overlap={self.chunk_overlap})"


class ParagraphChunker(BaseChunker):
    """
    Splits strictly along paragraph boundaries (\n\n).
    Maintains semantic cohesion of full thoughts and tabular rows.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100, **kwargs: Any):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n"]
        )

    def split(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunking_strategy"] = "paragraph"
        return chunks

    @property
    def description(self) -> str:
        return f"Paragraph & section splitter (size={self.chunk_size}, overlap={self.chunk_overlap})"


class SemanticWindowChunker(BaseChunker):
    """
    Sliding window chunker with heavier overlap (40%) to ensure
    cross-sentence and table context is never lost.
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 160, **kwargs: Any):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " "]
        )

    def split(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunking_strategy"] = "semantic_window"
        return chunks

    @property
    def description(self) -> str:
        return f"Semantic sliding window (size={self.chunk_size}, overlap={self.chunk_overlap})"


# Strategy Registry
CHUNKING_STRATEGIES: Dict[str, type[BaseChunker]] = {
    "recursive": RecursiveCharChunker,
    "sentence": SentenceChunker,
    "paragraph": ParagraphChunker,
    "semantic_window": SemanticWindowChunker,
}


def get_chunker(strategy_name: str = "recursive", **kwargs: Any) -> BaseChunker:
    """Factory helper to obtain an instance of any registered chunker."""
    chunker_cls = CHUNKING_STRATEGIES.get(strategy_name.lower())
    if not chunker_cls:
        raise ValueError(f"Unknown chunking strategy '{strategy_name}'. Available: {list(CHUNKING_STRATEGIES.keys())}")
    return chunker_cls(**kwargs)
