# -*- coding: utf-8 -*-
"""
ORMAS KNOWLEDGE BASE - Modular RAG Strategies
Defines retrieval and prompting behaviors for the knowledge base.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


class BaseRAGStrategy(ABC):
    """Abstract base class for all RAG retrieval and generation strategies."""

    @abstractmethod
    def format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into context string for the prompt."""
        pass

    @abstractmethod
    def get_prompt(self) -> ChatPromptTemplate:
        """Return the prompt template for this strategy."""
        pass

    @abstractmethod
    def extract_citations(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Extract clean citation objects for the frontend UI."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass


class StandardRAGStrategy(BaseRAGStrategy):
    """Standard semantic retrieval with factual grounding."""

    def format_context(self, docs: List[Document]) -> str:
        return "\n\n".join(
            f"[{d.metadata.get('filename', 'Doc')} | Page {d.metadata.get('page', d.metadata.get('slide', '1'))}]\n{d.page_content}"
            for d in docs
        )

    def get_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("""You are ORMAS Knowledge Base AI Assistant.
- If the user is greeting you (e.g., "hi", "hello", "hey"), greet them warmly and offer to help with ORMAS documents.
- For all document queries, answer accurately using ONLY the context provided below.
- If the answer cannot be found in the context, clearly state: "Not found in documents."

Context:
{context}

Question: {question}
""")

    def extract_citations(self, docs: List[Document]) -> List[Dict[str, Any]]:
        citations = []
        for i, d in enumerate(docs):
            citations.append({
                "id": i + 1,
                "filename": d.metadata.get("filename", "unknown"),
                "file_type": d.metadata.get("file_type", ""),
                "page": d.metadata.get("page", d.metadata.get("slide", "")),
                "section": d.metadata.get("section", d.metadata.get("slide_title", "")),
                "snippet": d.page_content[:250].strip() + ("..." if len(d.page_content) > 250 else "")
            })
        return citations

    @property
    def description(self) -> str:
        return "Balanced semantic retrieval with clear factual grounding."


class ContextRichStrategy(BaseRAGStrategy):
    """Rich context strategy that includes detailed section & chapter provenance."""

    def format_context(self, docs: List[Document]) -> str:
        items = []
        for d in docs:
            meta = d.metadata
            header = f"=== SOURCE: {meta.get('filename')} (Page/Slide: {meta.get('page', meta.get('slide', 'N/A'))}) | Section: {meta.get('section', meta.get('slide_title', 'General'))} ==="
            items.append(f"{header}\n{d.page_content}")
        return "\n\n".join(items)

    def get_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("""You are ORMAS Senior Knowledge Base Specialist.
Provide a comprehensive, well-structured response based ONLY on the supplied context.
Include source filenames and section references where appropriate.
If the information is missing, state: "Not found in documents."

Context:
{context}

Question: {question}
""")

    def extract_citations(self, docs: List[Document]) -> List[Dict[str, Any]]:
        return [
            {
                "id": i + 1,
                "filename": d.metadata.get("filename", "unknown"),
                "file_type": d.metadata.get("file_type", ""),
                "page": d.metadata.get("page", d.metadata.get("slide", "")),
                "section": d.metadata.get("section", d.metadata.get("slide_title", "")),
                "chapter": d.metadata.get("chapter", ""),
                "snippet": d.page_content[:300].strip() + ("..." if len(d.page_content) > 300 else "")
            }
            for i, d in enumerate(docs)
        ]

    @property
    def description(self) -> str:
        return "Comprehensive deep-context retrieval with rich source citations."


class ConciseStrategy(BaseRAGStrategy):
    """Concise summary strategy that outputs executive bullet points."""

    def format_context(self, docs: List[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs)

    def get_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("""You are ORMAS Executive Assistant.
Answer the question concisely in clear, high-impact bullet points using ONLY the context below.
Avoid unnecessary filler words. If not found, say: "Not found in documents."

Context:
{context}

Question: {question}
""")

    def extract_citations(self, docs: List[Document]) -> List[Dict[str, Any]]:
        return [
            {
                "id": i + 1,
                "filename": d.metadata.get("filename", "unknown"),
                "page": d.metadata.get("page", d.metadata.get("slide", "")),
                "snippet": d.page_content[:180].strip() + "..."
            }
            for i, d in enumerate(docs)
        ]

    @property
    def description(self) -> str:
        return "High-impact, concise bullet-point answers for quick executive review."


class ComparativeStrategy(BaseRAGStrategy):
    """Specialized for tabular data, metrics, and multi-entity status reports."""

    def format_context(self, docs: List[Document]) -> str:
        return "\n\n".join(
            f"[{d.metadata.get('filename')} | Slide/Page {d.metadata.get('page', d.metadata.get('slide', ''))}]\n{d.page_content}"
            for d in docs
        )

    def get_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("""You are ORMAS Data & Analytics Specialist.
The user is asking about statistics, project status, or comparative metrics.
Synthesize the context into clear Markdown tables, metric breakdowns, and bulleted observations.
Base your analysis ONLY on the given context. If not found, say: "Not found in documents."

Context:
{context}

Question: {question}
""")

    def extract_citations(self, docs: List[Document]) -> List[Dict[str, Any]]:
        return [
            {
                "id": i + 1,
                "filename": d.metadata.get("filename", "unknown"),
                "page": d.metadata.get("page", d.metadata.get("slide", "")),
                "section": d.metadata.get("section", d.metadata.get("slide_title", "")),
                "snippet": d.page_content[:250].strip() + "..."
            }
            for i, d in enumerate(docs)
        ]

    @property
    def description(self) -> str:
        return "Formats metrics, project statuses, and data into structured tables."


RAG_STRATEGIES: Dict[str, type[BaseRAGStrategy]] = {
    "standard": StandardRAGStrategy,
    "context_rich": ContextRichStrategy,
    "concise": ConciseStrategy,
    "comparative": ComparativeStrategy,
}


def get_rag_strategy(strategy_name: str = "standard") -> BaseRAGStrategy:
    cls = RAG_STRATEGIES.get(strategy_name.lower())
    if not cls:
        raise ValueError(f"Unknown RAG strategy '{strategy_name}'. Available: {list(RAG_STRATEGIES.keys())}")
    return cls()
