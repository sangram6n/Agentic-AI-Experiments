# -*- coding: utf-8 -*-
"""
ORMAS KNOWLEDGE BASE - Core RAG Engine
Coordinates vectorstore, embeddings, LLM chain, and strategy execution.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional
from collections import Counter

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from backend import config
from backend.rag.chunking import get_chunker, CHUNKING_STRATEGIES
from backend.rag.strategies import get_rag_strategy, RAG_STRATEGIES


class RAGEngine:
    """Singleton RAG Engine for ORMAS Knowledge Base."""

    def __init__(self):
        self.persist_directory = config.PERSIST_DIRECTORY
        self.embeddings = OllamaEmbeddings(
            model=config.DEFAULT_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )
        self.llm = init_chat_model(
            model=config.DEFAULT_LLM_MODEL,
            model_provider=config.DEFAULT_LLM_PROVIDER
        )
        self._vectorstore: Optional[Chroma] = None

    @property
    def vectorstore(self) -> Chroma:
        """Lazy-loaded Chroma vectorstore instance."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return self._vectorstore

    def query(
        self,
        question: str,
        strategy_name: str = "standard",
        top_k: int = 4
    ) -> Dict[str, Any]:
        """Execute a query through the selected RAG strategy."""
        start_time = time.time()
        strategy = get_rag_strategy(strategy_name)

        # Retrieve documents
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(question)

        # Build context & prompt
        context_str = strategy.format_context(docs)
        prompt = strategy.get_prompt()

        # Build execution chain
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "question": question})

        elapsed = round((time.time() - start_time) * 1000, 2)
        citations = strategy.extract_citations(docs)

        return {
            "answer": answer,
            "citations": citations,
            "strategy": strategy_name,
            "top_k": top_k,
            "elapsed_ms": elapsed
        }

    def get_document_stats(self) -> Dict[str, Any]:
        """Return counts and statistics of currently indexed files."""
        try:
            metas = self.vectorstore.get()["metadatas"]
            files = [m.get("filename", "unknown") for m in metas if m]
            counts = dict(Counter(files))
            return {
                "total_vectors": len(metas),
                "total_documents": len(counts),
                "files": counts
            }
        except Exception as e:
            return {"total_vectors": 0, "total_documents": 0, "files": {}, "error": str(e)}

    def sync_new_documents(self, chunking_strategy: str = "recursive") -> Dict[str, Any]:
        """Incremental sync: Ingests only new or unindexed files."""
        from ingestion_pipeline import load_directory

        stats = self.get_document_stats()
        existing_files = set(stats.get("files", {}).keys())

        # Load directory generator
        docs = list(load_directory(config.DATA_DIRECTORY))
        new_docs = [d for d in docs if d.metadata.get("filename") not in existing_files]

        if not new_docs:
            return {"status": "up_to_date", "new_chunks": 0, "total_vectors": stats["total_vectors"]}

        chunker = get_chunker(chunking_strategy)
        chunks = chunker.split(new_docs)

        if chunks:
            self.vectorstore.add_documents(chunks)

        updated_stats = self.get_document_stats()
        return {
            "status": "synced",
            "new_files": list(set(d.metadata.get("filename") for d in new_docs)),
            "new_chunks": len(chunks),
            "total_vectors": updated_stats["total_vectors"]
        }


# Global engine instance
engine = RAGEngine()
