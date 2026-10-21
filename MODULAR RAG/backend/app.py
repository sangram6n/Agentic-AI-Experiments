# -*- coding: utf-8 -*-
"""
ORMAS KNOWLEDGE BASE - REST API
Fully decoupled backend with CORS support, No-Cache headers, and static UI mounting.
"""

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from backend import config
from backend.rag.engine import engine
from backend.rag.strategies import RAG_STRATEGIES
from backend.rag.chunking import CHUNKING_STRATEGIES

app = FastAPI(
    title="ORMAS Knowledge Base API",
    description="Decoupled Modular RAG API with dynamic chunking & strategy selection",
    version="2.0.0"
)

# Enable CORS for full frontend/backend decoupling
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    """Prevent aggressive browser caching for development assets."""
    response: Response = await call_next(request)
    path = request.url.path
    if path.endswith((".js", ".html", ".css", "/")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


class ChatRequest(BaseModel):
    query: str = Field(..., description="User query or greeting")
    rag_strategy: str = Field("standard", description="Selected RAG Strategy")
    top_k: int = Field(4, ge=1, le=10, description="Number of context chunks to retrieve")


class ReindexRequest(BaseModel):
    chunking_strategy: str = Field("recursive", description="Selected chunking strategy")


@app.get("/api/health")
def health_check():
    """Verify backend status and vector store health."""
    stats = engine.get_document_stats()
    return {
        "status": "healthy",
        "app": "ORMAS Knowledge Base",
        "llm_model": config.DEFAULT_LLM_MODEL,
        "embedding_model": config.DEFAULT_EMBEDDING_MODEL,
        "total_vectors": stats.get("total_vectors", 0)
    }


@app.get("/api/strategies")
def list_strategies():
    """List all available RAG and Chunking strategies."""
    from backend.rag.strategies import get_rag_strategy
    from backend.rag.chunking import get_chunker

    rag_list = [
        {"id": k, "description": get_rag_strategy(k).description}
        for k in RAG_STRATEGIES
    ]
    chunking_list = [
        {"id": k, "description": get_chunker(k).description}
        for k in CHUNKING_STRATEGIES
    ]
    return {
        "rag_strategies": rag_list,
        "chunking_strategies": chunking_list
    }


@app.get("/api/documents")
def list_documents():
    """List indexed files and collection statistics."""
    return engine.get_document_stats()


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    """Handle chat queries and return grounded answers with citations."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        response = engine.query(
            question=req.query,
            strategy_name=req.rag_strategy,
            top_k=req.top_k
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reindex")
def reindex_endpoint(req: ReindexRequest):
    """Incrementally sync new files into ChromaDB."""
    try:
        result = engine.sync_new_documents(chunking_strategy=req.chunking_strategy)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount frontend directory for easy standalone hosting
frontend_path = config.BASE_DIR / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
