# -*- coding: utf-8 -*-
"""
ORMAS KNOWLEDGE BASE - Configuration Module
Always anchors to current directory.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Prioritize current working directory (MODULAR RAG)
cwd = Path.cwd().resolve()
if (cwd / "chroma_db_day2").exists() or (cwd / ".env").exists():
    BASE_DIR = cwd
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env
load_dotenv(BASE_DIR / ".env")

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Ollama & Embeddings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# LLM Configuration
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# Storage Paths — always current directory
PERSIST_DIRECTORY = str(BASE_DIR / "chroma_db_day2")
DATA_DIRECTORY = str(BASE_DIR)

# Server Configuration
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
