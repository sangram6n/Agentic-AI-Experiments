# -*- coding: utf-8 -*-
"""
ORMAS KNOWLEDGE BASE - Single Command Server Runner
Reliable on Windows CP1252 consoles, starts FastAPI with Uvicorn.
"""
import sys
import os
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Always set working directory to MODULAR RAG
current_dir = Path(__file__).resolve().parent
os.chdir(current_dir)
sys.path.insert(0, str(current_dir))

import uvicorn
from backend import config

def main():
    print("=" * 60)
    print("  [ORMAS] KNOWLEDGE BASE SERVER STARTING")
    print(f"  Working Directory: {current_dir}")
    print(f"  Chroma DB:         {config.PERSIST_DIRECTORY}")
    print(f"  Frontend URL:      http://{config.HOST}:{config.PORT}")
    print(f"  API Docs:          http://{config.HOST}:{config.PORT}/docs")
    print("=" * 60)
    
    uvicorn.run(
        "backend.app:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
