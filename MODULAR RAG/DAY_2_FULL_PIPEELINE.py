# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 20:18:17 2026

@author: Sangram Mishra
"""
# day2_full_pipeline.py
import logging
from ingestion_pipeline import load_directory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
# Show pipeline progress (INFO and above)


import os
from dotenv import load_dotenv

load_dotenv()
#os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

print("the enviroments variable are loaded")
logging.basicConfig(level=logging.INFO, format="%(message)s")

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── STEP 1: Load — generator, O(1) memory ──
doc_generator = load_directory(".")

# ── STEP 2: Chunk ──
# Consume generator here — splitter needs list
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(list(doc_generator))
print(f"✅ {len(chunks)} chunks ready")

# ── STEP 3: Embed + Store ──
embeddings = OllamaEmbeddings(model="nomic-embed-text")
persist_dir = "./chroma_db_day2"

if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
    print(f"📂 Loading existing ChromaDB from {persist_dir}...")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    count = vectorstore._collection.count()
    print(f"✅ Vectorstore currently has {count} indexed vectors")
    
    # Check for unindexed files
    existing_sources = set()
    try:
        all_metas = vectorstore.get()["metadatas"]
        existing_sources = {m.get("source") for m in all_metas if m and "source" in m}
    except Exception:
        pass

    new_chunks = [c for c in chunks if c.metadata.get("source") not in existing_sources]
    if new_chunks:
        print(f"⚡ Found {len(new_chunks)} new chunks from unindexed files. Adding to ChromaDB...")
        vectorstore.add_documents(new_chunks)
        print(f"✅ Vectorstore now has {vectorstore._collection.count()} total vectors")
    else:
        print("✅ All chunks are already up-to-date in ChromaDB.")
else:
    print(f"⚡ Creating fresh ChromaDB with {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"✅ Stored {len(chunks)} vectors in ChromaDB")

# ── STEP 4: Retriever ──
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ── STEP 5: Chain ──
llm = init_chat_model(
    model = "openai/gpt-oss-20b",
    model_provider= "groq"
)
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
- If the user is just greeting you (e.g., "hi", "hello", "hey"), greet them back warmly and offer to help them with the documents.
- For all questions about the content, answer using ONLY the context below.
- If the question cannot be answered from the context, say: "Not found in documents."
Context:
{context}
Question: {question}
""")


def format_docs(docs: list) -> str:
    """Build context string with rich citations from metadata."""
    return "\n\n".join(
        (
            f"[{d.metadata.get('filename', 'unknown')} | "
            f"Page {d.metadata.get('page', '')} | "
            f"Chapter: {d.metadata.get('chapter', '')} | "
            f"Section: {d.metadata.get('section', d.metadata.get('slide_title', ''))}]\n"
            f"{d.page_content}"
        )
        for d in docs
    )

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── STEP 6: Interactive loop ──
while True:
    q = input("\n❓ Question (or 'quit'): ").strip()
    if q.lower() == "quit":
        break
    if q:
        print(f"\n💬 {rag_chain.invoke(q)}")