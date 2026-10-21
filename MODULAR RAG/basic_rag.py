# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:15:06 2026

@author: Sangram Mishra
"""
# day1_rag_basics.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# ─────────────────────────────────────────
# STEP 1: LOAD — Bring documents into memory
# ─────────────────────────────────────────
loader = PyPDFLoader("attn.pdf")   # swap with any PDF
docs = loader.load()
print(f"✅ Loaded {len(docs)} pages")

# ─────────────────────────────────────────
# STEP 2: CHUNK — Split into searchable pieces
# ─────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # characters per chunk
    chunk_overlap=50,     # overlap so context isn't cut mid-sentence
)
chunks = splitter.split_documents(docs)
print(f"✅ Created {len(chunks)} chunks from {len(docs)} pages")

# ─────────────────────────────────────────
# STEP 3: EMBED + STORE — Convert to vectors, save to DB
# ─────────────────────────────────────────
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"   # saved to disk — no re-indexing next run
)
print(f"✅ Stored {len(chunks)} vectors in ChromaDB")

# ─────────────────────────────────────────
# STEP 4: RETRIEVE — Search for relevant chunks
# ─────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}   # return top 3 most relevant chunks
)

# ─────────────────────────────────────────
# STEP 5: GENERATE — LLM answers using retrieved context
# ─────────────────────────────────────────
llm = init_chat_model(
    model = "openai/gpt-oss-20b",
    model_provider= "groq"
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
""")

# Wire everything together as a chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ─────────────────────────────────────────
# STEP 6: ASK
# ─────────────────────────────────────────
question = "What is the main topic of this document?"
answer = rag_chain.invoke(question)
print(f"\n❓ Question: {question}")
print(f"💬 Answer: {answer}")

