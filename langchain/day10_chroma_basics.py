
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

# --------------------------------------------------
# 1. Persistent database location
# --------------------------------------------------
DB_PATH = Path("./storage/demo_chroma")
DB_PATH.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Connect to persistent Chroma
# --------------------------------------------------
client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

# --------------------------------------------------
# 3. Configure the embedding model
# Chroma will embed documents and queries automatically.
# --------------------------------------------------
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# --------------------------------------------------
# 4. Create or reopen the collection
# --------------------------------------------------
collection = client.get_or_create_collection(
    name="ormas_documents",
    embedding_function=embedding_function,
    metadata={"description": "ORMAS RAG learning dataset"},
)

# --------------------------------------------------
# 5. Prepare illustrative documents and metadata
# --------------------------------------------------
records = [
    {
        "id": "pe-001",
        "document": (
            "Producer Enterprises support rural enterprise "
            "development and livelihood opportunities."
        ),
        "metadata": {
            "district": "Kandhamal",
            "scheme": "PE",
            "financial_year": "2025-26",
        },
    },
    {
        "id": "pe-002",
        "document": (
            "Equipment and machinery can support production "
            "capacity in a Producer Enterprise."
        ),
        "metadata": {
            "district": "Kandhamal",
            "scheme": "PE",
            "financial_year": "2025-26",
        },
    },
    {
        "id": "fpo-001",
        "document": (
            "Farmer Producer Organisations can aggregate "
            "produce and improve market access for farmers."
        ),
        "metadata": {
            "district": "Ganjam",
            "scheme": "FPO",
            "financial_year": "2024-25",
        },
    },
    {
        "id": "marketing-001",
        "document": (
            "Marketing initiatives connect rural producers "
            "with institutional buyers and retail markets."
        ),
        "metadata": {
            "district": "Ganjam",
            "scheme": "Marketing",
            "financial_year": "2025-26",
        },
    },
]

# --------------------------------------------------
# 6. Insert or update records by ID
# --------------------------------------------------
collection.upsert(
    ids=[item["id"] for item in records],
    documents=[item["document"] for item in records],
    metadatas=[item["metadata"] for item in records],
)

print("Stored records:", collection.count())

# --------------------------------------------------
# 7. Semantic similarity search
# Chroma embeds the query using the same model.
# --------------------------------------------------
query = "How can rural producers improve their market access?"

results = collection.query(
    query_texts=[query],
    n_results=3,
)

print("\nSEMANTIC SEARCH RESULTS")

for document_id, document, metadata, distance in zip(
    results["ids"][0],
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
):
    print(f"\nID: {document_id}")
    print(f"Document: {document}")
    print(f"Metadata: {metadata}")
    print(f"Distance: {distance:.4f}")

# --------------------------------------------------
# 8. Metadata-filtered retrieval
# --------------------------------------------------
filtered_results = collection.query(
    query_texts=["production equipment"],
    where={
        "$and": [
            {"district": {"$eq": "Kandhamal"}},
            {"scheme": {"$eq": "PE"}},
        ]
    },
    n_results=3,
)

print("\nKANDHAMAL PE RESULTS")

for document_id, document, metadata in zip(
    filtered_results["ids"][0],
    filtered_results["documents"][0],
    filtered_results["metadatas"][0],
):
    print(f"\nID: {document_id}")
    print(f"Document: {document}")
    print(f"Metadata: {metadata}")

# --------------------------------------------------
# 9. Verify persistence
# Reopen the collection through the persistent client.
# --------------------------------------------------
reopened_collection = client.get_collection(
    name="ormas_documents",
    embedding_function=embedding_function,
)

print("\nRecords after reopening:", reopened_collection.count())


# 1. Retrieve records for one district and scheme
results = collection.get(
    where={
        "$and": [
            {"district": {"$eq": "Kandhamal"}},
            {"scheme": {"$eq": "PE"}},
        ]
    },
    include=["documents", "metadatas"],
)

print("Matching records:", len(results["ids"]))

# 2. Update an existing record
collection.update(
    ids=["pe-001"],
    documents=[
        "Producer Enterprises support rural livelihoods "
        "through enterprise development and market access."
    ],
    metadatas=[{
        "district": "Kandhamal",
        "scheme": "PE",
        "financial_year": "2025-26",
        "status": "updated",
    }],
)

# 3. Verify the update
updated = collection.get(
    ids=["pe-001"],
    include=["documents", "metadatas"],
)
print("Updated document:", updated["documents"][0])
print("Updated metadata:", updated["metadatas"][0])

# 4. Delete a record
collection.delete(ids=["marketing-001"])

# 5. Verify deletion
print("Records after deletion:", collection.count())
print(
    "Deleted ID lookup:",
    collection.get(ids=["marketing-001"])["ids"],
)

#
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    collection_name="ormas_documents",
    embedding_function=embeddings,
    persist_directory="./ormas_chroma",
    collection_configuration={
        "hnsw": {
            "space": "cosine",
            "ef_construction": 200,
            "ef_search": 100,
            "max_neighbors": 32,
        }
    },
)

results = vectorstore.similarity_search(
    "equipment and machinery allocation",
    k=5,
)

print(results)