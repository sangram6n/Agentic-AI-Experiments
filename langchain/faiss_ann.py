import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# =====================================================
# 1. CREATE AN ILLUSTRATIVE DOCUMENT COLLECTION
# =====================================================

topics = {
    "equipment": (
        "Equipment and machinery allocation supports productive "
        "assets and enterprise operations."
    ),
    "training": (
        "Training allocation supports production practices, "
        "quality management and market readiness."
    ),
    "software": (
        "Software and IT allocation supports digital operations "
        "and information management."
    ),
    "marketing": (
        "Marketing support covers branding, packaging and "
        "market linkage activities."
    ),
    "management": (
        "Management support covers administrative and operational "
        "requirements of the enterprise."
    ),
}

districts = [
    "Kandhamal", "Puri", "Ganjam", "Koraput",
    "Mayurbhanj", "Sambalpur", "Balasore", "Kalahandi",
]

years = list(range(2018, 2026))

documents = []
metadata = []

for topic, description in topics.items():
    for district in districts:
        for year in years:
            documents.append(
                f"Scheme: Producer Enterprise. "
                f"District: {district}. Year: {year}. "
                f"Section: {topic}. {description}"
            )

            metadata.append({
                "scheme": "PE",
                "district": district,
                "year": year,
                "section": topic,
            })

print("Total documents:", len(documents))
# =====================================================
# 2. CREATE EMBEDDINGS
# =====================================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

vectors = model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

vectors = np.ascontiguousarray(vectors, dtype=np.float32)

n, dimension = vectors.shape

print("Vector count:", n)
print("Dimensions:", dimension)


# =====================================================
# 3. EXACT SEARCH: FLAT INDEX
# Normalized vectors + inner product = cosine similarity.
# =====================================================

exact_index = faiss.IndexFlatIP(dimension)
exact_index.add(vectors)


# =====================================================
# 4. APPROXIMATE SEARCH: IVF-FLAT
# =====================================================

nlist = 8

quantizer = faiss.IndexFlatIP(dimension)

ivf_index = faiss.IndexIVFFlat(
    quantizer,
    dimension,
    nlist,
    faiss.METRIC_INNER_PRODUCT,
)

ivf_index.train(vectors)
ivf_index.add(vectors)

# Search a subset of the clusters.
ivf_index.nprobe = 2


# =====================================================
# 5. APPROXIMATE SEARCH: HNSW-FLAT
# =====================================================

hnsw_index = faiss.IndexHNSWFlat(
    dimension,
    32,
    faiss.METRIC_INNER_PRODUCT,
)

hnsw_index.add(vectors)

# Higher efSearch generally explores more candidates.
hnsw_index.hnsw.efSearch = 40


# =====================================================
# 6. SEARCH AND COMPARE RECALL
# =====================================================

def search(index, query_vector, k=5):
    scores, ids = index.search(query_vector, k)

    results = []

    for score, doc_id in zip(scores[0], ids[0]):
        if doc_id >= 0:
            results.append((int(doc_id), float(score)))

    return results


def recall_at_k(approximate_ids, exact_ids):
    if not exact_ids:
        return 0.0

    return len(set(approximate_ids) & set(exact_ids)) / len(exact_ids)


def show_results(label, results):
    print(f"\n--- {label} ---")

    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"ID: {doc_id}")
        print(f"Score: {score:.4f}")
        print("Metadata:", metadata[doc_id])
        print("Document:", documents[doc_id])


# Embed the query using the same embedding model.
query = (
    "What assistance is allocated to machinery "
    "in the Producer Enterprise scheme?"
)

query_vector = model.encode(
    [query],
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype(np.float32)

query_vector = np.ascontiguousarray(query_vector)

k = 5
hnsw_index.search(query_vector,3)
exact_results = search(exact_index, query_vector, k)
ivf_results = search(ivf_index, query_vector, k)
hnsw_results = search(hnsw_index, query_vector, k)

show_results("EXACT FLAT", exact_results)
show_results("IVF-FLAT", ivf_results)
show_results("HNSW-FLAT", hnsw_results)

exact_ids = [doc_id for doc_id, score in exact_results]

print("\n--- RECALL COMPARISON ---")
print(
    "IVF Recall@5:",
    recall_at_k(
        [doc_id for doc_id, score in ivf_results],
        exact_ids,
    ),
)
print(
    "HNSW Recall@5:",
    recall_at_k(
        [doc_id for doc_id, score in hnsw_results],
        exact_ids,
    ),
)


# =====================================================
# 7. TUNE IVF SEARCH BREADTH
# =====================================================

print("\n--- IVF nprobe EXPERIMENT ---")

for nprobe in [1, 2, 4, nlist]:
    ivf_index.nprobe = nprobe
    results = search(ivf_index, query_vector, k)

    recall = recall_at_k(
        [doc_id for doc_id, score in results],
        exact_ids,
    )

    print(
        f"nprobe={nprobe:2d} | "
        f"Recall@{k}={recall:.2f}"
    )

import matplotlib.pyplot as plt
