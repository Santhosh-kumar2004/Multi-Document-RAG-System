from pathlib import Path
from uuid import uuid4

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

DB_DIR.mkdir(parents=True, exist_ok=True)

client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

collection = client.get_or_create_collection(
    name="rag_documents"
)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def store_chunks(chunks, filename, document_type):
    """Convert chunks to embeddings and save them."""

    valid_chunks = [
        chunk for chunk in chunks
        if chunk.get("text", "").strip()
    ]

    if not valid_chunks:
        raise ValueError(
            "No text found to store in the database."
        )

    texts = [
        chunk["text"] for chunk in valid_chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    ids = [
        str(uuid4()) for _ in valid_chunks
    ]

    metadatas = []

    for chunk in valid_chunks:
        metadatas.append({
            "filename": filename,
            "document_type": document_type,
            "page": int(chunk.get("page", 1)),
            "page_end": int(chunk.get("page_end", chunk.get("page", 1))),
            "pages": ",".join(
                str(page) for page in chunk.get("pages", [])
            ),
            "chunking_method": str(
                chunk.get("chunking_method", "dynamic")
            ),
            "token_count": int(
                chunk.get("token_count", 0)
            )
        })

    collection.delete(
        where={
            "$and": [
                {"filename": filename},
                {"document_type": document_type}
            ]
        }
    )

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(ids)


def search_chunks(question, document_type=None, top_k=3):
    """Find relevant chunks and remove short or duplicate results."""

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    filters = None

    if document_type:
        filters = {
            "document_type": document_type
        }

    total_chunks = collection.count()

    if total_chunks == 0:
        return []
    
    candidate_count = min(
        max(top_k * 3, top_k),
        total_chunks
    )

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=candidate_count,
        where=filters
    )

    candidates = []
    seen_texts = set()

    if results["documents"]:
        for i, text in enumerate(results["documents"][0]):
            normalized_text = " ".join(text.lower().split())

            if normalized_text in seen_texts:
                continue

            seen_texts.add(normalized_text)

            metadata = results["metadatas"][0][i]
            token_count = int(metadata.get("token_count", 0))

            candidates.append({
                "text": text,
                "metadata": metadata,
                "distance": results["distances"][0][i],
                "token_count": token_count
            })
    meaningful_chunks = [
        chunk for chunk in candidates
        if chunk["token_count"] >= 25
    ]

    if not meaningful_chunks:
        meaningful_chunks = candidates

    return meaningful_chunks[:top_k]