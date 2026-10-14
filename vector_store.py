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

    ids = [str(uuid4()) for _ in texts]

    metadatas = [
        {
            "filename": filename,
            "document_type": document_type,
            "page": int(chunk["page"])
        }
        for chunk in valid_chunks
    ]
    collection.delete(where={"filename": filename})
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(ids)
def search_chunks(question, document_type=None, top_k=3):
    """Find chunks most relevant to a question."""
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    filters = None

    if document_type:
        filters = {
            "document_type": document_type
        }

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, max(collection.count(), 1)),
        where=filters
    )
    matches = []
    if results["documents"]:
        for i, text in enumerate(results["documents"][0]):
            matches.append({
                "text": text,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
    return matches
