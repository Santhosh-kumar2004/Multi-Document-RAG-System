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


def store_invoice_records(invoices, filename):
    """Store extracted invoice records in ChromaDB."""

    if not invoices:
        print("No invoice records found.")
        return 0

    documents = []
    metadatas = []
    ids = []

    for invoice in invoices:

        invoice_text = f"""
Invoice Number: {invoice.get('invoice_number')}
Invoice Date: {invoice.get('invoice_date')}
Total Amount: {invoice.get('total_amount')}
Pages: {invoice.get('pages')}
"""

        documents.append(invoice_text)

        metadatas.append({
            "filename": filename,
            "document_type": "scanned_invoice",
            "record_type": "invoice_record",
            "invoice_number": str(
                invoice.get("invoice_number") or ""
            ),
            "invoice_date": str(
                invoice.get("invoice_date") or ""
            ),
            "total_amount": str(
                invoice.get("total_amount") or ""
            ),
            "pages": ",".join(
                str(page)
                for page in invoice.get("pages", [])
            )
        })

        ids.append(str(uuid4()))

    embeddings = model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("Invoice records stored:", len(ids))
    print("Invoice metadata:", metadatas)

    return len(ids)


def search_invoice_records(filename=None):
    """Retrieve structured invoice records."""

    filters = {
        "$and": [
            {"document_type": "scanned_invoice"},
            {"record_type": "invoice_record"}
        ]
    }

    if filename:
        filters["$and"].append({
            "filename": filename
        })

    results = collection.get(
        where=filters,
        include=["documents", "metadatas"]
    )

    invoices = []

    for metadata in results["metadatas"]:

        invoices.append({
            "invoice_number": metadata.get("invoice_number", ""),
            "invoice_date": metadata.get("invoice_date", ""),
            "total_amount": metadata.get("total_amount", ""),
            "pages": metadata.get("pages", "")
        })

    print("Invoice records retrieved:", len(invoices))

    return invoices

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