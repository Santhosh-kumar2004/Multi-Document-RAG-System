
from pathlib import Path
from enum import Enum
from parsers.document_parser import extract_pdf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from parsers.policy_parser import parse_policy
from parsers.product_parser import parse_product_documentation
from parsers.invoice_parser import parse_invoice
from parsers.troubleshooting_parser import (parse_troubleshooting_guide)
from vector_store import store_chunks, search_chunks
from schemas import QuestionRequest
from llm import generate_answer
from vector_store import search_chunks
from schemas import QuestionRequest
from fastapi import HTTPException

app = FastAPI(
    title="Smart Multi-Document RAG System",
    version="1.0.0"
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class DocumentType(str, Enum):
    employee_policy = "employee_policy"
    product_documentation = "product_documentation"
    scanned_invoice = "scanned_invoice"
    troubleshooting_guide = "troubleshooting_guide"


@app.get("/")
def home():
    return {
        "message": "RAG API is running",
        "supported_documents": [item.value for item in DocumentType]
    }

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    document_type: DocumentType = Form(...)
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    content = await file.read()

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file."
        )

    safe_filename = Path(file.filename).name
    save_path = UPLOAD_DIR / (
        f"{document_type.value}_{safe_filename}"
    )

    save_path.write_bytes(content)

    
    try:
        pages = extract_pdf(save_path)
        if document_type == DocumentType.employee_policy:
            chunks = parse_policy(pages)
            extra_data = {}

        elif document_type == DocumentType.product_documentation:
            chunks = parse_product_documentation(pages)
            extra_data = {}

        elif document_type == DocumentType.scanned_invoice:
            chunks, invoice_fields = parse_invoice(pages)
            extra_data = {
                "invoice_fields": invoice_fields
            }

        elif document_type == DocumentType.troubleshooting_guide:
            chunks = parse_troubleshooting_guide(pages)
            extra_data = {}

        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported document type."
            )
        stored_count = store_chunks(
            chunks=chunks,
            filename=safe_filename,
            document_type=document_type.value
        )
        return {
            "message": "Document processed successfully",
            "filename": safe_filename,
            "document_type": document_type.value,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "stored_chunks": stored_count,
            "chunks": chunks,
            **extra_data
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {error}"
        )

@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )
    matches = search_chunks(
        question=question,
        document_type=request.document_type,
        top_k=request.top_k
    )

    if not matches:
        return {
            "question": question,
            "answer": "I could not find the answer in the uploaded documents.",
            "sources": []
        }
    sources = []

    for match in matches:
        metadata = match["metadata"]

        sources.append({
            "filename": metadata["filename"],
            "document_type": metadata["document_type"],
            "page": metadata["page"],
            "text": match["text"]
        })

    context = "\n\n".join(
        source["text"] for source in sources
    )

    try:
        answer = generate_answer(
            question=question,
            context=context
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=502,
            detail=str(error)
        )
    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }



