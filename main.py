from pathlib import Path
import re
from enum import Enum
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from parsers.invoice_parser import parse_invoice
from pdf_processor import process_pdf
from vector_store import (
    store_chunks,
    search_chunks,
    store_invoice_records,
    search_invoice_records
)
from llm import generate_answer
from schemas import QuestionRequest

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
        pages, chunks = process_pdf(
            file_path=str(save_path),
            document_type=document_type.value,
            chunking_method="dynamic"
        )

        extra_data = {}
        invoices = []

        if document_type == DocumentType.scanned_invoice:
            _, invoices = parse_invoice(pages)
            extra_data = {"invoices": invoices}

        stored_count = store_chunks(
            chunks=chunks,
            filename=safe_filename,
            document_type=document_type.value
        )

        # Store structured invoice records
        if document_type == DocumentType.scanned_invoice:
            store_invoice_records(
                invoices=invoices,
                filename=safe_filename
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
        top_k=5
    )

    sources = []

    for match in matches:
        metadata = match["metadata"]

        sources.append({
            "filename": metadata["filename"],
            "document_type": metadata["document_type"],
            "page": metadata["page"],
            "page_end": metadata["page_end"],
            "pages": metadata["pages"],
            "text": match["text"]
        })

    context = "\n\n".join(
        source["text"] for source in sources
    )

    if request.document_type == DocumentType.scanned_invoice:

        invoice_match = re.search(
            r"INV[-/]?\d{4}[-/]\d+",
            question,
            re.IGNORECASE
        )

        all_invoice_records = search_invoice_records()

        if invoice_match:
            invoice_number = invoice_match.group(0).upper()

            invoice_records = [
                invoice for invoice in all_invoice_records
                if invoice["invoice_number"].upper() == invoice_number
            ]
        else:
            invoice_records = all_invoice_records

        if not invoice_records:
            return {
                "question": question,
                "answer": "No matching invoice records found.",
                "sources": []
            }

        invoice_context = "\n\n".join(
            f"""
    Invoice Number: {invoice["invoice_number"]}
    Invoice Date: {invoice["invoice_date"]}
    Total Amount: {invoice["total_amount"]}
    Pages: {invoice["pages"]}
    """
            for invoice in invoice_records
        )

        context = f"""
        IMPORTANT: Use only the structured invoice records below
        for invoice number and total amount questions.
        Do not use amounts from other invoice chunks.

        STRUCTURED INVOICE RECORDS:
        {invoice_context}
        """

    elif not matches:
        return {
            "question": question,
            "answer": "I could not find the answer in the uploaded documents.",
            "sources": []
        }

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