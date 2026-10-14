from pdf_processor import load_pdf, create_chunks
from vector_store import store_chunks, search_chunks
from llm import generate_answer

def process_pdf(pdf_path, document_type="employee_policy"):
    text = load_pdf(pdf_path, document_type)
    chunks = create_chunks(text, document_type)

    if not chunks:
        raise ValueError(
            "No readable text or chunks found in PDF."
        )
    total_chunks = store_chunks(chunks)

    return total_chunks

def ask_question(question):
    documents = search_chunks(question)

    if not documents:
        return {
            "question": question,
            "answer": "I don't know based on the provided document."
        }
    answer = generate_answer(
        question,
        documents
    )

    return {
        "question": question,
        "answer": answer
    }
