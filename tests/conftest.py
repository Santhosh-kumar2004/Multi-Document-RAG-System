import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import main


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def mock_rag_functions():

    with patch("main.process_pdf") as process_pdf, \
         patch("main.store_chunks") as store_chunks, \
         patch("main.store_invoice_records") as store_invoice_records, \
         patch("main.search_chunks") as search_chunks, \
         patch("main.search_invoice_records") as search_invoice_records, \
         patch("main.generate_answer") as generate_answer:

        process_pdf.return_value = (
            [{"page": 1, "text": "Sample document content"}],
            [
                {
                    "text": "Sample document content",
                    "metadata": {
                        "filename": "test.pdf",
                        "document_type": "employee_policy",
                        "page": 1,
                        "page_end": 1,
                        "pages": [1]
                    }
                }
            ]
        )

        search_chunks.return_value = [
            {
                "text": "Sample document content",
                "metadata": {
                    "filename": "test.pdf",
                    "document_type": "general",
                    "page": 1,
                    "page_end": 1,
                    "pages": [1]
                }
            }
        ]

        search_invoice_records.return_value = []

        generate_answer.return_value = "This is a test answer."

        yield {
            "process_pdf": process_pdf,
            "store_chunks": store_chunks,
            "store_invoice_records": store_invoice_records,
            "search_chunks": search_chunks,
            "search_invoice_records": search_invoice_records,
            "generate_answer": generate_answer
        }