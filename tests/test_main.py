from io import BytesIO


def test_app_starts(client):
    assert client is not None


def test_upload_invalid_file(client):
    response = client.post(
        "/upload",
        files={
            "file": (
                "test.txt",
                BytesIO(b"Sample text"),
                "text/plain"
            )
        }
    )

    assert response.status_code in [400, 422]


def test_upload_pdf(client, mock_rag_functions):

    response = client.post(
        "/upload",
        data={
            "document_type": "employee_policy"
        },
        files={
            "file": (
                "test.pdf",
                BytesIO(b"%PDF-1.4 sample"),
                "application/pdf"
            )
        }
    )

    print(response.json())

    assert response.status_code == 200

def test_process_pdf_called(client, mock_rag_functions):

    client.post(
        "/upload",
        data={
            "document_type": "employee_policy"
        },
        files={
            "file": (
                "test.pdf",
                BytesIO(b"%PDF-1.4 sample"),
                "application/pdf"
            )
        }
    )

    mock_rag_functions["process_pdf"].assert_called_once()


def test_search_document(client, mock_rag_functions):

    response = client.post(
        "/ask",
        json={
            "question": "What is machine learning?"
        }
    )

    assert response.status_code == 200


def test_generate_answer_called(client, mock_rag_functions):

    client.post(
        "/ask",
        json={
            "question": "Explain artificial intelligence"
        }
    )

    mock_rag_functions["generate_answer"].assert_called_once()


def test_search_chunks_called(client, mock_rag_functions):

    client.post(
        "/ask",
        json={
            "question": "What is machine learning?"
        }
    )

    mock_rag_functions["search_chunks"].assert_called_once()

def test_empty_question(client):
    response = client.post(
        "/ask",
        json={"question": ""}
    )

    assert response.status_code == 422