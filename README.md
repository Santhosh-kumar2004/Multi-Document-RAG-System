# Multi-Document RAG System

## About the Project

This is a simple **Multi-Document Retrieval-Augmented Generation (RAG) System** built using Python and FastAPI.

It allows users to upload PDF documents and ask questions about their content. The system searches the uploaded documents and uses an AI model to generate answers based on the retrieved information.

## Document Types

The project supports four types of PDF documents:

- **Employee Policy** – Employee leave rules and company policies.
- **Product Documentation** – Product details and technical information.
- **Scanned Invoice** – Invoice details extracted using OCR.
- **Troubleshooting Guide** – Technical problems and their solutions.

## Technologies Used

- Python
- FastAPI
- PyMuPDF
- Tesseract OCR
- Sentence Transformers
- ChromaDB
- Ollama (Llama 3.2)

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Santhosh-kumar2004/Multi-Document-RAG-System.git
```

### 2. Open the Project

```bash
cd Multi-Document-RAG-System
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install the Requirements

```bash
pip install -r requirements.txt
```

### 5. Set Up Ollama

Install Ollama and start it. Then download the model:

```bash
ollama pull llama3.2
```

Make sure Ollama is running before asking questions.

### 6. Run the Application

```bash
uvicorn main:app --reload
```

### 7. Open Swagger UI

Open this URL in your browser:

http://127.0.0.1:8000/docs

## How to Use

1. Open the `/upload` API.
2. Click **Try it out**.
3. Select a PDF file.
4. Choose the correct document type.
5. Click **Execute** to upload the document.
6. Open the `/ask` API.
7. Enter your question and select the document type if needed.
8. Click **Execute** to get the answer and its source information.

## Example Question

**Question:** How many casual leave days are allowed per year?

**Answer:** The system retrieves the relevant information from the uploaded employee policy document and generates an answer.

## Project Goal

The main goal of this project is to make it easier to find information from different PDF documents without reading the entire document manually.

