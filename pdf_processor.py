
import re
from pathlib import Path
from io import BytesIO
import pymupdf
import pytesseract
from PIL import Image

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if Path(TESSERACT_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_pdf_pages(file_path):
    """
    Extract text from each PDF page.
    Use OCR when a page has little or no extractable text.
    """

    pages = []

    pdf = pymupdf.open(file_path)

    try:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if len(text) < 30:
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False
                )

                image = Image.open(
                    BytesIO(pix.tobytes("png"))
                )

                try:
                    text = pytesseract.image_to_string(
                        image,
                        lang="eng"
                    ).strip()

                except pytesseract.TesseractNotFoundError:
                    raise RuntimeError(
                        "Tesseract OCR is not installed. "
                        "Install it or check TESSERACT_PATH."
                    )

            pages.append({
                "page": page_number,
                "text": text
            })

    finally:
        pdf.close()

    return pages

def fixed_size_chunk(text, page, chunk_size=1000, overlap=150):
    """
    Split text into chunks using character length.
    Overlap preserves some context between chunks.
    """

    chunks = []
    text = text.strip()

    if not text:
        return chunks

    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "page": page,
                "text": chunk_text,
                "chunking_method": "fixed"
            })

        if end == len(text):
            break

        start = end - overlap

    return chunks

def recursive_chunk(text, page, chunk_size=1000, overlap=150):
    """
    Prefer paragraph and sentence boundaries.
    Fall back to smaller text boundaries when needed.
    """

    chunks = []
    text = text.strip()

    if not text:
        return chunks

    paragraphs = re.split(r"\n\s*\n", text)

    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue
        if len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append({
                    "page": page,
                    "text": current_chunk.strip(),
                    "chunking_method": "recursive"
                })
                current_chunk = ""

            sentences = re.split(
                r"(?<=[.!?])\s+",
                paragraph
            )

            for sentence in sentences:
                if len(sentence) > chunk_size:
                    chunks.extend(
                        fixed_size_chunk(
                            sentence,
                            page,
                            chunk_size,
                            overlap
                        )
                    )
                    continue

                if (
                    current_chunk
                    and len(current_chunk) + len(sentence) + 1
                    > chunk_size
                ):
                    chunks.append({
                        "page": page,
                        "text": current_chunk.strip(),
                        "chunking_method": "recursive"
                    })

                    current_chunk = current_chunk[-overlap:]

                current_chunk += " " + sentence

            continue

        if (
            current_chunk
            and len(current_chunk) + len(paragraph) + 2
            > chunk_size
        ):
            chunks.append({
                "page": page,
                "text": current_chunk.strip(),
                "chunking_method": "recursive"
            })

            current_chunk = current_chunk[-overlap:]

        current_chunk += "\n\n" + paragraph

    if current_chunk.strip():
        chunks.append({
            "page": page,
            "text": current_chunk.strip(),
            "chunking_method": "recursive"
        })

    return chunks

def structure_aware_chunk(text, page, chunk_size=1500):
    """
    Preserve numbered headings and related content.
    Useful for employee policies and technical documents.
    """

    chunks = []
    text = text.strip()

    if not text:
        return chunks

    sections = re.split(
        r"(?m)(?=^\s*(?:"
        r"\d+(?:\.\d+)*[.)]?\s+\S+|"
        r"Leave Type\s*$|"
        r"Annual Entitlement\s*$|"
        r"Approval Required\s*$|"
        r"Warning\s*:|"
        r"Caution\s*:|"
        r"Note\s*:|"
        r"Error\s*:"
        r"))",
        text,
        flags=re.IGNORECASE
    )

    current_chunk = ""

    for section in sections:
        section = section.strip()

        if not section:
            continue
        if len(section) > chunk_size:
            if current_chunk.strip():
                chunks.append({
                    "page": page,
                    "text": current_chunk.strip(),
                    "chunking_method": "structure_aware"
                })
                current_chunk = ""

            chunks.extend(
                recursive_chunk(
                    section,
                    page,
                    chunk_size=chunk_size,
                    overlap=100
                )
            )
            continue

        if (
            current_chunk
            and len(current_chunk) + len(section) + 2
            > chunk_size
        ):
            chunks.append({
                "page": page,
                "text": current_chunk.strip(),
                "chunking_method": "structure_aware"
            })
            current_chunk = ""

        current_chunk += "\n\n" + section

    if current_chunk.strip():
        chunks.append({
            "page": page,
            "text": current_chunk.strip(),
            "chunking_method": "structure_aware"
        })

    return chunks

def dynamic_chunk(text, page, document_type="employee_policy"):
    """
    Select a chunking strategy based on document type
    and text characteristics.
    """

    text = text.strip()

    if not text:
        return []

    if document_type in (
        "employee_policy",
        "product_documentation",
        "troubleshooting_guide"
    ):
        return structure_aware_chunk(
            text,
            page,
            chunk_size=1500
        )

    if document_type == "scanned_invoice":
        return structure_aware_chunk(
            text,
            page,
            chunk_size=1200
        )
    if len(text) <= 1500:
        return [{
            "page": page,
            "text": text,
            "chunking_method": "dynamic"
        }]

    return recursive_chunk(
        text,
        page,
        chunk_size=1000,
        overlap=150
    )

def process_pdf(
    file_path,
    document_type="employee_policy",
    chunking_method="dynamic"
):
    """
    Extract PDF text and create chunks.

    Returns:
        pages: extracted text by page
        chunks: chunk dictionaries ready for vector storage
    """

    pages = extract_pdf_pages(file_path)
    chunks = []

    for page in pages:
        text = page["text"]

        if chunking_method == "fixed":
            page_chunks = fixed_size_chunk(
                text,
                page["page"]
            )

        elif chunking_method == "recursive":
            page_chunks = recursive_chunk(
                text,
                page["page"]
            )

        elif chunking_method == "structure_aware":
            page_chunks = structure_aware_chunk(
                text,
                page["page"]
            )

        elif chunking_method == "dynamic":
            page_chunks = dynamic_chunk(
                text,
                page["page"],
                document_type
            )

        else:
            raise ValueError(
                "Invalid chunking method. Choose fixed, "
                "recursive, structure_aware, or dynamic."
            )

        for chunk in page_chunks:
            chunk["document_type"] = document_type

        chunks.extend(page_chunks)

    return pages, chunks
