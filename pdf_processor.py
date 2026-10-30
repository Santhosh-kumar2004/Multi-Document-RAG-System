
import re
from pathlib import Path
from io import BytesIO
from functools import lru_cache

import pymupdf
import pytesseract
from PIL import Image
from transformers import AutoTokenizer


# MiniLM tokenizer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_CHUNK_TOKENS = 220
CHUNK_OVERLAP = 40

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if Path(TESSERACT_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )


def extract_pdf_pages(file_path):
    """
    Extract text from every PDF page.
    Use OCR when a page has little or no readable text.
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



def combine_pages(pages):
    """
    Combine all page text into one continuous document.
    Preserve original page positions.
    """

    full_text = []
    page_ranges = []
    current_position = 0

    for page in pages:

        page_text = page["text"].strip()

        if not page_text:
            continue
        
        if full_text:
            separator = "\n\n"
            full_text.append(separator)
            current_position += len(separator)

        start = current_position

        full_text.append(page_text)

        current_position += len(page_text)

        end = current_position

        page_ranges.append({
            "page": page["page"],
            "start": start,
            "end": end
        })

    return "".join(full_text), page_ranges

def get_token_offsets(text):
    """
    Get token positions using the MiniLM tokenizer.
    """

    tokenizer = get_tokenizer()

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False
    )

    return encoded["offset_mapping"]


def count_tokens(text):
    """
    Count actual tokens using MiniLM tokenizer.
    """

    if not text.strip():
        return 0

    return len(get_token_offsets(text))


def create_chunk(start, end, text, method, page_ranges):
    """
    Create a chunk and identify its source pages.
    """

    source_pages = []

    for item in page_ranges:

        if (
            start < item["end"]
            and end > item["start"]
        ):
            source_pages.append(item["page"])

    token_count = count_tokens(text)

    return {
        "page": source_pages[0] if source_pages else 1,
        "page_end": source_pages[-1] if source_pages else 1,
        "pages": source_pages,
        "text": text.strip(),
        "chunking_method": method,
        "token_count": token_count
    }


def fixed_size_chunk(
    text,
    page_ranges=None,
    chunk_size=MAX_CHUNK_TOKENS,
    overlap=CHUNK_OVERLAP
):
    """
    Split text into fixed token-size chunks.
    """

    if not text.strip():
        return []

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap.")

    offsets = get_token_offsets(text)

    if not offsets:
        return []

    chunks = []
    start_token = 0

    while start_token < len(offsets):

        end_token = min(
            start_token + chunk_size,
            len(offsets)
        )

        char_start = offsets[start_token][0]
        char_end = offsets[end_token - 1][1]

        chunk_text = text[char_start:char_end].strip()

        if chunk_text:
            chunks.append(
                create_chunk(
                    char_start,
                    char_end,
                    chunk_text,
                    "fixed",
                    page_ranges or []
                )
            )

        if end_token == len(offsets):
            break

        start_token = end_token - overlap

    return chunks


def recursive_chunk(
    text,
    page_ranges=None,
    chunk_size=MAX_CHUNK_TOKENS,
    overlap=CHUNK_OVERLAP
):
    """
    Split text using paragraph, sentence and word boundaries.
    Enforce the maximum token size.
    """

    if not text.strip():
        return []

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap.")

    offsets = get_token_offsets(text)

    if not offsets:
        return []

    chunks = []
    start_token = 0

    while start_token < len(offsets):

        maximum_end = min(
            start_token + chunk_size,
            len(offsets)
        )

        end_token = maximum_end

        if maximum_end < len(offsets):

            target_position = offsets[maximum_end - 1][1]

            minimum_position = offsets[
                start_token + int(
                    (maximum_end - start_token) * 0.70
                )
            ][0]

            boundaries = list(
                re.finditer(
                    r"\n\s*\n|\n|(?<=[.!?])\s+|\s+",
                    text[minimum_position:target_position]
                )
            )

            if boundaries:

                boundary = boundaries[-1]

                boundary_position = (
                    minimum_position + boundary.end()
                )

                for index in range(
                    start_token + 1,
                    maximum_end
                ):

                    if offsets[index][0] >= boundary_position:
                        end_token = index
                        break

        char_start = offsets[start_token][0]
        char_end = offsets[end_token - 1][1]

        chunk_text = text[char_start:char_end].strip()

        if chunk_text:
            chunks.append(
                create_chunk(
                    char_start,
                    char_end,
                    chunk_text,
                    "recursive",
                    page_ranges or []
                )
            )

        if end_token == len(offsets):
            break

        start_token = max(
            start_token + 1,
            end_token - overlap
        )

    return chunks


def structure_aware_chunk(
    text,
    page_ranges=None,
    chunk_size=MAX_CHUNK_TOKENS,
    overlap=CHUNK_OVERLAP
):
    """
    Identify document headings and preserve their content.
    Split oversized sections using recursive chunking.
    """

    if not text.strip():
        return []

    heading_pattern = re.compile(
        r"(?im)^\s*(?:"
        r"\d+(?:\.\d+)*[.)]?\s+\S+|"
        r"#{1,6}\s+\S+|"
        r"Leave Type\s*$|"
        r"Annual Entitlement\s*$|"
        r"Approval Required\s*$|"
        r"Step\s+\d+\b|"
        r"Warning\s*:|"
        r"Caution\s*:|"
        r"Note\s*:|"
        r"Error\s*:|"
        r"Problem\s*:|"
        r"Solution\s*:"
        r")"
    )

    matches = list(heading_pattern.finditer(text))

    if not matches:
        return recursive_chunk(
            text,
            page_ranges,
            chunk_size,
            overlap
        )

    sections = []

    if matches[0].start() > 0:
        sections.append((0, matches[0].start()))

    for index, match in enumerate(matches):

        start = match.start()

        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )

        sections.append((start, end))

    chunks = []

    for section_start, section_end in sections:

        section_text = text[
            section_start:section_end
        ].strip()

        if not section_text:
            continue

        section_chunks = recursive_chunk(
            section_text,
            page_ranges=None,
            chunk_size=chunk_size,
            overlap=overlap
        )

        # Recover original document positions.
        for chunk in section_chunks:

            local_start = section_text.find(
                chunk["text"]
            )

            absolute_start = (
                section_start + local_start
            )

            absolute_end = (
                absolute_start + len(chunk["text"])
            )

            chunks.append(
                create_chunk(
                    absolute_start,
                    absolute_end,
                    chunk["text"],
                    "structure_aware",
                    page_ranges or []
                )
            )

    return chunks


def dynamic_chunk(
    text,
    page_ranges=None,
    document_type="employee_policy",
    chunk_size=MAX_CHUNK_TOKENS,
    overlap=CHUNK_OVERLAP
):
    """
    Select a chunking strategy based on document type.
    """

    if document_type in (
        "employee_policy",
        "product_documentation",
        "troubleshooting_guide",
        "scanned_invoice"
    ):

        return structure_aware_chunk(
            text,
            page_ranges,
            chunk_size,
            overlap
        )

    return recursive_chunk(
        text,
        page_ranges,
        chunk_size,
        overlap
    )

def merge_short_chunks(chunks, min_tokens=30, max_tokens=220):
    merged_chunks = []
    pending = None

    for chunk in chunks:

        if pending is None:
            pending = chunk.copy()
            continue

        combined_text = pending["text"] + "\n\n" + chunk["text"]
        combined_tokens = count_tokens(combined_text)

        if (
            pending["token_count"] < min_tokens
            and combined_tokens <= max_tokens
        ):
            pending["text"] = combined_text
            pending["token_count"] = combined_tokens

            pages = sorted(
                set(pending.get("pages", []) + chunk.get("pages", []))
            )

            pending["pages"] = pages
            pending["page"] = pages[0] if pages else None
            pending["page_end"] = pages[-1] if pages else None

        else:
            merged_chunks.append(pending)
            pending = chunk.copy()

    if pending is not None:
        merged_chunks.append(pending)

    return merged_chunks

def process_pdf(
    file_path,
    document_type="employee_policy",
    chunking_method="dynamic"
):
    """
    Extract the complete PDF and create searchable chunks.

    Returns:
        pages: Extracted page information.
        chunks: Processed chunks with source metadata.
    """

    pages = extract_pdf_pages(file_path)

    full_text, page_ranges = combine_pages(pages)

    if not full_text.strip():
        raise ValueError(
            "No readable text found in the PDF."
        )

    if chunking_method == "fixed":

        chunks = fixed_size_chunk(
            full_text,
            page_ranges
        )

    elif chunking_method == "recursive":

        chunks = recursive_chunk(
            full_text,
            page_ranges
        )

    elif chunking_method == "structure_aware":

        chunks = structure_aware_chunk(
            full_text,
            page_ranges
        )

    elif chunking_method == "dynamic":

        chunks = dynamic_chunk(
            full_text,
            page_ranges,
            document_type
        )

    else:
        raise ValueError(
            "Invalid chunking method. Choose "
            "fixed, recursive, structure_aware, or dynamic."
        )

    chunks = merge_short_chunks(chunks)

    for chunk in chunks:
        chunk["document_type"] = document_type
        chunk["chunking_method"] = chunking_method

    return pages, chunks
