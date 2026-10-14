
from pathlib import Path
from io import BytesIO

import pymupdf
import pytesseract
from PIL import Image

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if Path(TESSERACT_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def extract_pdf(file_path):
    results = []

    try:
        pdf = pymupdf.open(file_path)
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if len(text) < 30:
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False
                )
                image_bytes = pix.tobytes("png")
                image = Image.open(BytesIO(image_bytes))

                try:
                    text = pytesseract.image_to_string(
                        image,
                        lang="eng"
                    ).strip()
                except pytesseract.TesseractNotFoundError:
                    raise RuntimeError(
                        "Tesseract OCR is not installed "
                        "or its path is incorrect."
                    )

            results.append({
                "page": page_number,
                "text": text
            })

        pdf.close()
        return results

    except Exception:
        raise
