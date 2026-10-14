import re

def parse_invoice(pages):
    chunks = []
    full_text = "\n".join(
        page["text"] for page in pages
    )

    fields = {
        "invoice_number": r"invoice\s*(?:no|number|#)?\s*[:#-]?\s*([A-Z0-9/-]+)",
        "invoice_date": r"(?:invoice date|date)\s*[:#-]?\s*([0-9]{1,4}[/-][0-9]{1,2}[/-][0-9]{1,4})",
        "total_amount": r"(?:grand total|total amount|total)\s*[:₹$ ]*\s*([0-9,]+(?:\.[0-9]{1,2})?)"
    }

    extracted_fields = {}

    for field_name, pattern in fields.items():
        match = re.search(pattern, full_text, re.IGNORECASE)

        extracted_fields[field_name] = (
            match.group(1) if match else None
        )

    for page in pages:
        if page["text"].strip():
            chunks.append({
                "document_type": "scanned_invoice",
                "page": page["page"],
                "text": page["text"],
                "invoice_fields": extracted_fields
            })

    return chunks, extracted_fields
