import re

def parse_invoice(pages):
    invoices = []
    chunks = []

    invoice_pattern = re.compile(
        r"invoice\s*(?:no|number|#)\s*[:#-]?\s*([A-Z0-9/-]+)",
        re.IGNORECASE
    )

    date_pattern = re.compile(
        r"(?:invoice date|date)\s*[:#-]?\s*"
        r"([0-9]{1,2}[-/][A-Za-z]{3}[-/][0-9]{4}|"
        r"[0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
        re.IGNORECASE
    )

    amount_pattern = re.compile(
        r"(?:grand total|total\s*\(INR\)|total amount)"
        r"\s*[:₹$ ]*\s*([0-9,]+(?:\.[0-9]{1,2})?)",
        re.IGNORECASE
    )

    current_invoice = None

    for page in pages:
        page_text = page["text"].strip()

        if not page_text:
            continue

        invoice_match = invoice_pattern.search(page_text)

        # Start a new invoice when another invoice number is found.
        if invoice_match:
            if current_invoice:
                invoices.append(current_invoice)

            current_invoice = {
                "invoice_number": invoice_match.group(1),
                "invoice_date": None,
                "total_amount": None,
                "pages": []
            }

        # If no invoice number is found, keep the page
        # with the current invoice.
        if current_invoice is None:
            current_invoice = {
                "invoice_number": None,
                "invoice_date": None,
                "total_amount": None,
                "pages": []
            }

        current_invoice["pages"].append(page["page"])

        date_match = date_pattern.search(page_text)
        amount_match = amount_pattern.search(page_text)

        if date_match:
            current_invoice["invoice_date"] = date_match.group(1)

        if amount_match:
            current_invoice["total_amount"] = amount_match.group(1)

        chunks.append({
            "document_type": "scanned_invoice",
            "page": page["page"],
            "text": page_text
        })

    if current_invoice:
        invoices.append(current_invoice)

    return chunks, invoices
