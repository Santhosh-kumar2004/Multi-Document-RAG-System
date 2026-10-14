import re


def parse_product_documentation(pages):
    chunks = []

    for page in pages:
        text = page["text"]

        sections = re.split(
            r"(?m)(?=^#{1,6}\s|^\d+(?:\.\d+)*\s+\S)",
            text
        )

        for section in sections:
            section = section.strip()

            if section:
                chunks.append({
                    "document_type": "product_documentation",
                    "page": page["page"],
                    "text": section
                })

    return chunks
