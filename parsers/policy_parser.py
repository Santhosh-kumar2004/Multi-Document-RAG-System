
import re

def parse_policy(pages):
    chunks = []

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue
        sections = re.split(
            r"(?m)(?=^\d+(?:\.\d+)*\s+[A-Z])",
            text
        )

        for section in sections:
            section = section.strip()

            if len(section) > 30:
                chunks.append({
                    "document_type": "employee_policy",
                    "page": page["page"],
                    "text": section
                })

    return chunks
