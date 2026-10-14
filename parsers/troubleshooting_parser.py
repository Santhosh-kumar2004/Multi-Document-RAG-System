import re

def parse_troubleshooting_guide(pages):
    chunks = []

    for page in pages:
        text = page["text"]

        sections = re.split(
            r"(?im)(?=^(?:step\s+\d+|"
            r"\d+[.)]\s+|warning\s*:|"
            r"caution\s*:|note\s*:|"
            r"error\s*:))",
            text
        )

        for section in sections:
            section = section.strip()

            if section:
                chunks.append({
                    "document_type": "troubleshooting_guide",
                    "page": page["page"],
                    "text": section
                })

    return chunks
