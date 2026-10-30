from pdf_processor import process_pdf

pages, chunks = process_pdf(
    "uploads/employee_policy_Employee_Policy.pdf",
    document_type="employee_policy",
    chunking_method="dynamic"
)

print("Total Pages:", len(pages))
print("Total Chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print("\nChunk:", index)
    print("Pages:", chunk["pages"])
    print("Method:", chunk["chunking_method"])
    print("Tokens:", chunk["token_count"])
    print("Text:", chunk["text"][:150])