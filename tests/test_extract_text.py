from app.services.document_parser import extract_text

def test_extract_text():
    # Example PDF document URL (replace with your test URL)
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    # Test PDF extraction
    print("\nTesting PDF extraction...")
    pdf_text = extract_text(pdf_url, "pdf")
    print(f"Extracted PDF text length: {len(pdf_text)} characters")
    print("Sample extracted text (first 500 chars):\n", pdf_text[:500])
    
    # If you also have a DOCX URL, test like this:
    # docx_url = "https://example.com/sample.docx"
    # print("\nTesting DOCX extraction...")
    # docx_text = extract_text(docx_url, "docx")
    # print(f"Extracted DOCX text length: {len(docx_text)} characters")
    # print("Sample extracted text (first 500 chars):\n", docx_text[:500])

if __name__ == "__main__":
    test_extract_text()
