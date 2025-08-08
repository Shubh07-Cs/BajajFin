from app.services.document_parser import extract_text

url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

try:
    text = extract_text(url, "pdf")
    print("Text extracted successfully, length:", len(text))
    print("First 200 chars:", text[:200])
except Exception as e:
    print("Error:", e)
