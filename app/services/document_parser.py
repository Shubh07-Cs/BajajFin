import os
import tempfile
import requests
import pdfplumber
from docx import Document
from urllib.parse import urlparse


def _download_file(url: str) -> str:
    """
    Downloads a file from a URL to a temporary file, sanitizing the file suffix.
    """
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    parsed_url = urlparse(url)
    path = parsed_url.path  # Extract only the path part without query parameters
    suffix = os.path.splitext(path)[-1] or ".tmp"

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, 'wb') as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            tmp.write(chunk)

    return tmp_path


def extract_text(doc_url: str, doc_type: str) -> str:
    """
    Extract raw text from a PDF or DOCX file provided by URL.
    Supports: PDF, DOCX.
    """
    file_path = _download_file(doc_url)
    try:
        if doc_type.lower() == "pdf":
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        elif doc_type.lower() == "docx":
            doc = Document(file_path)
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        else:
            raise NotImplementedError("Only PDF and DOCX parsing implemented.")
        return text
    finally:
        os.remove(file_path)
