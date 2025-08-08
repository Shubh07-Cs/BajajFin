from typing import List

def chunk_text(text: str, size: int = 300, overlap: int = 50) -> List[str]:
    """
    Splits large text into overlapping word chunks for efficient retrieval.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += size - overlap
    return chunks
