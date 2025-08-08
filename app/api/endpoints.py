from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, Answer, Clause
from app.services.document_parser import extract_text
from app.services.chunker import chunk_text
from app.services.embedding_engine import get_embeddings_batch, get_embedding
from app.services.vector_search import create_vector_client
from app.services.llm_service import generate_intelligent_answer, create_clause_explanations
from app.core.config import settings

import logging
import asyncio
import os

router = APIRouter()
logger = logging.getLogger("endpoints")

# Initialize vector database client (Pinecone or FAISS fallback)
try:
    vector_client = create_vector_client(
        index_name=settings.VECTOR_INDEX_NAME, 
        dimension=settings.EMBEDDING_DIMENSION
    )
except Exception as e:
    logger.error(f"Failed to initialize vector database client: {e}")
    vector_client = None


@router.post("/hackrx/run", response_model=QueryResponse)
async def run_query(payload: QueryRequest):
    doc_url = payload.documents
    # Parse URL to extract file extension, ignoring query parameters
    from urllib.parse import urlparse
    parsed_url = urlparse(doc_url)
    file_path = parsed_url.path.lower()
    
    if file_path.endswith(".pdf"):
        doc_type = "pdf"
    elif file_path.endswith(".docx"):
        doc_type = "docx"
    else:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX URLs are supported.")

    try:
        # Step 1: Extract document text
        text = extract_text(doc_url, doc_type)
        if not text.strip():
            raise HTTPException(status_code=422, detail="Failed to extract any text from the document.")

        # Step 2: Chunk text for semantic processing
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=422, detail="No valid chunks generated from document text.")

        # Step 3: Generate embeddings for chunks in batch (async)
        chunk_embeddings = await get_embeddings_batch(
            chunks, 
            provider=settings.DEFAULT_EMBEDDING_PROVIDER
        )

        # Step 4: Upsert embeddings with metadata into vector DB
        if vector_client:
            vectors_to_upsert = []
            for idx, emb in enumerate(chunk_embeddings):
                vectors_to_upsert.append((str(idx), emb, {"text": chunks[idx]}))
            vector_client.upsert(vectors_to_upsert)
        else:
            raise HTTPException(status_code=500, detail="Vector database not available")

        # Step 5: Process each question: embed, search top-k chunks, build response
        results = []
        for question in payload.questions:
            # Embed query
            query_emb = await get_embedding(
                question, 
                provider=settings.DEFAULT_EMBEDDING_PROVIDER
            )

            # Search vector DB for top-k matching chunks
            matches = vector_client.query(query_emb, top_k=settings.TOP_K_RESULTS)

            # Extract relevant chunks and scores
            relevant_chunks = [match['metadata']['text'] for match in matches]
            scores = [match['score'] for match in matches]
            
            # Generate intelligent answer using LLM
            answer_text, rationale = await generate_intelligent_answer(
                question=question,
                relevant_chunks=relevant_chunks[:3],  # Use top 3 chunks for answer
                provider=settings.DEFAULT_LLM_PROVIDER
            )
            
            # Create detailed clause explanations
            matched_clauses = await create_clause_explanations(
                chunks=relevant_chunks,
                scores=scores,
                question=question
            )

            results.append(
                Answer(
                    answer=answer_text,
                    clauses=matched_clauses,
                    decision_rationale=rationale
                )
            )

        return QueryResponse(answers=results)

    except Exception as e:
        logger.exception(f"Error processing /hackrx/run: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "ok"}
