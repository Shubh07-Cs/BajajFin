import asyncio
import logging
from app.services.document_parser import extract_text
from app.services.chunker import chunk_text
from app.services.embedding_engine import get_embeddings_batch, get_embedding
from app.services.vector_search import create_vector_client
from app.services.llm_service import generate_intelligent_answer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_workflow():
    """Test the complete workflow"""
    url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    question = "What is the grace period for premium payment?"
    
    try:
        # Step 1: Extract text
        logger.info("Step 1: Extracting text...")
        text = extract_text(url, "pdf")
        logger.info(f"✅ Text extracted: {len(text)} characters")
        
        # Step 2: Chunk text
        logger.info("Step 2: Chunking text...")
        chunks = chunk_text(text)
        logger.info(f"✅ Text chunked: {len(chunks)} chunks")
        
        # Step 3: Create vector client
        logger.info("Step 3: Creating vector client...")
        vector_client = create_vector_client()
        logger.info(f"✅ Vector client created: {type(vector_client)}")
        
        # Step 4: Generate embeddings
        logger.info("Step 4: Generating embeddings...")
        embeddings = await get_embeddings_batch(chunks[:10], provider="gemini")  # Test with first 10 chunks
        logger.info(f"✅ Embeddings generated: {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
        
        # Step 5: Upsert vectors
        logger.info("Step 5: Upserting vectors...")
        vectors_to_upsert = []
        for idx, emb in enumerate(embeddings):
            vectors_to_upsert.append((str(idx), emb, {"text": chunks[idx]}))
        vector_client.upsert(vectors_to_upsert)
        logger.info("✅ Vectors upserted successfully")
        
        # Step 6: Query vector database
        logger.info("Step 6: Querying vector database...")
        query_emb = await get_embedding(question, provider="gemini")
        matches = vector_client.query(query_emb, top_k=3)
        logger.info(f"✅ Found {len(matches)} matches")
        
        # Step 7: Generate answer
        logger.info("Step 7: Generating answer...")
        relevant_chunks = [match['metadata']['text'] for match in matches]
        answer_text, rationale = await generate_intelligent_answer(
            question=question,
            relevant_chunks=relevant_chunks,
            provider="gemini"
        )
        
        logger.info("✅ Workflow completed successfully!")
        print("\n" + "="*50)
        print("FINAL RESULT:")
        print("="*50)
        print(f"Question: {question}")
        print(f"Answer: {answer_text}")
        print(f"Rationale: {rationale}")
        print("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_workflow())
    print("Overall test:", "PASSED" if success else "FAILED")
