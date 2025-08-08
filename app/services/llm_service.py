import os
import asyncio
from typing import List
from app.services.embedding_engine import generate_gemini_response
from app.models.schemas import Clause


async def generate_intelligent_answer(question: str, relevant_chunks: List[str], provider: str = "gemini") -> tuple[str, str]:
    """
    Generate an intelligent answer using LLM based on question and relevant document chunks.
    
    Args:
        question: User's question
        relevant_chunks: List of relevant text chunks from document
        provider: LLM provider ("gemini" or "deepseek")
    
    Returns:
        Tuple of (answer, rationale)
    """
    # Combine relevant chunks into context
    context = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(relevant_chunks)])
    
    prompt = f"""
Based on the following document excerpts, please provide a comprehensive answer to the question.

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

Instructions:
1. Provide a direct, accurate answer based only on the information in the document excerpts
2. If the answer is not clearly found in the excerpts, state "The document does not contain sufficient information to answer this question"
3. Cite specific parts of the document that support your answer
4. Be concise but thorough

ANSWER:
"""

    if provider == "gemini":
        response = await generate_gemini_response(prompt, max_tokens=800, temperature=0.3)
    else:
        # Add DeepSeek or other providers here if needed
        response = await generate_gemini_response(prompt, max_tokens=800, temperature=0.3)
    
    # Extract answer and create rationale
    answer = response.strip()
    
    rationale = f"Answer generated using {provider.upper()} LLM based on {len(relevant_chunks)} most relevant document sections retrieved through semantic vector search."
    
    return answer, rationale


async def create_clause_explanations(chunks: List[str], scores: List[float], question: str) -> List[Clause]:
    """
    Create Clause objects with intelligent explanations for why each chunk is relevant.
    
    Args:
        chunks: Text chunks from document
        scores: Similarity scores for each chunk
        question: Original question
    
    Returns:
        List of Clause objects with explanations
    """
    clauses = []
    
    for i, (chunk, score) in enumerate(zip(chunks, scores)):
        # Generate explanation for why this chunk is relevant
        explanation_prompt = f"""
Explain in 1-2 sentences why this document excerpt is relevant to the question: "{question}"

Document excerpt:
{chunk[:500]}...

Keep the explanation concise and specific.
"""
        
        try:
            explanation = await generate_gemini_response(explanation_prompt, max_tokens=150, temperature=0.2)
            explanation = f"Relevance score: {score:.3f} - {explanation.strip()}"
        except Exception:
            # Fallback explanation if LLM fails
            explanation = f"Semantic similarity score: {score:.3f} - This section contains content related to your query."
        
        clauses.append(Clause(
            text=chunk,
            explanation=explanation
        ))
    
    return clauses
