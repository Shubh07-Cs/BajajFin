import os
import aiohttp
import asyncio
import openai
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Initialize OpenAI client
openai_client = None
if OPENAI_API_KEY:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Gemini Embedding API endpoint (corrected)
GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"

# DeepSeek API endpoint
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


async def get_embedding(text: str, provider: str = "openai") -> List[float]:
    """
    Generate embedding for a single text using specified provider.
    
    Args:
        text: Input text to embed
        provider: "openai", "gemini", or "deepseek"
    
    Returns:
        List of embedding floats
    """
    if provider == "openai" and openai_client:
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    
    elif provider == "gemini" and GEMINI_API_KEY:
        return await _get_gemini_embedding(text)
    
    elif provider == "deepseek" and DEEPSEEK_API_KEY:
        return await _get_deepseek_embedding(text)
    
    else:
        raise ValueError(f"Provider '{provider}' not available or API key missing")


async def get_embeddings_batch(texts: List[str], provider: str = "openai") -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batch.
    """
    if provider == "openai" and openai_client:
        # OpenAI supports batch processing
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [item.embedding for item in response.data]
    
    else:
        # For other providers, process one by one
        tasks = [get_embedding(text, provider) for text in texts]
        return await asyncio.gather(*tasks)


async def _get_gemini_embedding(text: str) -> List[float]:
    """Get embedding from Gemini API"""
    async with aiohttp.ClientSession() as session:
        url = f"{GEMINI_EMBED_URL}?key={GEMINI_API_KEY}"
        payload = {
            "model": "models/embedding-001",
            "content": {
                "parts": [{
                    "text": text
                }]
            }
        }
        
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Gemini Embedding API error {resp.status}: {error_text}")
            
            data = await resp.json()
            return data['embedding']['values']


async def _get_deepseek_embedding(text: str) -> List[float]:
    """Get embedding from DeepSeek API"""
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "text-embedding-ada-002",  # DeepSeek compatible model
            "input": text
        }
        
        async with session.post(f"{DEEPSEEK_BASE_URL}/embeddings", 
                               json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"DeepSeek API error {resp.status}: {error_text}")
            
            data = await resp.json()
            return data['data'][0]['embedding']


# Text generation functions (keeping your existing Gemini chat functionality)
async def generate_gemini_response(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Generate text response using Gemini"""
    gemini_chat_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    
    async with aiohttp.ClientSession() as session:
        url = f"{gemini_chat_url}?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "candidateCount": 1
            }
        }
        
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Gemini API error {resp.status}: {error_text}")
            
            data = await resp.json()
            return data['candidates'][0]['content']['parts'][0]['text']


async def generate_gemini_responses(prompts: List[str], max_tokens: int = 512, temperature: float = 0.7) -> List[str]:
    """Generate multiple text responses using Gemini"""
    tasks = [generate_gemini_response(p, max_tokens=max_tokens, temperature=temperature) for p in prompts]
    return await asyncio.gather(*tasks)
