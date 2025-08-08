import asyncio
from app.services.embedding_engine import get_embedding

async def test():
    try:
        emb = await get_embedding('test text', 'gemini')
        print('Embedding created, dimension:', len(emb))
        return True
    except Exception as e:
        print('Error:', e)
        return False

if __name__ == "__main__":
    result = asyncio.run(test())
    print("Test passed:" if result else "Test failed:", result)
