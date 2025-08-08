# app/services/vector_search.py

import os
from typing import List, Dict, Optional, Union
import logging
import numpy as np
import json

logger = logging.getLogger(__name__)

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Pinecone not available, using FAISS fallback")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available")


class PineconeClient:
    """
    Wrapper class for Pinecone Vector DB operations: initialization, upsert, and query.
    """

    def __init__(self, api_key: Optional[str] = None, environment: Optional[str] = None, index_name: str = "policy-index"):
        """
        Initialize Pinecone client and index.

        Args:
            api_key (str): Pinecone API key (optional if set in env var).
            environment (str): Pinecone environment (optional if set in env var).
            index_name (str): Name of the Pinecone index to use/create.
        """
        api_key = api_key or os.getenv("PINECONE_API_KEY")
        environment = environment or os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")

        if not api_key:
            raise ValueError("PINECONE_API_KEY must be set")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

        # Check if index exists
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            # Create index with serverless spec
            self.pc.create_index(
                name=index_name,
                dimension=1536,  # Dimension for text-embedding-ada-002 or similar
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
            logger.info(f"Created new Pinecone index: {index_name}")

        self.index = self.pc.Index(index_name)
        logger.info(f"Pinecone index initialized: {index_name}")

    def upsert(self, vectors: List[tuple]):
        """
        Upsert vectors into Pinecone index.

        Args:
            vectors (List[tuple]): List of tuples (id:str, embedding:List[float], metadata:dict).
        """
        # Batch limit is usually 100 vectors per upsert @ Pinecone, chunk if needed
        chunk_size = 100
        for i in range(0, len(vectors), chunk_size):
            batch = vectors[i:i+chunk_size]
            self.index.upsert(vectors=batch)
            logger.debug(f"Upserted batch of {len(batch)} vectors to Pinecone")

    def query(self, embedding: List[float], top_k: int = 5, include_metadata: bool = True) -> List[Dict]:
        """
        Query the Pinecone index to find top_k similar vectors.

        Args:
            embedding (List[float]): Query embedding vector.
            top_k (int): Number of top results to return.
            include_metadata (bool): Whether to include metadata in results.

        Returns:
            List[Dict]: List of matching items with 'id', 'score', and optionally 'metadata'.
        """
        response = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=include_metadata
        )
        return response['matches']


class FAISSClient:
    """
    Local FAISS vector database for development and testing.
    """
    
    def __init__(self, index_name: str = "policy-index", dimension: int = 1536):
        """
        Initialize FAISS index.
        
        Args:
            index_name (str): Name of the index (used for local file storage).
            dimension (int): Vector dimension.
        """
        self.index_name = index_name
        self.dimension = dimension
        self.index_file = f"{index_name}.faiss"
        self.metadata_file = f"{index_name}_metadata.json"
        self.id_mapping_file = f"{index_name}_id_mapping.json"
        
        # Initialize or load index
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
            with open(self.id_mapping_file, 'r') as f:
                self.id_mapping = json.load(f)
            logger.info(f"Loaded existing FAISS index: {index_name}")
        else:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product similarity
            self.metadata = {}
            self.id_mapping = {}  # Maps FAISS index -> original vector ID
            logger.info(f"Created new FAISS index: {index_name}")
    
    def upsert(self, vectors: List[tuple]):
        """
        Upsert vectors into FAISS index.
        
        Args:
            vectors (List[tuple]): List of tuples (id:str, embedding:List[float], metadata:dict).
        """
        embeddings = []
        start_idx = self.index.ntotal  # Current size of index
        
        for i, (vector_id, embedding, metadata) in enumerate(vectors):
            embeddings.append(embedding)
            self.metadata[vector_id] = metadata
            # Map FAISS index to original vector ID
            self.id_mapping[str(start_idx + i)] = vector_id
        
        # Convert to numpy array and normalize for cosine similarity
        embeddings_np = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_np)  # Normalize for cosine similarity
        
        # Add to index
        self.index.add(embeddings_np)
        
        # Save index and metadata
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
        with open(self.id_mapping_file, 'w') as f:
            json.dump(self.id_mapping, f)
        
        logger.debug(f"Upserted {len(vectors)} vectors to FAISS index")
    
    def query(self, embedding: List[float], top_k: int = 5, include_metadata: bool = True) -> List[Dict]:
        """
        Query the FAISS index to find top_k similar vectors.
        
        Args:
            embedding (List[float]): Query embedding vector.
            top_k (int): Number of top results to return.
            include_metadata (bool): Whether to include metadata in results.
        
        Returns:
            List[Dict]: List of matching items with 'id', 'score', and optionally 'metadata'.
        """
        if self.index.ntotal == 0:
            return []
        
        # Convert query to numpy and normalize
        query_np = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(query_np)
        
        # Search
        scores, indices = self.index.search(query_np, min(top_k, self.index.ntotal))
        
        # Format results
        matches = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # No more results
                break
            
            # Map FAISS index back to original vector ID
            faiss_idx = str(idx)
            original_id = self.id_mapping.get(faiss_idx, faiss_idx)
            
            match = {
                'id': original_id,
                'score': float(score)
            }
            
            if include_metadata and original_id in self.metadata:
                match['metadata'] = self.metadata[original_id]
            
            matches.append(match)
        
        return matches


def create_vector_client(index_name: str = "policy-index", dimension: int = 768) -> Union[PineconeClient, FAISSClient]:
    """
    Create a vector database client, preferring Pinecone but falling back to FAISS.
    
    Args:
        index_name (str): Name of the index.
    
    Returns:
        Union[PineconeClient, FAISSClient]: Vector database client.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    
    if PINECONE_AVAILABLE and api_key:
        try:
            return PineconeClient(index_name=index_name)
        except Exception as e:
            logger.warning(f"Failed to initialize Pinecone client: {e}")
    
    if FAISS_AVAILABLE:
        return FAISSClient(index_name=index_name, dimension=dimension)
    
    raise RuntimeError("No vector database available (neither Pinecone nor FAISS)")
