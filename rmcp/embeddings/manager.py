"""
Embedding Manager for RMCP - manages vector embeddings and similarity
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .similarity import CosineSimilarity
from ..llm.manager import LLMManager


class EmbeddingManager:
    """Manages vector embeddings and similarity calculations"""
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        self.llm_manager = llm_manager
        self.embedding_cache: Dict[str, List[float]] = {}
        self.cache_size_limit = 1000  # Maximum number of cached embeddings
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Vector embedding as list of floats
        """
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Generate embedding using LLM manager
        if self.llm_manager:
            try:
                embedding = await self.llm_manager.generate_embedding(text)
            except Exception as e:
                print(f"Failed to generate embedding: {e}")
                # Fallback to mock embedding
                embedding = self._generate_mock_embedding(text)
        else:
            # Fallback to mock embedding
            embedding = self._generate_mock_embedding(text)
        
        # Cache the embedding
        self._cache_embedding(text, embedding)
        
        return embedding
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """
        Generate a mock embedding for testing purposes
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Mock embedding vector
        """
        # Simple hash-based mock embedding
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert hash to embedding-like vector
        embedding = []
        for i in range(0, len(hash_bytes), 4):
            chunk = hash_bytes[i:i+4]
            if len(chunk) == 4:
                # Convert 4 bytes to float
                value = int.from_bytes(chunk, byteorder='big')
                # Normalize to [-1, 1] range
                normalized = (value / (2**32 - 1)) * 2 - 1
                embedding.append(normalized)
        
        # Pad or truncate to standard embedding size (384)
        while len(embedding) < 384:
            embedding.append(0.0)
        
        return embedding[:384]
    
    def _cache_embedding(self, text: str, embedding: List[float]):
        """
        Cache an embedding
        
        Args:
            text: Text that was embedded
            embedding: Generated embedding
        """
        # Implement simple LRU cache
        if len(self.embedding_cache) >= self.cache_size_limit:
            # Remove oldest entry (simple implementation)
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        self.embedding_cache[text] = embedding
    
    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score
        """
        embedding1 = await self.generate_embedding(text1)
        embedding2 = await self.generate_embedding(text2)
        
        return CosineSimilarity.calculate(embedding1, embedding2)
    
    async def find_most_similar(
        self, 
        query_text: str, 
        candidate_texts: List[str], 
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find most similar texts to a query
        
        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity) tuples
        """
        query_embedding = await self.generate_embedding(query_text)
        
        # Generate embeddings for all candidates
        candidate_embeddings = []
        for candidate_text in candidate_texts:
            embedding = await self.generate_embedding(candidate_text)
            candidate_embeddings.append(embedding)
        
        return CosineSimilarity.find_most_similar(
            query_embedding, 
            candidate_embeddings, 
            top_k
        )
    
    async def calculate_affinity_score(
        self, 
        query_text: str, 
        stored_embeddings: List[List[float]]
    ) -> float:
        """
        Calculate affinity score between query and stored embeddings
        
        Args:
            query_text: Query text
            stored_embeddings: List of stored embeddings (successful precedents)
            
        Returns:
            Affinity score (0.0 to 1.0)
        """
        if not stored_embeddings:
            return 0.0
        
        query_embedding = await self.generate_embedding(query_text)
        
        # Calculate similarities with all stored embeddings
        similarities = CosineSimilarity.calculate_batch(query_embedding, stored_embeddings)
        
        # Return maximum similarity as affinity score
        return max(similarities) if similarities else 0.0
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.embedding_cache),
            "cache_limit": self.cache_size_limit,
            "cache_usage": len(self.embedding_cache) / self.cache_size_limit
        }

