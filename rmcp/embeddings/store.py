"""
Embedding Store for RMCP - manages storage and retrieval of embeddings
"""

import json
import struct
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .manager import EmbeddingManager
from .similarity import CosineSimilarity


class EmbeddingStore:
    """Manages storage and retrieval of embeddings for semantic affinity"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.max_embeddings_per_tool = 10  # Maximum number of embeddings to store per tool
    
    def serialize_embeddings(self, embeddings: List[List[float]]) -> bytes:
        """
        Serialize embeddings to binary format for database storage
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Binary data containing serialized embeddings
        """
        if not embeddings:
            return b''
        
        # Convert to numpy array for efficient serialization
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Get shape information
        shape = embeddings_array.shape
        
        # Serialize: shape (2 ints) + data (floats)
        shape_bytes = struct.pack('2I', *shape)
        data_bytes = embeddings_array.tobytes()
        
        return shape_bytes + data_bytes
    
    def deserialize_embeddings(self, data: bytes) -> List[List[float]]:
        """
        Deserialize embeddings from binary format
        
        Args:
            data: Binary data containing serialized embeddings
            
        Returns:
            List of embedding vectors
        """
        if not data:
            return []
        
        try:
            # Read shape information
            shape_bytes = data[:8]  # 2 ints * 4 bytes each
            shape = struct.unpack('2I', shape_bytes)
            
            # Read data
            data_bytes = data[8:]
            embeddings_array = np.frombuffer(data_bytes, dtype=np.float32)
            embeddings_array = embeddings_array.reshape(shape)
            
            return embeddings_array.tolist()
            
        except Exception as e:
            print(f"Failed to deserialize embeddings: {e}")
            return []
    
    async def add_successful_embedding(
        self, 
        tool_id: str, 
        request_text: str, 
        current_embeddings: List[List[float]]
    ) -> List[List[float]]:
        """
        Add a successful embedding to the tool's affinity embeddings
        
        Args:
            tool_id: Tool identifier
            request_text: Text of the successful request
            current_embeddings: Current stored embeddings
            
        Returns:
            Updated list of embeddings
        """
        # Generate embedding for the successful request
        new_embedding = await self.embedding_manager.generate_embedding(request_text)
        
        # Check if this embedding is sufficiently unique
        if self._is_embedding_unique(new_embedding, current_embeddings):
            # Add to the list
            updated_embeddings = current_embeddings + [new_embedding]
            
            # Keep only the most recent embeddings if we exceed the limit
            if len(updated_embeddings) > self.max_embeddings_per_tool:
                updated_embeddings = updated_embeddings[-self.max_embeddings_per_tool:]
            
            return updated_embeddings
        
        return current_embeddings
    
    def _is_embedding_unique(
        self, 
        new_embedding: List[float], 
        existing_embeddings: List[List[float]], 
        threshold: float = 0.8
    ) -> bool:
        """
        Check if an embedding is sufficiently unique compared to existing ones
        
        Args:
            new_embedding: New embedding to check
            existing_embeddings: List of existing embeddings
            threshold: Similarity threshold for uniqueness
            
        Returns:
            True if embedding is unique, False otherwise
        """
        if not existing_embeddings:
            return True
        
        # Calculate similarities with existing embeddings
        similarities = CosineSimilarity.calculate_batch(new_embedding, existing_embeddings)
        
        # Check if any similarity exceeds the threshold
        max_similarity = max(similarities) if similarities else 0.0
        
        return max_similarity < threshold
    
    async def calculate_affinity_score(
        self, 
        query_text: str, 
        tool_embeddings: List[List[float]]
    ) -> float:
        """
        Calculate affinity score between query and tool's successful precedents
        
        Args:
            query_text: Query text
            tool_embeddings: Tool's stored successful embeddings
            
        Returns:
            Affinity score (0.0 to 1.0)
        """
        return await self.embedding_manager.calculate_affinity_score(query_text, tool_embeddings)
    
    def get_embedding_stats(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Get statistics about stored embeddings
        
        Args:
            embeddings: List of embeddings
            
        Returns:
            Dictionary with embedding statistics
        """
        if not embeddings:
            return {
                "count": 0,
                "dimension": 0,
                "avg_similarity": 0.0,
                "diversity_score": 0.0
            }
        
        # Calculate average similarity between all pairs
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = CosineSimilarity.calculate(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        # Calculate diversity score (1 - average similarity)
        diversity_score = 1.0 - avg_similarity
        
        return {
            "count": len(embeddings),
            "dimension": len(embeddings[0]) if embeddings else 0,
            "avg_similarity": float(avg_similarity),
            "diversity_score": float(diversity_score)
        }
    
    def validate_embeddings(self, embeddings: List[List[float]]) -> bool:
        """
        Validate that embeddings are in correct format
        
        Args:
            embeddings: List of embeddings to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not embeddings:
            return True
        
        # Check that all embeddings have the same dimension
        dimension = len(embeddings[0])
        for embedding in embeddings:
            if len(embedding) != dimension:
                return False
            
            # Check that all values are finite numbers
            for value in embedding:
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    return False
        
        return True

