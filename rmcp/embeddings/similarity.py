"""
Cosine similarity calculations for embeddings
"""

import numpy as np
from typing import List, Union
import math


class CosineSimilarity:
    """Cosine similarity calculations for vector embeddings"""
    
    @staticmethod
    def calculate(a: List[float], b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        if not a or not b:
            return 0.0
        
        if len(a) != len(b):
            return 0.0
        
        # Convert to numpy arrays for efficient computation
        vec_a = np.array(a, dtype=np.float32)
        vec_b = np.array(b, dtype=np.float32)
        
        # Calculate dot product
        dot_product = np.dot(vec_a, vec_b)
        
        # Calculate magnitudes
        magnitude_a = np.linalg.norm(vec_a)
        magnitude_b = np.linalg.norm(vec_b)
        
        # Avoid division by zero
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = dot_product / (magnitude_a * magnitude_b)
        
        # Ensure result is within valid range
        return float(np.clip(similarity, -1.0, 1.0))
    
    @staticmethod
    def calculate_batch(query_vector: List[float], candidate_vectors: List[List[float]]) -> List[float]:
        """
        Calculate cosine similarity between a query vector and multiple candidate vectors
        
        Args:
            query_vector: Query vector
            candidate_vectors: List of candidate vectors
            
        Returns:
            List of similarity scores
        """
        if not query_vector or not candidate_vectors:
            return [0.0] * len(candidate_vectors)
        
        # Convert to numpy arrays
        query = np.array(query_vector, dtype=np.float32)
        candidates = np.array(candidate_vectors, dtype=np.float32)
        
        # Calculate dot products
        dot_products = np.dot(candidates, query)
        
        # Calculate magnitudes
        query_magnitude = np.linalg.norm(query)
        candidate_magnitudes = np.linalg.norm(candidates, axis=1)
        
        # Avoid division by zero
        if query_magnitude == 0:
            return [0.0] * len(candidate_vectors)
        
        # Calculate similarities
        similarities = dot_products / (candidate_magnitudes * query_magnitude)
        
        # Handle zero magnitudes
        similarities[candidate_magnitudes == 0] = 0.0
        
        # Ensure results are within valid range
        similarities = np.clip(similarities, -1.0, 1.0)
        
        return similarities.tolist()
    
    @staticmethod
    def find_most_similar(
        query_vector: List[float], 
        candidate_vectors: List[List[float]], 
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find the most similar vectors to a query vector
        
        Args:
            query_vector: Query vector
            candidate_vectors: List of candidate vectors
            top_k: Number of top results to return
            
        Returns:
            List of tuples (index, similarity_score) sorted by similarity
        """
        similarities = CosineSimilarity.calculate_batch(query_vector, candidate_vectors)
        
        # Create list of (index, similarity) tuples
        indexed_similarities = [(i, sim) for i, sim in enumerate(similarities)]
        
        # Sort by similarity (descending)
        indexed_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return indexed_similarities[:top_k]
    
    @staticmethod
    def calculate_distance(a: List[float], b: List[float]) -> float:
        """
        Calculate cosine distance (1 - cosine similarity)
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine distance between 0 and 2
        """
        similarity = CosineSimilarity.calculate(a, b)
        return 1.0 - similarity
    
    @staticmethod
    def is_similar(
        a: List[float], 
        b: List[float], 
        threshold: float = 0.7
    ) -> bool:
        """
        Check if two vectors are similar based on threshold
        
        Args:
            a: First vector
            b: Second vector
            threshold: Similarity threshold (default 0.7)
            
        Returns:
            True if vectors are similar, False otherwise
        """
        similarity = CosineSimilarity.calculate(a, b)
        return similarity >= threshold

