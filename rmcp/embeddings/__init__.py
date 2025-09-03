"""
Embeddings module for RMCP
Provides vector operations and semantic similarity
"""

from .manager import EmbeddingManager
from .store import EmbeddingStore
from .similarity import CosineSimilarity

__all__ = ["EmbeddingManager", "EmbeddingStore", "CosineSimilarity"]

