"""
Stage 2: The Compass - Semantic guidance and ranking
Implements semantic ranking using cosine similarity of request embedding with successful tool request embeddings
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from ..storage.database import DatabaseManager
from ..models.tool import Tool
from ..embeddings.manager import EmbeddingManager
from ..embeddings.store import EmbeddingStore
from ..embeddings.similarity import CosineSimilarity


class CompassStage:
    """
    Stage 2: The Compass - Semantic ranking (2-5ms)
    
    Performs semantic ranking using:
    - Cosine similarity with successful precedents
    - Dynamic metrics (success_rate, latency, cost)
    - Affinity scoring
    """
    
    def __init__(
        self, 
        db_manager: DatabaseManager, 
        embedding_manager: EmbeddingManager,
        embedding_store: EmbeddingStore
    ):
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        self.embedding_store = embedding_store
        self.max_candidates = 20  # Maximum candidates for semantic ranking
    
    async def rank_candidates(
        self, 
        goal: str, 
        context: Dict[str, Any],
        candidates: List[Tool],
        max_candidates: Optional[int] = None
    ) -> List[Tuple[Tool, float]]:
        """
        Perform semantic ranking of tool candidates
        
        Args:
            goal: Task goal/description
            context: Additional context
            candidates: List of tool candidates from Sieve stage
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of (tool, affinity_score) tuples, sorted by affinity
        """
        start_time = time.time()
        
        if not candidates:
            return []
        
        # Limit candidates for semantic processing
        limit = max_candidates or self.max_candidates
        candidates = candidates[:limit]
        
        # Step 1: Calculate affinity scores for all candidates
        affinity_scores = []
        for tool in candidates:
            affinity_score = await self._calculate_affinity_score(tool, goal, context)
            affinity_scores.append((tool, affinity_score))
        
        # Step 2: Apply dynamic metrics weighting
        weighted_scores = self._apply_dynamic_metrics(affinity_scores, context)
        
        # Step 3: Sort by final score
        ranked_candidates = sorted(weighted_scores, key=lambda x: x[1], reverse=True)
        
        # Log performance
        elapsed_ms = (time.time() - start_time) * 1000
        print(f"CompassStage: Ranked {len(ranked_candidates)} candidates in {elapsed_ms:.2f}ms")
        
        return ranked_candidates
    
    async def _calculate_affinity_score(
        self, 
        tool: Tool, 
        goal: str, 
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate semantic affinity score for a tool
        
        Args:
            tool: Tool to score
            goal: Task goal
            context: Additional context
            
        Returns:
            Affinity score (0.0 to 1.0)
        """
        # Get stored embeddings for this tool
        embeddings_data, embedding_count = self.db_manager.get_tool_embeddings(tool.id)
        
        if not embeddings_data or embedding_count == 0:
            # No stored embeddings, use fallback scoring
            return self._fallback_affinity_score(tool, goal, context)
        
        # Deserialize embeddings
        stored_embeddings = self.embedding_store.deserialize_embeddings(embeddings_data)
        
        if not stored_embeddings:
            return self._fallback_affinity_score(tool, goal, context)
        
        # Calculate affinity score using stored embeddings
        affinity_score = await self.embedding_store.calculate_affinity_score(goal, stored_embeddings)
        
        return affinity_score
    
    def _fallback_affinity_score(
        self, 
        tool: Tool, 
        goal: str, 
        context: Dict[str, Any]
    ) -> float:
        """
        Fallback affinity scoring when no stored embeddings are available
        
        Args:
            tool: Tool to score
            goal: Task goal
            context: Additional context
            
        Returns:
            Fallback affinity score (0.0 to 1.0)
        """
        # Simple keyword-based scoring
        goal_keywords = set(goal.lower().split())
        tool_description = (tool.description or "").lower()
        tool_name = tool.name.lower()
        
        # Count keyword matches
        description_matches = len(goal_keywords.intersection(set(tool_description.split())))
        name_matches = len(goal_keywords.intersection(set(tool_name.split())))
        
        # Calculate base score with enhanced matching
        total_keywords = len(goal_keywords)
        if total_keywords == 0:
            return 0.5  # Neutral score
        
        # Enhanced scoring with tool-specific bonuses
        base_score = (description_matches + name_matches * 2) / (total_keywords * 3)
        
        # Add tool-specific scoring bonuses
        tool_bonus = 0.0
        goal_lower = goal.lower()
        
        # Echo tool bonus
        if "echo" in tool_name and ("echo" in goal_lower or "print" in goal_lower or "output" in goal_lower):
            tool_bonus += 0.3
        
        # File listing bonus
        elif "ls" in tool_name and ("list" in goal_lower or "directory" in goal_lower or "files" in goal_lower):
            tool_bonus += 0.3
        
        # Word count bonus
        elif "wc" in tool_name and ("count" in goal_lower or "words" in goal_lower or "lines" in goal_lower):
            tool_bonus += 0.3
        
        # Search/grep bonus
        elif "grep" in tool_name and ("search" in goal_lower or "find" in goal_lower or "pattern" in goal_lower):
            tool_bonus += 0.3
        
        # File creation bonus
        elif "create_file" in tool_name and ("create" in goal_lower and "file" in goal_lower):
            tool_bonus += 0.3
        
        # File reading bonus
        elif "cat" in tool_name and ("show" in goal_lower or "content" in goal_lower or "read" in goal_lower):
            tool_bonus += 0.3
        
        base_score = min(1.0, base_score + tool_bonus)
        
        # Boost score based on tool metrics
        success_boost = tool.success_rate * 0.2
        latency_boost = max(0, (1 - tool.p95_latency_ms / 10000)) * 0.1
        
        final_score = min(1.0, base_score + success_boost + latency_boost)
        
        return final_score
    
    def _apply_dynamic_metrics(
        self, 
        affinity_scores: List[Tuple[Tool, float]], 
        context: Dict[str, Any]
    ) -> List[Tuple[Tool, float]]:
        """
        Apply dynamic metrics weighting to affinity scores
        
        Args:
            affinity_scores: List of (tool, affinity_score) tuples
            context: Additional context
            
        Returns:
            List of (tool, final_score) tuples
        """
        # Get weighting preferences from context
        success_weight = context.get("success_weight", 0.3)
        latency_weight = context.get("latency_weight", 0.2)
        cost_weight = context.get("cost_weight", 0.1)
        affinity_weight = context.get("affinity_weight", 0.4)
        
        weighted_scores = []
        
        for tool, affinity_score in affinity_scores:
            # Normalize metrics
            success_score = tool.success_rate
            latency_score = max(0, 1 - (tool.p95_latency_ms / 10000))  # Normalize to 10s max
            cost_score = max(0, 1 - (tool.cost_hint / 100))  # Normalize to $100 max
            
            # Calculate weighted final score
            final_score = (
                affinity_score * affinity_weight +
                success_score * success_weight +
                latency_score * latency_weight +
                cost_score * cost_weight
            )
            
            weighted_scores.append((tool, final_score))
        
        return weighted_scores
    
    async def update_tool_affinity(
        self, 
        tool_id: str, 
        request_text: str, 
        success: bool
    ) -> None:
        """
        Update tool's affinity embeddings with successful request
        
        Args:
            tool_id: Tool identifier
            request_text: Text of the successful request
            success: Whether the request was successful
        """
        if not success:
            return  # Only update with successful requests
        
        # Get current embeddings
        embeddings_data, embedding_count = self.db_manager.get_tool_embeddings(tool_id)
        
        # Deserialize current embeddings
        current_embeddings = []
        if embeddings_data:
            current_embeddings = self.embedding_store.deserialize_embeddings(embeddings_data)
        
        # Add new successful embedding
        updated_embeddings = await self.embedding_store.add_successful_embedding(
            tool_id, 
            request_text, 
            current_embeddings
        )
        
        # Serialize and store updated embeddings
        if updated_embeddings != current_embeddings:
            new_embeddings_data = self.embedding_store.serialize_embeddings(updated_embeddings)
            new_embedding_count = len(updated_embeddings)
            
            self.db_manager.update_tool_embeddings(tool_id, new_embeddings_data, new_embedding_count)
    
    def get_ranking_stats(self) -> Dict[str, Any]:
        """
        Get statistics about ranking performance
        
        Returns:
            Dictionary with ranking statistics
        """
        return {
            "max_candidates": self.max_candidates,
            "stage": "compass",
            "description": "Semantic guidance and ranking",
            "embedding_cache_stats": self.embedding_manager.get_cache_stats()
        }
    
    async def analyze_tool_affinity(self, tool_id: str) -> Dict[str, Any]:
        """
        Analyze tool's affinity embeddings
        
        Args:
            tool_id: Tool identifier
            
        Returns:
            Dictionary with affinity analysis
        """
        embeddings_data, embedding_count = self.db_manager.get_tool_embeddings(tool_id)
        
        if not embeddings_data:
            return {
                "tool_id": tool_id,
                "embedding_count": 0,
                "has_embeddings": False,
                "stats": {}
            }
        
        # Deserialize embeddings
        embeddings = self.embedding_store.deserialize_embeddings(embeddings_data)
        
        # Get embedding statistics
        stats = self.embedding_store.get_embedding_stats(embeddings)
        
        return {
            "tool_id": tool_id,
            "embedding_count": embedding_count,
            "has_embeddings": True,
            "stats": stats
        }

