"""
Stage 1: The Sieve - Fast lexical-declarative filtering
Implements instant filtering using SQL operations (FTS5, tags, capabilities, schema compatibility)
"""

import time
from typing import List, Dict, Any, Optional, Set
from ..storage.database import DatabaseManager
from ..models.tool import Tool


class SieveStage:
    """
    Stage 1: The Sieve - Fast filtering (< 1ms)
    
    Performs instant filtering using:
    - Full-text search (FTS5)
    - Tag matching
    - Capability filtering
    - Schema compatibility checks
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.max_candidates = 50  # Maximum candidates to return
    
    async def filter_candidates(
        self, 
        goal: str, 
        context: Dict[str, Any],
        max_candidates: Optional[int] = None
    ) -> List[Tool]:
        """
        Fast filtering of tool candidates
        
        Args:
            goal: Task goal/description
            context: Additional context (capabilities, constraints, etc.)
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of filtered tool candidates
        """
        start_time = time.time()
        
        # Extract filtering criteria from context
        required_capabilities = context.get("required_capabilities", [])
        forbidden_capabilities = context.get("forbidden_capabilities", [])
        required_tags = context.get("required_tags", [])
        forbidden_tags = context.get("forbidden_tags", [])
        
        # Step 1: Full-text search on goal
        fts_candidates = await self._full_text_search(goal, max_candidates or self.max_candidates)
        
        # Step 2: Filter by capabilities
        capability_filtered = self._filter_by_capabilities(
            fts_candidates, 
            required_capabilities, 
            forbidden_capabilities
        )
        
        # Step 3: Filter by tags
        tag_filtered = self._filter_by_tags(
            capability_filtered,
            required_tags,
            forbidden_tags
        )
        
        # Step 4: Schema compatibility check
        schema_filtered = self._filter_by_schema_compatibility(
            tag_filtered,
            context.get("input_schema", {})
        )
        
        # Step 5: Apply final limits and ranking
        final_candidates = self._apply_final_ranking(schema_filtered, goal)
        
        # If no candidates found, check if query might be non-English
        if not final_candidates:
            import re
            # Check for non-Latin characters (like Cyrillic)
            if re.search(r'[^\x00-\x7F]', goal):
                print("SieveStage: Non-English query detected, returning all tools for intelligent evaluation")
                final_candidates = self.db_manager.get_all_tools()  # Get all tools for LLM evaluation
            else:
                print("SieveStage: No candidates found for English query")
        
        # Limit results
        limit = max_candidates or self.max_candidates
        final_candidates = final_candidates[:limit]
        
        # Log performance
        elapsed_ms = (time.time() - start_time) * 1000
        print(f"SieveStage: Filtered {len(final_candidates)} candidates in {elapsed_ms:.2f}ms")
        
        return final_candidates
    
    async def _full_text_search(self, goal: str, limit: int) -> List[Tool]:
        """
        Perform full-text search using FTS5
        
        Args:
            goal: Search query
            limit: Maximum results
            
        Returns:
            List of tools matching the search query
        """
        # Extract keywords from goal
        keywords = self._extract_keywords(goal)
        
        if not keywords:
            # Fallback to simple search
            return self.db_manager.search_tools(goal, limit)
        
        # Build FTS5 query
        fts_query = " OR ".join(f'"{keyword}"' for keyword in keywords)
        
        # Search using FTS5 with proper syntax
        try:
            from ..storage.schema import get_connection
            with get_connection(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                
                # Use proper FTS5 MATCH syntax without alias in WHERE clause
                cursor.execute("""
                    SELECT t.* FROM tools t
                    JOIN tool_fts ON t.id = tool_fts.tool_id
                    WHERE tool_fts MATCH ?
                    ORDER BY bm25(tool_fts) ASC
                    LIMIT ?
                """, (fts_query, limit))
                
                rows = cursor.fetchall()
                tools = []
                for row in rows:
                    tool = self.db_manager._row_to_tool(row)
                    if tool:
                        tools.append(tool)
                
                print(f"FTS5 search found {len(tools)} tools for query: {fts_query}")
                if tools:
                    print(f"FTS5 found tools: {[tool.name for tool in tools[:3]]}")
                return tools
                
        except Exception as e:
            print(f"FTS5 search failed, using fallback: {e}")
            return self.db_manager.search_tools(goal, limit)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from text with intelligent task type detection
        
        Args:
            text: Input text (multilingual)
            
        Returns:
            List of keywords
        """
        import re
        
        # Simple universal keywords for FTS5 search
        # LLM will do intelligent matching, FTS5 just needs broad coverage
        universal_keywords = [
            "create", "generate", "build", "develop", "make",
            "code", "script", "application", "web", "game", 
            "file", "data", "content", "tool", "service"
        ]
        
        text_lower = text.lower()
        enhanced_keywords = []
        
        # Add universal keywords that match the task intent
        if any(action in text_lower for action in ["создай", "create", "make", "build", "generate"]):
            enhanced_keywords.extend(["create", "generate", "build"])
        if any(tech in text_lower for tech in ["html", "web", "game", "игра"]):
            enhanced_keywords.extend(["web", "application", "content"])
        if any(code in text_lower for code in ["code", "script", "программ", "код"]):
            enhanced_keywords.extend(["code", "script", "tool"])
        
        # Extract English words only (original logic as fallback)
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
            "of", "with", "by", "from", "up", "about", "into", "through", "during",
            "before", "after", "above", "below", "between", "among", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might", "must", "can",
            "using", "use", "create", "make", "get", "set"
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
        english_keywords = [word for word in words if word not in stop_words]
        
        # Combine enhanced and extracted keywords
        all_keywords = list(set(enhanced_keywords + english_keywords))
        
        return all_keywords[:15]  # Increased limit for better coverage
    
    def _filter_by_capabilities(
        self, 
        tools: List[Tool], 
        required: List[str], 
        forbidden: List[str]
    ) -> List[Tool]:
        """
        Filter tools by capability requirements
        
        Args:
            tools: List of tools to filter
            required: Required capabilities
            forbidden: Forbidden capabilities
            
        Returns:
            Filtered list of tools
        """
        if not required and not forbidden:
            return tools
        
        filtered = []
        for tool in tools:
            tool_capabilities = set(tool.capabilities)
            
            # Check required capabilities
            if required:
                required_set = set(required)
                if not required_set.issubset(tool_capabilities):
                    continue
            
            # Check forbidden capabilities
            if forbidden:
                forbidden_set = set(forbidden)
                if forbidden_set.intersection(tool_capabilities):
                    continue
            
            filtered.append(tool)
        
        return filtered
    
    def _filter_by_tags(
        self, 
        tools: List[Tool], 
        required: List[str], 
        forbidden: List[str]
    ) -> List[Tool]:
        """
        Filter tools by tag requirements
        
        Args:
            tools: List of tools to filter
            required: Required tags
            forbidden: Forbidden tags
            
        Returns:
            Filtered list of tools
        """
        if not required and not forbidden:
            return tools
        
        filtered = []
        for tool in tools:
            tool_tags = set(tool.tags)
            
            # Check required tags
            if required:
                required_set = set(required)
                if not required_set.intersection(tool_tags):
                    continue
            
            # Check forbidden tags
            if forbidden:
                forbidden_set = set(forbidden)
                if forbidden_set.intersection(tool_tags):
                    continue
            
            filtered.append(tool)
        
        return filtered
    
    def _filter_by_schema_compatibility(
        self, 
        tools: List[Tool], 
        input_schema: Dict[str, Any]
    ) -> List[Tool]:
        """
        Filter tools by input schema compatibility
        
        Args:
            tools: List of tools to filter
            input_schema: Required input schema
            
        Returns:
            Filtered list of tools
        """
        if not input_schema:
            return tools
        
        filtered = []
        for tool in tools:
            if self._is_schema_compatible(tool.input_schema, input_schema):
                filtered.append(tool)
        
        return filtered
    
    def _is_schema_compatible(
        self, 
        tool_schema: Dict[str, Any], 
        required_schema: Dict[str, Any]
    ) -> bool:
        """
        Check if tool schema is compatible with required schema
        
        Args:
            tool_schema: Tool's input schema
            required_schema: Required input schema
            
        Returns:
            True if compatible, False otherwise
        """
        # Simple compatibility check
        # In a full implementation, this would use JSON Schema validation
        
        tool_properties = tool_schema.get("properties", {})
        required_properties = required_schema.get("properties", {})
        
        # Check if all required properties exist in tool schema
        for prop_name, prop_schema in required_properties.items():
            if prop_name not in tool_properties:
                return False
            
            # Check type compatibility
            tool_prop_type = tool_properties[prop_name].get("type")
            required_prop_type = prop_schema.get("type")
            
            if tool_prop_type and required_prop_type:
                if not self._are_types_compatible(tool_prop_type, required_prop_type):
                    return False
        
        return True
    
    def _are_types_compatible(self, tool_type: str, required_type: str) -> bool:
        """
        Check if two JSON Schema types are compatible
        
        Args:
            tool_type: Tool's property type
            required_type: Required property type
            
        Returns:
            True if compatible, False otherwise
        """
        # Simple type compatibility matrix
        compatibility = {
            "string": ["string"],
            "number": ["number", "integer"],
            "integer": ["integer", "number"],
            "boolean": ["boolean"],
            "array": ["array"],
            "object": ["object"]
        }
        
        return required_type in compatibility.get(tool_type, [])
    
    def _apply_final_ranking(self, tools: List[Tool], goal: str) -> List[Tool]:
        """
        Apply final ranking based on basic heuristics
        
        Args:
            tools: List of tools to rank
            goal: Task goal for ranking
            
        Returns:
            Ranked list of tools
        """
        # Simple ranking based on:
        # 1. Success rate (higher is better)
        # 2. Lower latency (lower is better)
        # 3. Number of matching tags (more is better)
        
        goal_keywords = set(self._extract_keywords(goal))
        
        def rank_tool(tool: Tool) -> float:
            # Success rate score (0-1)
            success_score = tool.success_rate
            
            # Latency score (inverted, 0-1)
            latency_score = max(0, 1 - (tool.p95_latency_ms / 10000))  # Normalize to 10s max
            
            # Tag relevance score
            tool_tags = set(tool.tags)
            tag_score = len(goal_keywords.intersection(tool_tags)) / max(len(goal_keywords), 1)
            
            # Weighted combination
            final_score = (
                success_score * 0.4 +      # 40% weight on success rate
                latency_score * 0.3 +      # 30% weight on latency
                tag_score * 0.3            # 30% weight on tag relevance
            )
            
            return final_score
        
        # Sort by ranking score
        ranked_tools = sorted(tools, key=rank_tool, reverse=True)
        
        return ranked_tools
    
    def get_filtering_stats(self) -> Dict[str, Any]:
        """
        Get statistics about filtering performance
        
        Returns:
            Dictionary with filtering statistics
        """
        return {
            "max_candidates": self.max_candidates,
            "stage": "sieve",
            "description": "Fast lexical-declarative filtering"
        }
