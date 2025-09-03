"""
Intelligent Tool Selector - LLM-based semantic tool selection
Uses LLM to understand task intent and match with tool capabilities
"""

import json
from typing import List, Dict, Any, Tuple, Optional
from ..models.tool import Tool
from ..llm.manager import LLMManager
from ..llm.roles import LLMRole
from ..logging.config import get_logger


class IntelligentToolSelector:
    """
    Intelligent tool selection using LLM semantic understanding
    
    This system replaces keyword-based matching with deep semantic analysis:
    - Understands user intent and task semantics
    - Analyzes tool capabilities and constraints  
    - Makes intelligent matches based on context
    - Supports natural language task descriptions
    """
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.logger = get_logger(__name__)
        
        # Cache for tool capability analysis
        self.tool_capabilities_cache = {}
        
    async def select_best_tools(
        self, 
        task_description: str, 
        context: Dict[str, Any],
        available_tools: List[Tool],
        max_tools: int = 5
    ) -> List[Tuple[Tool, float, str]]:
        """
        Select best tools for a task using intelligent semantic analysis
        
        Args:
            task_description: Natural language task description
            context: Additional context information
            available_tools: List of available tools
            max_tools: Maximum number of tools to return
            
        Returns:
            List of (tool, confidence_score, reasoning) tuples
        """
        self.logger.info(f"Intelligent selection for task: {task_description[:100]}...")
        
        # Step 1: Analyze task semantics
        task_analysis = await self._analyze_task_semantics(task_description, context)
        
        # Step 2: Pre-filter tools by action alignment (for performance with many tools)
        pre_filtered_tools = self._pre_filter_by_action(available_tools, task_analysis)
        
        # Step 3: Analyze tool capabilities for pre-filtered tools
        # For large tool sets (100+ tools), limit LLM analysis to top candidates
        tools_to_analyze = pre_filtered_tools
        if len(pre_filtered_tools) > 20:
            # Sort by success rate and take top candidates for deep analysis
            tools_to_analyze = sorted(pre_filtered_tools, key=lambda t: t.success_rate, reverse=True)[:20]
            self.logger.info(f"Large tool set detected ({len(pre_filtered_tools)}), analyzing top 20 by success rate")
        
        tool_analyses = []
        for tool in tools_to_analyze:
            capability_analysis = await self._analyze_tool_capabilities(tool)
            tool_analyses.append((tool, capability_analysis))
        
        # Step 3: Match tools to task using LLM
        matches = await self._match_tools_to_task(
            task_analysis, 
            tool_analyses, 
            max_tools
        )
        
        self.logger.info(f"Selected {len(matches)} tools with confidence scores")
        return matches
    
    async def _analyze_task_semantics(
        self, 
        task_description: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze task semantics using LLM to understand user intent and translate to English
        """
        # First, translate natural language to standardized English if needed
        normalized_task = await self._normalize_task_to_english(task_description)
        
        prompt = f"""Analyze this task and extract semantic information:

Original Task: "{task_description}"
Normalized Task (English): "{normalized_task}"
Context: {json.dumps(context, indent=2)}

Please analyze and return JSON with:
{{
    "primary_intent": "What is the main goal? (e.g., 'create_content', 'analyze_data', 'search_information', 'process_files', 'system_operation')",
    "action_type": "What type of action? (e.g., 'create', 'read', 'update', 'delete', 'search', 'transform', 'analyze')",
    "target_objects": ["What objects/entities are involved? (e.g., 'file', 'text', 'directory', 'data', 'system')"],
    "required_capabilities": ["What capabilities are needed? (e.g., 'filesystem_write', 'text_processing', 'pattern_matching', 'data_analysis')"],
    "input_type": "What type of input is expected? (e.g., 'text', 'file_path', 'pattern', 'data')",
    "output_type": "What type of output is expected? (e.g., 'file', 'text', 'analysis', 'list', 'boolean')",
    "complexity_level": "Task complexity (1-5)",
    "urgency": "Task urgency (1-5)",
    "risk_level": "Risk level (1-5, where 5 is high risk like deletion)",
    "additional_requirements": ["Any special requirements or constraints"]
}}

Focus on understanding the TRUE intent behind the natural language, not just keywords.
Use the normalized English task for analysis while keeping the original for context."""

        try:
            response = await self.llm_manager.generate_text_for_role(
                role=LLMRole.PLANNER_JUDGE,
                prompt=prompt,
                temperature=0.2,
                max_tokens=800
            )
            
            # Parse JSON response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            analysis = json.loads(content)
            return analysis
            
        except Exception as e:
            self.logger.error(f"Task analysis failed: {e}")
            # Fallback to simple analysis
            return {
                "primary_intent": "unknown",
                "action_type": "unknown", 
                "target_objects": [],
                "required_capabilities": [],
                "input_type": "text",
                "output_type": "text",
                "complexity_level": 3,
                "urgency": 3,
                "risk_level": 1,
                "additional_requirements": []
            }
    
    async def _normalize_task_to_english(self, task_description: str) -> str:
        """
        Normalize natural language task to standardized English
        """
        # Check if task is already in English (simple heuristic)
        import re
        if re.search(r'[а-яё]', task_description.lower()):
            # Contains Cyrillic, likely Russian
            return await self._translate_to_english(task_description)
        else:
            # Likely already English, just normalize
            return task_description.strip()
    
    async def _translate_to_english(self, text: str) -> str:
        """
        Translate non-English text to standardized English using LLM
        """
        prompt = f"""Translate this task description to clear, standardized English. Focus on the ACTION and INTENT, not literal translation.

Original text: "{text}"

Translate to English focusing on:
1. What ACTION the user wants to perform (create, analyze, search, etc.)
2. What OBJECT they want to work with (file, document, data, etc.)
3. Any specific REQUIREMENTS or CONTEXT

Return ONLY the English translation, no explanations."""

        try:
            response = await self.llm_manager.generate_text_for_role(
                role=LLMRole.PLANNER_JUDGE,
                prompt=prompt,
                temperature=0.2,
                max_tokens=200
            )
            
            english_text = response.content.strip()
            self.logger.info(f"Translated '{text[:50]}...' to '{english_text[:50]}...'")
            return english_text
            
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            # Fallback: return original text
            return text
    
    def _pre_filter_by_action(self, tools: List[Tool], task_analysis: Dict[str, Any]) -> List[Tool]:
        """
        Pre-filter tools by action type for performance (especially with 100+ MCP servers)
        """
        required_action = task_analysis.get("action_type", "").lower()
        if not required_action:
            return tools  # No filtering if action type unclear
        
        # Action keyword mappings for fast filtering
        action_keywords = {
            "create": ["create", "write", "make", "generate", "build", "new"],
            "read": ["read", "show", "display", "cat", "view", "get"],
            "search": ["search", "find", "grep", "query", "locate"],
            "analyze": ["analyze", "count", "wc", "stat", "measure", "check"],
            "delete": ["delete", "remove", "rm", "clean"],
            "update": ["update", "modify", "edit", "change", "append"],
            "list": ["list", "ls", "dir", "enumerate"]
        }
        
        relevant_keywords = action_keywords.get(required_action, [required_action])
        
        # Fast filtering by tool name and description
        filtered_tools = []
        for tool in tools:
            tool_text = f"{tool.name} {tool.description}".lower()
            
            # Check if tool matches required action
            for keyword in relevant_keywords:
                if keyword in tool_text:
                    filtered_tools.append(tool)
                    break
        
        # If no tools match, return all tools (fallback to LLM decision)
        if not filtered_tools:
            self.logger.warning(f"No tools matched action '{required_action}', returning all tools")
            return tools
        
        self.logger.info(f"Pre-filtered {len(tools)} tools to {len(filtered_tools)} based on action '{required_action}'")
        return filtered_tools
    
    async def _analyze_tool_capabilities(self, tool: Tool) -> Dict[str, Any]:
        """
        Analyze tool capabilities using LLM understanding
        """
        # Check cache first
        cache_key = f"{tool.id}_{tool.name}"
        if cache_key in self.tool_capabilities_cache:
            return self.tool_capabilities_cache[cache_key]
        
        prompt = f"""Analyze this tool and describe its capabilities in detail:

Tool Name: {tool.name}
Description: {tool.description}
Input Schema: {json.dumps(tool.input_schema, indent=2) if tool.input_schema else 'Not available'}
Capabilities: {tool.capabilities}
Tags: {tool.tags}

Please analyze and return JSON with:
{{
    "primary_function": "Main purpose of this tool (e.g., 'text_output', 'file_creation', 'data_analysis', 'search', 'system_query')",
    "action_types": ["What actions can it perform? (e.g., 'create', 'read', 'search', 'count', 'display')"],
    "input_requirements": {{
        "required_params": ["List of required parameters"],
        "optional_params": ["List of optional parameters"],
        "input_types": ["What types of input it accepts"]
    }},
    "output_capabilities": {{
        "output_format": "What format does it output? (e.g., 'text', 'file', 'structured_data')",
        "output_content": "What content does it produce?"
    }},
    "use_cases": ["When should this tool be used? List specific scenarios"],
    "limitations": ["What can't this tool do?"],
    "risk_level": "Safety risk level (1-5)",
    "performance": {{
        "speed": "Expected speed (fast/medium/slow)",
        "reliability": "Reliability level (high/medium/low)"
    }},
    "best_for": ["What types of tasks is this tool BEST suited for?"]
}}

Be very specific about when this tool should and shouldn't be used."""

        try:
            response = await self.llm_manager.generate_text_for_role(
                role=LLMRole.INGESTOR,
                prompt=prompt,
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse JSON response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            analysis = json.loads(content)
            
            # Cache the result
            self.tool_capabilities_cache[cache_key] = analysis
            return analysis
            
        except Exception as e:
            self.logger.error(f"Tool capability analysis failed for {tool.name}: {e}")
            # Fallback to simple analysis
            return {
                "primary_function": "unknown",
                "action_types": [],
                "input_requirements": {"required_params": [], "optional_params": [], "input_types": []},
                "output_capabilities": {"output_format": "text", "output_content": "unknown"},
                "use_cases": [],
                "limitations": [],
                "risk_level": 1,
                "performance": {"speed": "medium", "reliability": "medium"},
                "best_for": []
            }
    
    async def _match_tools_to_task(
        self,
        task_analysis: Dict[str, Any],
        tool_analyses: List[Tuple[Tool, Dict[str, Any]]],
        max_tools: int
    ) -> List[Tuple[Tool, float, str]]:
        """
        Match tools to task using LLM intelligent reasoning
        """
        # Prepare tools summary for LLM
        tools_summary = []
        for tool, analysis in tool_analyses:
            tools_summary.append({
                "tool_name": tool.name,
                "tool_id": tool.id,
                "primary_function": analysis.get("primary_function", "unknown"),
                "action_types": analysis.get("action_types", []),
                "use_cases": analysis.get("use_cases", []),
                "best_for": analysis.get("best_for", []),
                "input_requirements": analysis.get("input_requirements", {}),
                "output_capabilities": analysis.get("output_capabilities", {}),
                "risk_level": analysis.get("risk_level", 1),
                "limitations": analysis.get("limitations", [])
            })
        
        # Let LLM analyze task type and agent suitability without hardcoded rules
        task_guidance = """
INTELLIGENT AGENT SELECTION GUIDANCE:
- Consider each agent's role and professional focus when matching to tasks
- Full-Stack Developers can handle both frontend and backend work
- Frontend specialists are better for UI-focused tasks
- Backend specialists are better for server/API-focused tasks  
- File creation tools are good for simple static content
- Code generation agents are better for interactive/dynamic applications
- Match the complexity of the tool to the complexity of the task
"""

        prompt = f"""You are an expert tool selection AI. Your goal is to select tools that can ACTUALLY ACCOMPLISH the user's task, not just tools that are generally good.

TASK ANALYSIS:
{json.dumps(task_analysis, indent=2)}

AVAILABLE TOOLS:
{json.dumps(tools_summary, indent=2)}

{task_guidance}

CRITICAL RULE: The task's PRIMARY ACTION TYPE must match the tool's PRIMARY FUNCTION. 
- If task = "create" → ONLY select tools that CREATE things
- If task = "read/analyze" → ONLY select tools that READ/ANALYZE things  
- If task = "search" → ONLY select tools that SEARCH things
- If task = "delete" → ONLY select tools that DELETE things

Please select the {max_tools} best tools for this task and return JSON with:
{{
    "selected_tools": [
        {{
            "tool_name": "exact tool name from the list",
            "confidence_score": 0.95,
            "reasoning": "WHY this tool's PRIMARY FUNCTION directly accomplishes the task's PRIMARY ACTION",
            "action_alignment": "How this tool's action matches the required task action",
            "task_completion": "Exactly what part of the task this tool will complete"
        }}
    ],
    "reasoning_summary": "Why these tools were selected based on ACTION ALIGNMENT"
}}

SELECTION CRITERIA (in priority order):
1. **ACTION ALIGNMENT** (MOST IMPORTANT) - Can this tool perform the EXACT action the task requires?
2. **TASK COMPLETION** - Will this tool directly accomplish the user's goal?
3. **INPUT/OUTPUT MATCH** - Does it accept the right inputs and produce the right outputs?
4. **USE CASE RELEVANCE** - Is this task in the tool's optimal use cases?

SCORING GUIDE:
- 0.9-1.0: Perfect action alignment + task completion
- 0.7-0.8: Good action alignment + partial task completion  
- 0.5-0.6: Some relevance but not primary function
- 0.1-0.4: Tool is capable but not designed for this action
- 0.0: Tool cannot perform the required action

FORBIDDEN: Do NOT select tools just because they are "high quality" or "well-designed" if they don't match the required action!

Example: If task is "create file", do NOT select "word_count" tool even if it's excellent at counting words - it cannot CREATE files!"""

        try:
            response = await self.llm_manager.generate_text_for_role(
                role=LLMRole.PLANNER_JUDGE,
                prompt=prompt,
                temperature=0.3,
                max_tokens=1500
            )
            
            # Parse JSON response with robust error handling
            content = response.content.strip()
            
            # Extract JSON from response (handle various formats)
            import re
            
            # Try to find JSON block in markdown
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
            elif content.startswith("```") and content.endswith("```"):
                # Remove code block markers
                content = content[3:-3].strip()
            
            # Try to extract only the JSON part (find first { to last })
            json_start = content.find('{')
            json_end = content.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                content = content[json_start:json_end + 1]
            
            try:
                selection_result = json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing failed: {e}")
                self.logger.error(f"Raw content: {repr(content[:500])}...")
                # Try to parse line by line to find valid JSON
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line.startswith('{'):
                        try:
                            # Try to parse from this line to end
                            remaining_content = '\n'.join(lines[i:])
                            json_end_in_remaining = remaining_content.rfind('}')
                            if json_end_in_remaining != -1:
                                json_candidate = remaining_content[:json_end_in_remaining + 1]
                                selection_result = json.loads(json_candidate)
                                break
                        except json.JSONDecodeError:
                            continue
                else:
                    # If all parsing fails, create fallback result
                    self.logger.warning("Using fallback selection due to JSON parsing failure")
                    selection_result = {
                        "selected_tools": [
                            {
                                "tool_name": tool_analyses[0][0].name if tool_analyses else "unknown",
                                "confidence_score": 0.5,
                                "reasoning": "Fallback selection due to JSON parsing error"
                            }
                        ]
                    }
            
            # Convert to result format
            matches = []
            for selection in selection_result.get("selected_tools", []):
                tool_name = selection.get("tool_name", "")
                confidence = selection.get("confidence_score", 0.5)
                reasoning = selection.get("reasoning", "No reasoning provided")
                
                # Find the actual tool object
                for tool, _ in tool_analyses:
                    if tool.name == tool_name:
                        matches.append((tool, confidence, reasoning))
                        break
            
            return matches[:max_tools]
            
        except Exception as e:
            self.logger.error(f"Tool matching failed: {e}")
            # Fallback to first few tools
            return [(tool, 0.5, "Fallback selection") for tool, _ in tool_analyses[:max_tools]]
    
    async def explain_selection(
        self, 
        task_description: str, 
        selected_tools: List[Tuple[Tool, float, str]]
    ) -> str:
        """
        Generate human-readable explanation of tool selection
        """
        explanations = []
        for tool, confidence, reasoning in selected_tools:
            explanations.append(f"**{tool.name}** (confidence: {confidence:.2f}): {reasoning}")
        
        return f"For task '{task_description}':\n\n" + "\n\n".join(explanations)
    
    async def extract_arguments_for_tool(
        self,
        task_description: str,
        context: Dict[str, Any],
        tool: Tool
    ) -> Dict[str, Any]:
        """
        Extract tool arguments using LLM intelligent analysis
        """
        # First, normalize task to English
        normalized_task = await self._normalize_task_to_english(task_description)
        
        # Get tool schema for reference (fetch full tool data if schema is missing)
        input_schema = tool.input_schema
        if not input_schema:
            # Tool loaded from list endpoint doesn't have schema, fetch full details
            try:
                from ..storage.database import DatabaseManager
                # This is a workaround - in production we'd inject the db_manager
                # For now, we'll use the fallback extraction
                input_schema = {}
            except:
                input_schema = {}
        
        input_schema = input_schema or {}
        required_params = []
        optional_params = []
        
        if input_schema and "properties" in input_schema:
            for param_name, param_info in input_schema["properties"].items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                param_entry = f"{param_name} ({param_type}): {param_desc}"
                
                if param_name in input_schema.get("required", []):
                    required_params.append(param_entry)
                else:
                    optional_params.append(param_entry)
        
        prompt = f"""Extract arguments for tool execution from natural language task.

TASK:
Original: "{task_description}"
Normalized: "{normalized_task}"
Context: {json.dumps(context, indent=2)}

TOOL INFORMATION:
Tool Name: {tool.name}
Description: {tool.description}

REQUIRED PARAMETERS:
{chr(10).join(required_params) if required_params else "None specified"}

OPTIONAL PARAMETERS:
{chr(10).join(optional_params) if optional_params else "None specified"}

EXTRACTION RULES:
1. Extract EXACT values needed by the tool from the task description
2. For file paths: extract specific file names mentioned in task
3. For content: extract what should be written/created
4. For patterns: extract search terms or patterns
5. For text: extract specific text to process
6. Use normalized task for extraction, but consider original for context

Return JSON with extracted arguments:
{{
    "path": "extracted file path if mentioned",
    "content": "extracted content if mentioned", 
    "text": "extracted text if mentioned",
    "pattern": "extracted search pattern if mentioned",
    "query": "extracted search query if mentioned",
    "other_params": {{"param_name": "value"}}
}}

IMPORTANT:
- Only include parameters that have actual values from the task
- If file path not explicitly mentioned, suggest logical name based on task
- If content not specified, suggest appropriate content based on task intent
- Return empty object {{}} if no arguments can be extracted

Examples:
Task: "create file report.txt with system status"
Result: {{"path": "report.txt", "content": "system status"}}

Task: "search for ERROR in logs"  
Result: {{"pattern": "ERROR", "text": "logs"}}"""

        try:
            response = await self.llm_manager.generate_text_for_role(
                role=LLMRole.PLANNER_JUDGE,
                prompt=prompt,
                temperature=0.2,
                max_tokens=800
            )
            
            # Parse JSON response with robust handling
            content = response.content.strip()
            
            # Extract JSON using the same robust method as tool selection
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
            elif content.startswith("```") and content.endswith("```"):
                content = content[3:-3].strip()
            
            json_start = content.find('{')
            json_end = content.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                content = content[json_start:json_end + 1]
            
            try:
                arguments = json.loads(content)
                
                # Flatten other_params into main dict
                if "other_params" in arguments and isinstance(arguments["other_params"], dict):
                    arguments.update(arguments["other_params"])
                    del arguments["other_params"]
                
                # Remove empty values
                arguments = {k: v for k, v in arguments.items() if v and v != ""}
                
                self.logger.info(f"Extracted arguments for {tool.name}: {arguments}")
                return arguments
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Argument extraction JSON parsing failed: {e}")
                self.logger.error(f"Raw content: {repr(content[:200])}...")
                # Fallback to simple extraction
                return self._simple_argument_extraction(normalized_task, tool)
                
        except Exception as e:
            self.logger.error(f"Argument extraction failed: {e}")
            return self._simple_argument_extraction(normalized_task, tool)
    
    def _simple_argument_extraction(self, task: str, tool: Tool) -> Dict[str, Any]:
        """
        Simple fallback argument extraction
        """
        args = {}
        task_lower = task.lower()
        tool_name = tool.name.lower()
        
        # Extract based on common patterns
        import re
        
        if "create_file" in tool_name:
            # Try to extract file path and content
            path_match = re.search(r'(?:create|file)\s+(\S+\.?\w*)', task)
            if path_match:
                args["path"] = path_match.group(1)
            else:
                # Suggest file name based on task content
                if "document" in task_lower:
                    args["path"] = "document.txt"
                elif "report" in task_lower:
                    args["path"] = "report.txt"
                elif "technical" in task_lower:
                    args["path"] = "technical_specs.txt"
                else:
                    args["path"] = "file.txt"
            
            # Extract content
            content_match = re.search(r'(?:with|content|containing)\s+(.+)', task)
            if content_match:
                args["content"] = content_match.group(1).strip()
            else:
                # Generate content based on task
                if "technical" in task_lower and "specification" in task_lower:
                    args["content"] = "Technical Specifications\n\n[Content to be filled]"
                elif "document" in task_lower:
                    args["content"] = "Document Content\n\n[Content to be filled]"
                else:
                    args["content"] = task
        
        elif "echo" in tool_name:
            args["text"] = task
        
        elif "grep" in tool_name or "search" in tool_name:
            # Extract search pattern
            pattern_match = re.search(r'(?:search|find|pattern)\s+(?:for\s+)?(\w+)', task)
            if pattern_match:
                args["pattern"] = pattern_match.group(1)
                args["text"] = task
        
        elif "wc" in tool_name:
            args["text"] = task
        
        # If no specific args, use task as goal
        if not args:
            args["goal"] = task
        
        return args
