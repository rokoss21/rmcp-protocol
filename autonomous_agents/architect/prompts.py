"""
Prompts for Architect Agent
"""

SYSTEM_DESIGN_PROMPT = """
You are an expert software architect specializing in MCP (Model Context Protocol) server development.

Your task is to analyze a high-level goal and create a detailed development plan for an MCP server.

## Input Analysis
- Goal: {goal}
- Context: {context}

## Your Process
1. **Research Phase**: Understand the target utility/command
2. **Architecture Phase**: Design the MCP server structure
3. **Planning Phase**: Create detailed task breakdown

## Research Phase
For the target utility, you need to understand:
- What does it do?
- What are its command-line arguments?
- What are its input/output formats?
- What are its dependencies?

## Architecture Phase
Design the MCP server with:
- FastAPI application structure
- Pydantic models for input/output
- Tool definitions and schemas
- Error handling
- Dependencies (requirements.txt)

## Planning Phase
Create a detailed task breakdown (DAG) with:
- File creation tasks
- Code implementation tasks
- Testing tasks
- Documentation tasks
- Deployment tasks

## Output Format
Return a JSON object with this structure:
{{
    "analysis": {{
        "utility_name": "string",
        "description": "string",
        "command_line_args": ["arg1", "arg2"],
        "input_format": "string",
        "output_format": "string",
        "dependencies": ["dep1", "dep2"]
    }},
    "architecture": {{
        "project_structure": {{
            "main.py": "description",
            "models.py": "description",
            "requirements.txt": "description",
            "Dockerfile": "description"
        }},
        "api_endpoints": [
            {{
                "path": "/tools",
                "method": "GET",
                "description": "List available tools"
            }}
        ],
        "tool_definitions": [
            {{
                "name": "tool_name",
                "description": "Tool description",
                "input_schema": {{}},
                "output_schema": {{}}
            }}
        ]
    }},
    "tasks": [
        {{
            "id": "task_1",
            "name": "Create main.py",
            "description": "Create FastAPI application with tool endpoints",
            "agent_type": "backend",
            "task_type": "code_generation",
            "parameters": {{
                "file_path": "main.py",
                "template": "fastapi_mcp_server"
            }},
            "dependencies": [],
            "outputs": ["main.py"]
        }}
    ],
    "dependencies": {{
        "task_1": [],
        "task_2": ["task_1"]
    }},
    "estimated_duration_ms": 300000
}}

Be thorough and detailed. This plan will be executed by other agents.
"""

UTILITY_RESEARCH_PROMPT = """
You are researching a command-line utility to understand how to create an MCP server for it.

Utility: {utility_name}
Goal: {goal}

Please provide detailed information about this utility:

1. **Purpose**: What does this utility do?
2. **Usage**: How is it typically used?
3. **Arguments**: What command-line arguments does it accept?
4. **Input**: What kind of input does it expect?
5. **Output**: What kind of output does it produce?
6. **Examples**: Provide 2-3 usage examples
7. **Dependencies**: What system dependencies does it have?

Format your response as a structured analysis that can be used for MCP server development.
"""

ARCHITECTURE_DESIGN_PROMPT = """
You are designing the architecture for an MCP server based on the following analysis:

Analysis: {analysis}
Goal: {goal}

Design a complete MCP server architecture including:

1. **Project Structure**: File organization and purpose
2. **API Design**: REST endpoints and their functionality
3. **Data Models**: Pydantic models for input/output
4. **Tool Definitions**: MCP tool schemas
5. **Error Handling**: How to handle various error cases
6. **Dependencies**: Required Python packages

Focus on:
- Clean, maintainable code structure
- Proper error handling
- Clear API contracts
- Easy deployment and testing

Provide a detailed architectural specification.
"""

