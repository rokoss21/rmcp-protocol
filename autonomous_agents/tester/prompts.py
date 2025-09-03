"""
Prompts for Tester Agent
"""

TEST_GENERATION_PROMPT = """
You are an expert Python test developer specializing in pytest and FastAPI testing.

Your task is to generate comprehensive, high-quality tests for the given code.

## Requirements
- Target File: {target_file}
- Test File: {test_file}
- Test Cases: {test_cases}
- Context: {context}

## Test Generation Guidelines
1. **Write comprehensive test coverage**
2. **Use pytest and FastAPI TestClient**
3. **Include both positive and negative test cases**
4. **Test edge cases and error conditions**
5. **Use proper fixtures and parametrization**
6. **Include proper assertions**
7. **Add descriptive test names and docstrings**
8. **Mock external dependencies when needed**

## Test Structure
- Import necessary modules
- Create test fixtures
- Test all endpoints and functions
- Test error handling
- Test edge cases
- Include integration tests

## Output Format
Return ONLY the complete Python test code. No explanations or markdown.

Generate the test file:
"""

FASTAPI_TEST_PROMPT = """
You are an expert in testing FastAPI applications with pytest.

Generate comprehensive tests for the following FastAPI application:

## Application Code
{application_code}

## Test Requirements
- Test File: {test_file}
- Test Cases: {test_cases}
- Context: {context}

## Test Guidelines
1. **Test all endpoints (GET, POST, etc.)**
2. **Test request/response validation**
3. **Test error handling (400, 500, etc.)**
4. **Test authentication if applicable**
5. **Test edge cases and boundary conditions**
6. **Use FastAPI TestClient**
7. **Mock external services**
8. **Include both unit and integration tests**

## Required Test Categories
- Health check tests
- Endpoint functionality tests
- Error handling tests
- Validation tests
- Edge case tests

## Output Format
Return ONLY the complete pytest test code. No explanations or markdown.

Generate the tests:
"""

PYDANTIC_MODEL_TEST_PROMPT = """
You are an expert in testing Pydantic models.

Generate comprehensive tests for the following Pydantic models:

## Model Code
{model_code}

## Test Requirements
- Test File: {test_file}
- Context: {context}

## Test Guidelines
1. **Test model validation**
2. **Test field types and constraints**
3. **Test default values**
4. **Test serialization/deserialization**
5. **Test error cases**
6. **Test edge cases**

## Output Format
Return ONLY the complete pytest test code. No explanations or markdown.

Generate the model tests:
"""

