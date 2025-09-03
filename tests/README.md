# RMCP Tests

This directory contains all test files for the RMCP project.

## Test Categories

### Unit Tests
- `test_basic_tools.py` - Basic tool functionality tests
- `test_llm_integration.py` - LLM provider integration tests

### Integration Tests  
- `test_e2e_simple.py` - End-to-end simple workflow tests
- `test_ecosystem_e2e.py` - Full ecosystem integration tests
- `test_meta_orchestration.py` - Meta-orchestration system tests

### Agent Tests
- `test_agent_simple.py` - Basic agent functionality
- `test_all_agents.py` - Comprehensive agent testing
- `test_architect_agent.py` - Architecture agent tests
- `test_backend_agent.py` - Backend agent tests

### System Tests
- `test_fractal_orchestration.py` - Fractal scaling tests
- `test_brigade_orchestrator.py` - Brigade coordination tests
- `test_orchestrator_logic.py` - Core orchestration logic
- `test_prometheus_experiment.py` - Prometheus engine tests

### Performance Tests
- `test_live_prometheus.py` - Live performance monitoring
- `test_comprehensive_system_diagnosis.py` - System diagnostics

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_e2e_simple.py

# Run with coverage
pytest tests/ --cov=rmcp --cov-report=html
```