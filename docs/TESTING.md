# RMCP Meta-Orchestration Testing Guide

This guide explains how to test the RMCP meta-orchestration system with real agents and MCP servers.

## Overview

The test environment includes:
- **RMCP Main Service**: The meta-orchestrator that plans and executes tasks
- **Mock MCP Server**: Provides atomic tools (grep, find, cat)
- **Mock Agent**: Provides AI agent capabilities (security audit, deployment, etc.)

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- curl (for health checks)

### Running E2E Tests

```bash
# Run the complete E2E test suite
python test_meta_orchestration.py
```

This will:
1. Start all services using docker-compose
2. Wait for services to be healthy
3. Run test scenarios
4. Validate results
5. Clean up the environment

### Manual Testing

#### Start the Environment

```bash
# Start all services
docker-compose -f docker-compose.test.yml up -d --build

# Check service health
curl http://localhost:8000/health  # Mock MCP Server
curl http://localhost:8001/health  # Mock Agent
curl http://localhost:8080/health  # RMCP Main
```

#### Test Agent Delegation

```bash
# Send high-level security audit request
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Conduct comprehensive security audit of our Terraform infrastructure",
    "context": {
      "repo_path": "/path/to/terraform",
      "environment": "production"
    },
    "user_id": "test-user",
    "tenant_id": "test-tenant"
  }'
```

#### Test Atomic Tool Execution

```bash
# Send low-level grep request
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Find all files containing the word error",
    "context": {
      "file_pattern": "*.py"
    },
    "user_id": "test-user",
    "tenant_id": "test-tenant"
  }'
```

#### Test Mixed Execution

```bash
# Send complex request that might use both agents and tools
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Deploy new version and run security tests",
    "context": {
      "version": "v1.2.3",
      "environment": "staging"
    },
    "user_id": "test-user",
    "tenant_id": "test-tenant"
  }'
```

### Cleanup

```bash
# Stop and remove all containers
docker-compose -f docker-compose.test.yml down -v
```

## Test Scenarios

### 1. Agent Delegation Test

**Purpose**: Verify that high-level tasks are delegated to appropriate agents.

**Request**: Security audit goal
**Expected**: Agent response with security-specific data (vulnerabilities, recommendations)

### 2. Atomic Tool Execution Test

**Purpose**: Verify that low-level tasks use atomic tools.

**Request**: File search goal
**Expected**: Tool response with search results

### 3. Mixed Execution Test

**Purpose**: Verify complex scenarios that might use both agents and tools.

**Request**: Deployment + testing goal
**Expected**: Coordinated response from multiple components

### 4. Service Health Test

**Purpose**: Verify all services are running and healthy.

**Expected**: All services respond to health checks

## Service Endpoints

### RMCP Main Service (Port 8080)

- `GET /health` - Health check
- `POST /execute` - Execute task (main endpoint)
- `GET /metrics` - Prometheus metrics
- `GET /docs` - API documentation

### Mock MCP Server (Port 8000)

- `GET /health` - Health check
- `GET /tools` - List available tools
- `POST /execute` - Execute MCP tool
- `GET /stats` - Server statistics

### Mock Agent (Port 8001)

- `GET /health` - Health check
- `POST /execute` - Execute agent task
- `GET /stats` - Agent statistics

## Configuration

The test configuration is in `config/test_config.yaml` and includes:

- MCP server definitions
- Agent registry
- Planning parameters
- Execution settings
- Observability configuration

## Troubleshooting

### Services Not Starting

```bash
# Check logs
docker-compose -f docker-compose.test.yml logs

# Check individual service logs
docker-compose -f docker-compose.test.yml logs rmcp
docker-compose -f docker-compose.test.yml logs mock-mcp-server
docker-compose -f docker-compose.test.yml logs mock-agent
```

### Health Check Failures

```bash
# Check if ports are accessible
netstat -tulpn | grep :8000
netstat -tulpn | grep :8001
netstat -tulpn | grep :8080

# Test endpoints manually
curl -v http://localhost:8000/health
curl -v http://localhost:8001/health
curl -v http://localhost:8080/health
```

### Database Issues

```bash
# Check database file permissions
ls -la /app/data/

# Remove old database
docker-compose -f docker-compose.test.yml down -v
```

## Development

### Adding New Test Scenarios

1. Add new test method to `MetaOrchestrationTester` class
2. Call the method in `_run_test_scenarios()`
3. Update this documentation

### Modifying Mock Services

- **Mock MCP Server**: Edit `mock_mcp_server/main.py`
- **Mock Agent**: Edit `mock_agent/main.py`
- **Configuration**: Edit `config/test_config.yaml`

### Adding New Tools/Agents

1. Add tool/agent definition to test configuration
2. Implement corresponding mock service
3. Update docker-compose configuration
4. Add test scenarios

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RMCP Main     │    │  Mock MCP       │    │   Mock Agent    │
│   (Port 8080)   │    │  Server         │    │   (Port 8001)   │
│                 │    │  (Port 8000)    │    │                 │
│  - Planning     │◄──►│  - grep         │    │  - Security     │
│  - Execution    │    │  - find         │    │  - Deployment   │
│  - Orchestration│    │  - cat          │    │  - Testing      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

The RMCP main service acts as a meta-orchestrator, deciding whether to:
- Use atomic tools from the MCP server for low-level tasks
- Delegate to agents for high-level, complex tasks
- Combine both approaches for mixed scenarios

