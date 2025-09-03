"""
Tests for Strategic Planning functionality
"""

import pytest
import tempfile
import os
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path

from rmcp.planning.strategic_three_stage import StrategicThreeStagePlanner
from rmcp.planning.strategic_compass import StrategicCompassStage
from rmcp.planning.strategic_judge import StrategicJudgeStage
from rmcp.storage.database import DatabaseManager
from rmcp.storage.schema import init_database
from rmcp.models.tool import Tool, Agent
from rmcp.models.plan import ExecutionPlan, ExecutionStrategy, ExecutionStep
from rmcp.agents.registry import AgentRegistry


class TestStrategicCompassStage:
    """Test Strategic Compass Stage functionality"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        init_database(db_path)
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def db_manager(self, temp_db):
        """Create database manager"""
        return DatabaseManager(temp_db)
    
    @pytest.fixture
    def mock_embedding_manager(self):
        """Create mock embedding manager"""
        manager = Mock()
        manager.encode_text = AsyncMock(return_value=[0.1] * 384)
        return manager
    
    @pytest.fixture
    def mock_embedding_store(self):
        """Create mock embedding store"""
        store = Mock()
        store.calculate_affinity_score = AsyncMock(return_value=0.8)
        store.deserialize_embeddings = Mock(return_value=[[0.1] * 384])
        return store
    
    @pytest.fixture
    def strategic_compass(self, db_manager, mock_embedding_manager, mock_embedding_store):
        """Create strategic compass stage"""
        return StrategicCompassStage(db_manager, mock_embedding_manager, mock_embedding_store)
    
    @pytest.fixture
    def test_agent(self):
        """Create test agent"""
        return Tool(
            id="test-agent-1",
            server_id="agent-server-1",
            name="Security Auditor Agent",
            description="Autonomous agent for security auditing",
            tool_type="agent",
            capabilities=["security_audit", "vulnerability_scan"],
            tags=["security", "audit"]
        )
    
    @pytest.fixture
    def test_atomic_tool(self):
        """Create test atomic tool"""
        return Tool(
            id="test-tool-1",
            server_id="server-1",
            name="grep",
            description="Search text in files",
            tool_type="atomic",
            capabilities=["text_search", "file_search"],
            tags=["search", "text"]
        )
    
    def test_analyze_task_abstraction_level(self, strategic_compass):
        """Test task abstraction level analysis"""
        # High-level strategic task
        high_level_goal = "Conduct comprehensive security audit of our Terraform infrastructure"
        assert strategic_compass._analyze_task_abstraction_level(high_level_goal) == "high"
        
        # Medium-level task
        medium_level_goal = "Review the authentication module for security issues"
        assert strategic_compass._analyze_task_abstraction_level(medium_level_goal) == "medium"
        
        # Low-level tactical task
        low_level_goal = "Find all files containing the word 'password'"
        assert strategic_compass._analyze_task_abstraction_level(low_level_goal) == "low"
    
    def test_extract_task_specializations(self, strategic_compass):
        """Test task specialization extraction"""
        # Security task
        security_goal = "Audit our infrastructure for security vulnerabilities and compliance issues"
        specializations = strategic_compass._extract_task_specializations(security_goal)
        assert "security" in specializations
        
        # Deployment task
        deployment_goal = "Deploy the new authentication service to production using blue-green deployment"
        specializations = strategic_compass._extract_task_specializations(deployment_goal)
        assert "deployment" in specializations
        
        # Testing task
        testing_goal = "Run comprehensive unit and integration tests for the new API"
        specializations = strategic_compass._extract_task_specializations(testing_goal)
        assert "testing" in specializations
        
        # Multiple specializations
        multi_goal = "Deploy and test the security module in production"
        specializations = strategic_compass._extract_task_specializations(multi_goal)
        assert "deployment" in specializations
        assert "testing" in specializations
    
    def test_calculate_abstraction_boost(self, strategic_compass, test_agent, test_atomic_tool):
        """Test abstraction level boost calculation"""
        # Add abstraction level to test agent
        test_agent.abstraction_level = "high"
        test_atomic_tool.abstraction_level = "low"
        
        # Perfect match
        boost = strategic_compass._calculate_abstraction_boost(test_agent, "high")
        assert boost == strategic_compass.abstraction_boost_factor
        
        # Adjacent levels
        boost = strategic_compass._calculate_abstraction_boost(test_agent, "medium")
        assert boost == strategic_compass.abstraction_boost_factor * 0.5
        
        # Too far apart
        boost = strategic_compass._calculate_abstraction_boost(test_agent, "low")
        assert boost == 0.0
    
    def test_calculate_specialization_boost(self, strategic_compass, test_agent):
        """Test specialization boost calculation"""
        # Add specialization to test agent
        test_agent.specialization = "security"
        
        # Perfect match
        boost = strategic_compass._calculate_specialization_boost(test_agent, ["security"])
        assert boost == strategic_compass.specialization_boost_factor
        
        # No match
        boost = strategic_compass._calculate_specialization_boost(test_agent, ["deployment"])
        assert boost == 0.0
        
        # No specializations
        boost = strategic_compass._calculate_specialization_boost(test_agent, [])
        assert boost == 0.0
    
    def test_calculate_agent_preference_boost(self, strategic_compass, test_agent, test_atomic_tool):
        """Test agent preference boost calculation"""
        # Agent gets boost
        boost = strategic_compass._calculate_agent_preference_boost(test_agent)
        assert boost == strategic_compass.agent_preference_factor
        
        # Atomic tool gets no boost
        boost = strategic_compass._calculate_agent_preference_boost(test_atomic_tool)
        assert boost == 0.0
    
    @pytest.mark.asyncio
    async def test_strategic_ranking(self, strategic_compass, test_agent, test_atomic_tool):
        """Test strategic ranking of candidates"""
        # Add agent attributes
        test_agent.abstraction_level = "high"
        test_agent.specialization = "security"
        test_atomic_tool.abstraction_level = "low"
        
        candidates = [test_agent, test_atomic_tool]
        goal = "Conduct comprehensive security audit of our infrastructure"
        context = {}
        
        # Mock the base affinity calculation
        with patch.object(strategic_compass, '_calculate_affinity_score', new_callable=AsyncMock) as mock_affinity:
            mock_affinity.return_value = 0.5
            
            ranked = await strategic_compass.rank_candidates(goal, context, candidates)
            
            # Should have 2 candidates
            assert len(ranked) == 2
            
            # Agent should be ranked higher due to strategic boosts
            agent_score = next(score for tool, score in ranked if tool.tool_type == "agent")
            atomic_score = next(score for tool, score in ranked if tool.tool_type == "atomic")
            
            assert agent_score > atomic_score


class TestStrategicJudgeStage:
    """Test Strategic Judge Stage functionality"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        init_database(db_path)
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def db_manager(self, temp_db):
        """Create database manager"""
        return DatabaseManager(temp_db)
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Create mock LLM manager"""
        manager = Mock()
        manager.generate_response = AsyncMock(return_value="""
        {
          "strategy": "SOLO",
          "steps": [
            {
              "tool_id": "test-agent-1",
              "tool_name": "Security Auditor Agent",
              "parameters": {"goal": "audit security"},
              "estimated_duration_ms": 30000,
              "requires_approval": true
            }
          ],
          "estimated_duration_ms": 30000,
          "requires_approval": true,
          "complexity_score": 0.8
        }
        """)
        return manager
    
    @pytest.fixture
    def strategic_judge(self, db_manager, mock_llm_manager):
        """Create strategic judge stage"""
        return StrategicJudgeStage(db_manager, mock_llm_manager)
    
    @pytest.fixture
    def test_agent(self):
        """Create test agent"""
        agent = Tool(
            id="test-agent-1",
            server_id="agent-server-1",
            name="Security Auditor Agent",
            description="Autonomous agent for security auditing",
            tool_type="agent",
            capabilities=["security_audit", "vulnerability_scan"],
            tags=["security", "audit"]
        )
        agent.abstraction_level = "high"
        agent.specialization = "security"
        agent.avg_execution_time_ms = 30000
        return agent
    
    @pytest.fixture
    def test_atomic_tool(self):
        """Create test atomic tool"""
        return Tool(
            id="test-tool-1",
            server_id="server-1",
            name="grep",
            description="Search text in files",
            tool_type="atomic",
            capabilities=["text_search", "file_search"],
            tags=["search", "text"]
        )
    
    def test_analyze_strategic_context(self, strategic_judge, test_agent, test_atomic_tool):
        """Test strategic context analysis"""
        goal = "Conduct comprehensive security audit of our Terraform infrastructure"
        candidates = [(test_agent, 0.9), (test_atomic_tool, 0.6)]
        
        context = strategic_judge._analyze_strategic_context(goal, candidates)
        
        assert context['agent_count'] == 1
        assert context['atomic_tool_count'] == 1
        assert context['strategic_keyword_count'] >= 2
        assert context['top_agent'] == test_agent
        assert context['top_atomic'] == test_atomic_tool
        assert context['has_high_abstraction_agents'] == True
        assert context['has_specialized_agents'] == True
    
    def test_should_use_agent_delegation(self, strategic_judge, test_agent, test_atomic_tool):
        """Test agent delegation decision"""
        goal = "Conduct comprehensive security audit of our infrastructure"
        candidates = [(test_agent, 0.9), (test_atomic_tool, 0.6)]
        context = strategic_judge._analyze_strategic_context(goal, candidates)
        
        # Should use agent delegation
        assert strategic_judge._should_use_agent_delegation(goal, candidates, context) == True
        
        # Test with low-scoring agent
        low_score_candidates = [(test_agent, 0.5), (test_atomic_tool, 0.6)]
        low_context = strategic_judge._analyze_strategic_context(goal, low_score_candidates)
        assert strategic_judge._should_use_agent_delegation(goal, low_score_candidates, low_context) == False
    
    def test_should_use_strategic_orchestration(self, strategic_judge, test_agent, test_atomic_tool):
        """Test strategic orchestration decision"""
        goal = "Deploy and test the security module"
        candidates = [(test_agent, 0.8), (test_atomic_tool, 0.7)]
        context = strategic_judge._analyze_strategic_context(goal, candidates)
        
        # Should use strategic orchestration
        assert strategic_judge._should_use_strategic_orchestration(goal, candidates, context) == True
        
        # Test with only agents
        agent_only_candidates = [(test_agent, 0.8)]
        agent_only_context = strategic_judge._analyze_strategic_context(goal, agent_only_candidates)
        assert strategic_judge._should_use_strategic_orchestration(goal, agent_only_candidates, agent_only_context) == False
    
    def test_requires_approval_for_agent(self, strategic_judge, test_agent):
        """Test approval requirement for agents"""
        # High-level agent should require approval
        goal = "Audit security"
        assert strategic_judge._requires_approval_for_agent(test_agent, goal) == True
        
        # Production-related goal should require approval
        production_goal = "Deploy to production"
        assert strategic_judge._requires_approval_for_agent(test_agent, production_goal) == True
        
        # Simple goal might not require approval
        simple_goal = "Check status"
        # This depends on the agent's abstraction level
        assert strategic_judge._requires_approval_for_agent(test_agent, simple_goal) == True  # High abstraction level
    
    def test_calculate_agent_complexity(self, strategic_judge, test_agent):
        """Test agent complexity calculation"""
        # Base complexity
        goal = "Audit security"
        complexity = strategic_judge._calculate_agent_complexity(test_agent, goal)
        assert 0.0 <= complexity <= 1.0
        
        # Comprehensive goal should increase complexity
        comprehensive_goal = "Conduct comprehensive security audit"
        complexity = strategic_judge._calculate_agent_complexity(test_agent, comprehensive_goal)
        assert complexity > 0.5
    
    @pytest.mark.asyncio
    async def test_create_agent_delegation_plan(self, strategic_judge, test_agent, test_atomic_tool):
        """Test agent delegation plan creation"""
        goal = "Conduct comprehensive security audit"
        context = {}
        candidates = [(test_agent, 0.9), (test_atomic_tool, 0.6)]
        strategic_context = strategic_judge._analyze_strategic_context(goal, candidates)
        
        plan = await strategic_judge._create_agent_delegation_plan(
            goal, context, candidates, strategic_context
        )
        
        assert plan.strategy == ExecutionStrategy.SOLO
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_id == test_agent.id
        assert plan.requires_approval == True  # High-level agent
        assert plan.metadata.get('complexity_score', 0.0) > 0.0
    
    @pytest.mark.asyncio
    async def test_create_strategic_execution_plan(self, strategic_judge, test_agent, test_atomic_tool):
        """Test strategic execution plan creation"""
        goal = "Conduct comprehensive security audit of our infrastructure"
        context = {}
        candidates = [(test_agent, 0.9), (test_atomic_tool, 0.6)]
        
        plan = await strategic_judge.create_execution_plan(goal, context, candidates)
        
        assert isinstance(plan, ExecutionPlan)
        assert plan.strategy in [ExecutionStrategy.SOLO, ExecutionStrategy.PARALLEL, ExecutionStrategy.DAG]
        assert len(plan.steps) >= 1
        assert plan.max_execution_time_ms > 0
        assert 0.0 <= plan.metadata.get('complexity_score', 0.0) <= 1.0


class TestStrategicThreeStagePlanner:
    """Test Strategic Three-Stage Planner integration"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        init_database(db_path)
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def db_manager(self, temp_db):
        """Create database manager"""
        return DatabaseManager(temp_db)
    
    @pytest.fixture
    def mock_embedding_manager(self):
        """Create mock embedding manager"""
        manager = Mock()
        manager.encode_text = AsyncMock(return_value=[0.1] * 384)
        return manager
    
    @pytest.fixture
    def mock_embedding_store(self):
        """Create mock embedding store"""
        store = Mock()
        store.calculate_affinity_score = AsyncMock(return_value=0.8)
        store.deserialize_embeddings = Mock(return_value=[[0.1] * 384])
        return store
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Create mock LLM manager"""
        manager = Mock()
        manager.generate_response = AsyncMock(return_value="""
        {
          "strategy": "SOLO",
          "steps": [
            {
              "tool_id": "test-agent-1",
              "tool_name": "Security Auditor Agent",
              "parameters": {"goal": "audit security"},
              "estimated_duration_ms": 30000,
              "requires_approval": true
            }
          ],
          "estimated_duration_ms": 30000,
          "requires_approval": true,
          "complexity_score": 0.8
        }
        """)
        return manager
    
    @pytest.fixture
    def strategic_planner(self, db_manager, mock_embedding_manager, mock_embedding_store, mock_llm_manager):
        """Create strategic three-stage planner"""
        return StrategicThreeStagePlanner(
            db_manager, mock_embedding_manager, mock_embedding_store, mock_llm_manager
        )
    
    @pytest.fixture
    def test_agent(self):
        """Create test agent"""
        agent = Tool(
            id="test-agent-1",
            server_id="agent-server-1",
            name="Security Auditor Agent",
            description="Autonomous agent for security auditing",
            tool_type="agent",
            capabilities=["security_audit", "vulnerability_scan"],
            tags=["security", "audit"]
        )
        agent.abstraction_level = "high"
        agent.specialization = "security"
        agent.avg_execution_time_ms = 30000
        return agent
    
    @pytest.fixture
    def test_atomic_tool(self):
        """Create test atomic tool"""
        return Tool(
            id="test-tool-1",
            server_id="server-1",
            name="grep",
            description="Search text in files",
            tool_type="atomic",
            capabilities=["text_search", "file_search"],
            tags=["search", "text"]
        )
    
    @pytest.mark.asyncio
    async def test_strategic_planning_high_level_task(self, strategic_planner, test_agent, test_atomic_tool):
        """Test strategic planning for high-level task"""
        # Add tools to database
        strategic_planner.db_manager.add_tool(test_agent)
        strategic_planner.db_manager.add_tool(test_atomic_tool)
        
        goal = "Conduct comprehensive security audit of our Terraform infrastructure"
        context = {"request_id": "test-123", "user_id": "user-1", "tenant_id": "tenant-1"}
        
        # Mock sieve stage to return our test tools
        with patch.object(strategic_planner.sieve, 'filter_candidates', new_callable=AsyncMock) as mock_sieve:
            mock_sieve.return_value = [test_agent, test_atomic_tool]
            
            plan = await strategic_planner.create_plan(goal, context)
            
            assert isinstance(plan, ExecutionPlan)
            assert plan.strategy in [ExecutionStrategy.SOLO, ExecutionStrategy.PARALLEL, ExecutionStrategy.DAG]
            assert len(plan.steps) >= 1
            assert plan.max_execution_time_ms > 0
            assert 0.0 <= plan.metadata.get('complexity_score', 0.0) <= 1.0
    
    @pytest.mark.asyncio
    async def test_strategic_planning_low_level_task(self, strategic_planner, test_agent, test_atomic_tool):
        """Test strategic planning for low-level task"""
        # Add tools to database
        strategic_planner.db_manager.add_tool(test_agent)
        strategic_planner.db_manager.add_tool(test_atomic_tool)
        
        goal = "Find all files containing the word 'password'"
        context = {"request_id": "test-456", "user_id": "user-1", "tenant_id": "tenant-1"}
        
        # Mock sieve stage to return our test tools
        with patch.object(strategic_planner.sieve, 'filter_candidates', new_callable=AsyncMock) as mock_sieve:
            mock_sieve.return_value = [test_agent, test_atomic_tool]
            
            plan = await strategic_planner.create_plan(goal, context)
            
            assert isinstance(plan, ExecutionPlan)
            # For low-level tasks, should prefer atomic tools
            assert plan.strategy in [ExecutionStrategy.SOLO, ExecutionStrategy.PARALLEL, ExecutionStrategy.DAG]
            assert len(plan.steps) >= 1
    
    @pytest.mark.asyncio
    async def test_strategic_planning_no_candidates(self, strategic_planner):
        """Test strategic planning with no candidates"""
        goal = "Some impossible task"
        context = {"request_id": "test-789", "user_id": "user-1", "tenant_id": "tenant-1"}
        
        # Mock sieve stage to return no candidates
        with patch.object(strategic_planner.sieve, 'filter_candidates', new_callable=AsyncMock) as mock_sieve:
            mock_sieve.return_value = []
            
            plan = await strategic_planner.create_plan(goal, context)
            
            assert isinstance(plan, ExecutionPlan)
            assert plan.strategy == ExecutionStrategy.SOLO
            assert len(plan.steps) == 0
            assert plan.max_execution_time_ms == 0
            assert plan.metadata.get('complexity_score', 0.0) == 0.0
    
    @pytest.mark.asyncio
    async def test_get_planning_statistics(self, strategic_planner, test_agent, test_atomic_tool):
        """Test planning statistics"""
        # Add tools to database
        strategic_planner.db_manager.add_tool(test_agent)
        strategic_planner.db_manager.add_tool(test_atomic_tool)
        
        stats = await strategic_planner.get_planning_statistics()
        
        assert stats['total_tools'] == 2
        assert stats['agents'] == 1
        assert stats['atomic_tools'] == 1
        assert stats['agent_ratio'] == 0.5
        assert stats['strategic_planning_enabled'] == True
        assert stats['agent_registry_enabled'] == True
    
    def test_configure_strategic_parameters(self, strategic_planner):
        """Test strategic parameter configuration"""
        strategic_planner.configure_strategic_parameters(
            abstraction_boost_factor=0.5,
            specialization_boost_factor=0.6,
            agent_preference_factor=0.3,
            agent_delegation_threshold=0.9
        )
        
        assert strategic_planner.compass.abstraction_boost_factor == 0.5
        assert strategic_planner.compass.specialization_boost_factor == 0.6
        assert strategic_planner.compass.agent_preference_factor == 0.3
        assert strategic_planner.judge.agent_delegation_threshold == 0.9
    
    def test_enable_strategic_mode(self, strategic_planner):
        """Test strategic mode enable/disable"""
        strategic_planner.enable_strategic_mode(False)
        assert strategic_planner.enable_strategic_planning == False
        
        strategic_planner.enable_strategic_mode(True)
        assert strategic_planner.enable_strategic_planning == True
