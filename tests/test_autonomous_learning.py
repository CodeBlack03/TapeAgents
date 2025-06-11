"""
Tests for the autonomous learning module.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock

from tapeagents.agent import Agent, Node
from tapeagents.core import Prompt, SetNextNode
from tapeagents.dialog_tape import AssistantStep, DialogTape, UserStep
from tapeagents.llms import MockLLM
from tapeagents.prompting import tape_to_messages

from tapeagents.autonomous_learning import (
    SeedTaskManager,
    TrajectoryOptimizer,
    MemoryWarmer,
    AutonomousTaskGenerator,
    EnvironmentLearner,
    SeedTask,
    TaskCategory,
    TaskDifficulty,
    TrajectoryOutcome,
    OptimizedTrajectory,
    EnvironmentState,
    TaskGenerationContext,
    LearningMetrics
)


class TestNode(Node):
    """Test node for autonomous learning tests."""
    name: str = "test"

    def make_prompt(self, agent: Agent, tape: DialogTape) -> Prompt:
        return Prompt(messages=tape_to_messages(tape))

    def generate_steps(self, agent, tape, llm_stream):
        yield AssistantStep(content=llm_stream.get_text())
        yield SetNextNode(next_node="test")


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = MockLLM(model_name="test-model")
    llm.cached_responses = [
        "This is a test response for autonomous learning.",
        "Another test response with different content.",
        "A third response for testing purposes."
    ]
    return llm


@pytest.fixture
def test_agent(mock_llm):
    """Create a test agent."""
    return Agent[DialogTape].create(mock_llm, nodes=[TestNode()])


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestSeedTaskManager:
    """Test the SeedTaskManager component."""

    def test_add_seed_task_from_description(self, temp_storage):
        """Test adding a seed task from description."""
        manager = SeedTaskManager(storage_path=str(Path(temp_storage) / "tasks.json"))
        
        task = manager.add_seed_task_from_description(
            description="Test task for data analysis",
            environment_context={"tools": ["analyzer"]},
            success_criteria=["Complete analysis"]
        )
        
        assert task.description == "Test task for data analysis"
        assert task.category == TaskCategory.DATA_ANALYSIS
        assert task.environment_context == {"tools": ["analyzer"]}
        assert task.success_criteria == ["Complete analysis"]
        assert len(manager.seed_tasks) == 1

    def test_generate_task_variations(self, temp_storage):
        """Test generating task variations."""
        manager = SeedTaskManager(storage_path=str(Path(temp_storage) / "tasks.json"))
        
        base_task = manager.add_seed_task_from_description(
            description="Base task for testing",
            category=TaskCategory.EXPLORATION,
            difficulty=TaskDifficulty.INTERMEDIATE
        )
        
        variations = manager.generate_task_variations(base_task, num_variations=3)
        
        assert len(variations) == 3
        assert all(var.metadata["base_task_id"] == base_task.id for var in variations)
        assert variations[0].difficulty == TaskDifficulty.BEGINNER  # Easier
        assert variations[1].difficulty == TaskDifficulty.ADVANCED  # Harder

    def test_analyze_environment_from_tasks(self, temp_storage):
        """Test environment analysis from tasks."""
        manager = SeedTaskManager(storage_path=str(Path(temp_storage) / "tasks.json"))
        
        # Add tasks with different tool mentions
        manager.add_seed_task_from_description("Use calculator for math")
        manager.add_seed_task_from_description("Search the web for information")
        manager.add_seed_task_from_description("Read files from database")
        
        env_state = manager.analyze_environment_from_tasks()
        
        assert "calculator" in env_state.available_tools
        assert "search" in env_state.available_tools
        assert "database" in env_state.data_sources

    def test_save_and_load_tasks(self, temp_storage):
        """Test saving and loading tasks."""
        storage_path = str(Path(temp_storage) / "tasks.json")
        manager = SeedTaskManager(storage_path=storage_path)
        
        # Add some tasks
        task1 = manager.add_seed_task_from_description("Task 1")
        task2 = manager.add_seed_task_from_description("Task 2")
        
        # Save tasks
        manager.save_seed_tasks()
        
        # Create new manager and load
        new_manager = SeedTaskManager(storage_path=storage_path)
        
        assert len(new_manager.seed_tasks) == 2
        assert new_manager.seed_tasks[0].description == "Task 1"
        assert new_manager.seed_tasks[1].description == "Task 2"


class TestTrajectoryOptimizer:
    """Test the TrajectoryOptimizer component."""

    def test_initialization(self, test_agent, mock_llm):
        """Test trajectory optimizer initialization."""
        optimizer = TrajectoryOptimizer(
            agent=test_agent,
            llm=mock_llm,
            max_exploration_rounds=3,
            trajectories_per_round=5
        )
        
        assert optimizer.agent == test_agent
        assert optimizer.llm == mock_llm
        assert optimizer.max_exploration_rounds == 3
        assert optimizer.trajectories_per_round == 5

    def test_create_initial_tape(self, test_agent, mock_llm):
        """Test creating initial tape for tasks."""
        optimizer = TrajectoryOptimizer(agent=test_agent, llm=mock_llm)
        
        task = SeedTask(
            id="test_task",
            description="Test task description",
            category=TaskCategory.EXPLORATION,
            difficulty=TaskDifficulty.BEGINNER
        )
        
        tape = optimizer._create_initial_tape(task)
        
        assert isinstance(tape, DialogTape)
        assert len(tape.steps) == 1
        assert tape.steps[0].content == "Test task description"

    def test_evaluate_trajectory(self, test_agent, mock_llm):
        """Test trajectory evaluation."""
        optimizer = TrajectoryOptimizer(agent=test_agent, llm=mock_llm)
        
        task = SeedTask(
            id="test_task",
            description="Test task",
            category=TaskCategory.EXPLORATION,
            difficulty=TaskDifficulty.BEGINNER,
            success_criteria=["complete", "success"]
        )
        
        # Test successful trajectory
        success_tape = DialogTape(steps=[
            UserStep(content="Test task"),
            AssistantStep(content="Task completed successfully")
        ])
        
        outcome, reward = optimizer._evaluate_trajectory(success_tape, task)
        
        assert outcome == TrajectoryOutcome.SUCCESS
        assert reward > 0.5

    def test_create_trajectory_pairs(self, test_agent, mock_llm):
        """Test creating trajectory pairs for contrastive learning."""
        optimizer = TrajectoryOptimizer(agent=test_agent, llm=mock_llm)
        
        # Create mock trajectories
        success_tape = DialogTape(steps=[
            UserStep(content="Task"),
            AssistantStep(content="Success")
        ])
        
        failure_tape = DialogTape(steps=[
            UserStep(content="Task")
        ])
        
        trajectories = [
            OptimizedTrajectory(
                id="success_1",
                original_task_id="task_1",
                tape=success_tape,
                outcome=TrajectoryOutcome.SUCCESS,
                reward=0.8,
                optimization_round=1,
                confidence_score=0.7
            ),
            OptimizedTrajectory(
                id="failure_1",
                original_task_id="task_1",
                tape=failure_tape,
                outcome=TrajectoryOutcome.FAILURE,
                reward=0.2,
                optimization_round=1,
                confidence_score=0.3
            )
        ]
        
        pairs = optimizer._create_trajectory_pairs(trajectories)
        
        assert len(pairs) == 1
        assert pairs[0].task_id == "task_1"
        assert pairs[0].success_reward == 0.8
        assert pairs[0].failure_reward == 0.2


class TestMemoryWarmer:
    """Test the MemoryWarmer component."""

    def test_initialization(self, test_agent):
        """Test memory warmer initialization."""
        warmer = MemoryWarmer(
            agent=test_agent,
            max_memory_size=500,
            similarity_threshold=0.8
        )
        
        assert warmer.agent == test_agent
        assert warmer.max_memory_size == 500
        assert warmer.similarity_threshold == 0.8

    def test_organize_trajectories_by_task(self, test_agent):
        """Test organizing trajectories by task characteristics."""
        warmer = MemoryWarmer(agent=test_agent)
        
        # Create test data
        trajectories = [
            OptimizedTrajectory(
                id="traj_1",
                original_task_id="task_1",
                tape=DialogTape(steps=[]),
                outcome=TrajectoryOutcome.SUCCESS,
                reward=0.8,
                optimization_round=1,
                confidence_score=0.7
            )
        ]
        
        seed_tasks = [
            SeedTask(
                id="task_1",
                description="Test task",
                category=TaskCategory.DATA_ANALYSIS,
                difficulty=TaskDifficulty.BEGINNER
            )
        ]
        
        warmer._organize_trajectories_by_task(trajectories, seed_tasks)
        
        assert "task_task_1" in warmer.memory_bank
        assert "category_data_analysis" in warmer.memory_bank
        assert "difficulty_beginner" in warmer.memory_bank
        assert "outcome_success" in warmer.memory_bank

    def test_retrieve_relevant_memory(self, test_agent):
        """Test retrieving relevant memory."""
        warmer = MemoryWarmer(agent=test_agent)
        
        # Add some trajectories to memory
        trajectory = OptimizedTrajectory(
            id="traj_1",
            original_task_id="task_1",
            tape=DialogTape(steps=[
                UserStep(content="analyze data"),
                AssistantStep(content="data analysis complete")
            ]),
            outcome=TrajectoryOutcome.SUCCESS,
            reward=0.8,
            optimization_round=1,
            confidence_score=0.7
        )
        
        warmer.memory_bank["category_data_analysis"].append(trajectory)
        
        # Retrieve relevant memory
        relevant = warmer.retrieve_relevant_memory(
            task_description="analyze data files",
            task_category=TaskCategory.DATA_ANALYSIS,
            max_results=5
        )
        
        assert len(relevant) >= 1
        assert relevant[0].id == "traj_1"

    def test_memory_cleanup(self, test_agent):
        """Test memory cleanup functionality."""
        warmer = MemoryWarmer(agent=test_agent, max_memory_size=2)
        
        # Add trajectories that exceed memory limit
        for i in range(5):
            trajectory = OptimizedTrajectory(
                id=f"traj_{i}",
                original_task_id=f"task_{i}",
                tape=DialogTape(steps=[]),
                outcome=TrajectoryOutcome.SUCCESS,
                reward=0.5,
                optimization_round=1,
                confidence_score=0.5
            )
            warmer.memory_bank["test"].append(trajectory)
            warmer.memory_quality_scores[f"traj_{i}"] = 0.1  # Low quality
            warmer.memory_usage_stats[f"traj_{i}"] = 0  # No usage
        
        cleanup_stats = warmer.cleanup_memory(force_cleanup=True)
        
        assert cleanup_stats["removed"] > 0
        assert cleanup_stats["remaining"] <= warmer.max_memory_size


class TestAutonomousTaskGenerator:
    """Test the AutonomousTaskGenerator component."""

    def test_initialization(self, mock_llm):
        """Test task generator initialization."""
        generator = AutonomousTaskGenerator(
            llm=mock_llm,
            max_tasks_per_generation=5,
            exploration_bias=0.4
        )
        
        assert generator.llm == mock_llm
        assert generator.max_tasks_per_generation == 5
        assert generator.exploration_bias == 0.4

    def test_generate_exploration_tasks(self, mock_llm):
        """Test generating exploration tasks."""
        generator = AutonomousTaskGenerator(llm=mock_llm)
        
        environment_state = EnvironmentState(
            available_tools=["tool1", "tool2", "tool3"],
            data_sources=["source1", "source2"]
        )
        
        generator.unexplored_areas = ["tool1", "tool2"]
        
        tasks = generator._generate_exploration_tasks(environment_state, num_tasks=2)
        
        assert len(tasks) == 2
        assert all(task.category == TaskCategory.EXPLORATION for task in tasks)
        assert all(task.difficulty == TaskDifficulty.BEGINNER for task in tasks)

    def test_generate_progressive_tasks(self, mock_llm):
        """Test generating progressive task sequences."""
        generator = AutonomousTaskGenerator(llm=mock_llm)
        
        base_task = SeedTask(
            id="base_task",
            description="Base task for progression",
            category=TaskCategory.PROBLEM_SOLVING,
            difficulty=TaskDifficulty.BEGINNER,
            success_criteria=["complete task"]
        )
        
        progressive_tasks = generator.generate_progressive_tasks(
            base_task=base_task,
            progression_steps=3
        )
        
        assert len(progressive_tasks) == 3
        assert progressive_tasks[0].difficulty == TaskDifficulty.BEGINNER
        assert progressive_tasks[1].difficulty == TaskDifficulty.INTERMEDIATE
        assert progressive_tasks[2].difficulty == TaskDifficulty.ADVANCED

    def test_task_generation_with_context(self, mock_llm):
        """Test task generation with learning context."""
        generator = AutonomousTaskGenerator(llm=mock_llm)
        
        context = TaskGenerationContext(
            completed_tasks=["task1", "task2"],
            successful_patterns=["pattern1", "pattern2"],
            failure_patterns=["failure1"],
            learning_gaps=["gap1", "gap2"],
            current_difficulty=TaskDifficulty.INTERMEDIATE
        )
        
        environment_state = EnvironmentState(
            available_tools=["tool1", "tool2"],
            data_sources=["source1"]
        )
        
        metrics = LearningMetrics(
            total_tasks_attempted=10,
            total_tasks_completed=7,
            success_rate=0.7
        )
        
        tasks = generator.generate_tasks(
            context=context,
            environment_state=environment_state,
            learning_metrics=metrics,
            num_tasks=3
        )
        
        assert len(tasks) <= 3
        assert all(isinstance(task, SeedTask) for task in tasks)


class TestEnvironmentLearner:
    """Test the EnvironmentLearner main orchestrator."""

    def test_initialization(self, test_agent, mock_llm, temp_storage):
        """Test environment learner initialization."""
        learner = EnvironmentLearner(
            agent=test_agent,
            llm=mock_llm,
            storage_path=temp_storage,
            max_learning_rounds=5
        )
        
        assert learner.agent == test_agent
        assert learner.llm == mock_llm
        assert learner.max_learning_rounds == 5
        assert learner.storage_path == Path(temp_storage)

    def test_start_learning_session(self, test_agent, mock_llm, temp_storage):
        """Test starting a learning session."""
        learner = EnvironmentLearner(
            agent=test_agent,
            llm=mock_llm,
            storage_path=temp_storage
        )
        
        seed_tasks = [
            "Learn to use file tools",
            "Practice web searching",
            "Analyze data patterns"
        ]
        
        environment_context = {
            "available_tools": ["file_reader", "web_search", "analyzer"],
            "data_sources": ["files", "web"]
        }
        
        session = learner.start_learning_session(
            seed_task_descriptions=seed_tasks,
            environment_context=environment_context,
            session_id="test_session"
        )
        
        assert session.session_id == "test_session"
        assert len(session.seed_tasks) == 3
        assert len(session.environment_state.available_tools) >= 3

    def test_add_seed_tasks_during_learning(self, test_agent, mock_llm, temp_storage):
        """Test adding seed tasks during active learning."""
        learner = EnvironmentLearner(
            agent=test_agent,
            llm=mock_llm,
            storage_path=temp_storage
        )
        
        # Start session
        session = learner.start_learning_session(
            seed_task_descriptions=["Initial task"],
            session_id="test_session"
        )
        
        initial_count = len(session.seed_tasks)
        
        # Add new tasks
        new_tasks = learner.add_seed_tasks_during_learning([
            "New task 1",
            "New task 2"
        ])
        
        assert len(new_tasks) == 2
        assert len(session.seed_tasks) == initial_count + 2

    def test_get_learning_progress(self, test_agent, mock_llm, temp_storage):
        """Test getting learning progress."""
        learner = EnvironmentLearner(
            agent=test_agent,
            llm=mock_llm,
            storage_path=temp_storage
        )
        
        # Start session
        learner.start_learning_session(
            seed_task_descriptions=["Test task"],
            session_id="test_session"
        )
        
        progress = learner.get_learning_progress()
        
        assert "session_id" in progress
        assert "current_metrics" in progress
        assert "environment_state" in progress
        assert progress["session_id"] == "test_session"

    def test_create_task_generation_context(self, test_agent, mock_llm, temp_storage):
        """Test creating task generation context."""
        learner = EnvironmentLearner(
            agent=test_agent,
            llm=mock_llm,
            storage_path=temp_storage
        )
        
        # Start session
        learner.start_learning_session(
            seed_task_descriptions=["Test task"],
            session_id="test_session"
        )
        
        # Add some optimization insights
        learner.trajectory_optimizer.optimization_insights = [
            "Success factor: good planning",
            "Failure factor: poor execution"
        ]
        
        context = learner._create_task_generation_context()
        
        assert isinstance(context, TaskGenerationContext)
        assert len(context.successful_patterns) > 0
        assert len(context.failure_patterns) > 0


def test_integration_workflow(test_agent, mock_llm, temp_storage):
    """Test the complete integration workflow."""
    # Create environment learner
    learner = EnvironmentLearner(
        agent=test_agent,
        llm=mock_llm,
        storage_path=temp_storage,
        max_learning_rounds=2,  # Reduced for testing
        tasks_per_round=2,
        optimization_rounds_per_cycle=1
    )
    
    # Start learning session
    session = learner.start_learning_session(
        seed_task_descriptions=[
            "Learn basic file operations",
            "Practice simple calculations"
        ],
        environment_context={
            "available_tools": ["file_reader", "calculator"],
            "data_sources": ["files", "user_input"]
        },
        session_id="integration_test"
    )
    
    assert session is not None
    assert len(session.seed_tasks) == 2
    
    # Test individual components work
    progress = learner.get_learning_progress()
    assert progress["session_id"] == "integration_test"
    
    # Test task generation
    new_tasks = learner.generate_next_tasks(num_tasks=2)
    assert len(new_tasks) <= 2
    
    # Test session summary
    summary = learner.get_session_summary()
    assert summary["session_id"] == "integration_test"
    assert summary["seed_tasks_count"] == 2


if __name__ == "__main__":
    pytest.main([__file__])