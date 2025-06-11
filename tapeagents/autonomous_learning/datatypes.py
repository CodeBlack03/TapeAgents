"""
Data types for autonomous learning module.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

from tapeagents.core import Tape, Step


class TaskDifficulty(str, Enum):
    """Task difficulty levels for progressive learning."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TaskCategory(str, Enum):
    """Categories of tasks for environment learning."""
    EXPLORATION = "exploration"
    DATA_ANALYSIS = "data_analysis"
    TOOL_USAGE = "tool_usage"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVE = "creative"
    INTEGRATION = "integration"


class SeedTask(BaseModel):
    """A seed task provided by the user to bootstrap learning."""
    
    id: str = Field(description="Unique identifier for the task")
    description: str = Field(description="Natural language description of the task")
    category: TaskCategory = Field(description="Category of the task")
    difficulty: TaskDifficulty = Field(description="Difficulty level of the task")
    environment_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context about the environment (tools, data sources, etc.)"
    )
    success_criteria: List[str] = Field(
        default_factory=list,
        description="Criteria for determining task success"
    )
    examples: List[str] = Field(
        default_factory=list,
        description="Example inputs or scenarios for the task"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the task"
    )


class TrajectoryOutcome(str, Enum):
    """Outcome of a trajectory execution."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ERROR = "error"


class TrajectoryPair(BaseModel):
    """A pair of trajectories for contrastive learning (success vs failure)."""
    
    id: str = Field(description="Unique identifier for the trajectory pair")
    task_id: str = Field(description="ID of the task this pair relates to")
    success_trajectory: Tape = Field(description="Successful trajectory")
    failure_trajectory: Tape = Field(description="Failed trajectory")
    success_reward: float = Field(description="Reward for successful trajectory")
    failure_reward: float = Field(description="Reward for failed trajectory")
    contrast_points: List[str] = Field(
        default_factory=list,
        description="Key differences between success and failure"
    )
    learning_insights: List[str] = Field(
        default_factory=list,
        description="Insights derived from the contrast"
    )


class OptimizedTrajectory(BaseModel):
    """An optimized trajectory after ETO-style learning."""
    
    id: str = Field(description="Unique identifier for the trajectory")
    original_task_id: str = Field(description="ID of the original task")
    tape: Tape = Field(description="The optimized trajectory tape")
    outcome: TrajectoryOutcome = Field(description="Outcome of the trajectory")
    reward: float = Field(description="Reward achieved by this trajectory")
    optimization_round: int = Field(description="Which optimization round this came from")
    improvements: List[str] = Field(
        default_factory=list,
        description="Specific improvements made during optimization"
    )
    confidence_score: float = Field(
        default=0.0,
        description="Confidence in the quality of this trajectory"
    )


class TaskGenerationContext(BaseModel):
    """Context for generating new tasks based on past learning."""
    
    completed_tasks: List[str] = Field(
        default_factory=list,
        description="IDs of completed tasks"
    )
    successful_patterns: List[str] = Field(
        default_factory=list,
        description="Patterns that led to success"
    )
    failure_patterns: List[str] = Field(
        default_factory=list,
        description="Patterns that led to failure"
    )
    environment_capabilities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Known capabilities of the environment"
    )
    learning_gaps: List[str] = Field(
        default_factory=list,
        description="Areas where more learning is needed"
    )
    current_difficulty: TaskDifficulty = Field(
        default=TaskDifficulty.BEGINNER,
        description="Current difficulty level to target"
    )


class LearningMetrics(BaseModel):
    """Metrics for tracking autonomous learning progress."""
    
    total_tasks_attempted: int = Field(default=0)
    total_tasks_completed: int = Field(default=0)
    success_rate: float = Field(default=0.0)
    average_reward: float = Field(default=0.0)
    improvement_rate: float = Field(default=0.0)
    
    # Task-specific metrics
    tasks_by_category: Dict[TaskCategory, int] = Field(default_factory=dict)
    tasks_by_difficulty: Dict[TaskDifficulty, int] = Field(default_factory=dict)
    
    # Learning progression
    optimization_rounds: int = Field(default=0)
    trajectory_pairs_generated: int = Field(default=0)
    memory_warmup_completions: int = Field(default=0)
    
    # Environment understanding
    tools_discovered: List[str] = Field(default_factory=list)
    data_sources_explored: List[str] = Field(default_factory=list)
    patterns_learned: List[str] = Field(default_factory=list)
    
    def update_success_rate(self):
        """Update the success rate based on completed vs attempted tasks."""
        if self.total_tasks_attempted > 0:
            self.success_rate = self.total_tasks_completed / self.total_tasks_attempted
    
    def add_task_attempt(self, category: TaskCategory, difficulty: TaskDifficulty, success: bool):
        """Record a task attempt."""
        self.total_tasks_attempted += 1
        if success:
            self.total_tasks_completed += 1
        
        # Update category counts
        if category not in self.tasks_by_category:
            self.tasks_by_category[category] = 0
        self.tasks_by_category[category] += 1
        
        # Update difficulty counts
        if difficulty not in self.tasks_by_difficulty:
            self.tasks_by_difficulty[difficulty] = 0
        self.tasks_by_difficulty[difficulty] += 1
        
        self.update_success_rate()


class EnvironmentState(BaseModel):
    """Current state of the environment being learned."""
    
    available_tools: List[str] = Field(
        default_factory=list,
        description="Tools available in the environment"
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources available in the environment"
    )
    discovered_capabilities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Capabilities discovered through exploration"
    )
    interaction_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of interactions with the environment"
    )
    current_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current context and state variables"
    )


class LearningSession(BaseModel):
    """A complete learning session with multiple optimization rounds."""
    
    session_id: str = Field(description="Unique identifier for the session")
    seed_tasks: List[SeedTask] = Field(description="Initial seed tasks")
    environment_state: EnvironmentState = Field(description="Environment state")
    optimization_rounds: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of optimization rounds"
    )
    generated_tasks: List[SeedTask] = Field(
        default_factory=list,
        description="Tasks generated during the session"
    )
    optimized_trajectories: List[OptimizedTrajectory] = Field(
        default_factory=list,
        description="All optimized trajectories from the session"
    )
    metrics: LearningMetrics = Field(
        default_factory=LearningMetrics,
        description="Learning metrics for the session"
    )
    insights: List[str] = Field(
        default_factory=list,
        description="Key insights learned during the session"
    )
    start_time: Optional[str] = Field(default=None)
    end_time: Optional[str] = Field(default=None)