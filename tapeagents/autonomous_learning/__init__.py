"""
Autonomous Learning Module for TapeAgents

This module implements autonomous task generation and optimization inspired by ETO
(Exploration-based Trajectory Optimization). It enables TapeAgents to learn and improve
in environments without explicit instructions, using only data sources and tools.

Key Components:
- SeedTaskManager: Handles initial task input and generation
- TrajectoryOptimizer: Implements ETO-style optimization for TapeAgents
- MemoryWarmer: Populates Tape memory with optimized trajectories
- AutonomousTaskGenerator: Generates new tasks based on past interactions
- EnvironmentLearner: Main orchestrator for autonomous learning
"""

from .seed_task_manager import SeedTaskManager
from .trajectory_optimizer import TrajectoryOptimizer
from .memory_warmer import MemoryWarmer
from .task_generator import AutonomousTaskGenerator
from .environment_learner import EnvironmentLearner
from .datatypes import (
    SeedTask,
    TrajectoryPair,
    OptimizedTrajectory,
    TaskGenerationContext,
    LearningMetrics
)

__all__ = [
    "SeedTaskManager",
    "TrajectoryOptimizer", 
    "MemoryWarmer",
    "AutonomousTaskGenerator",
    "EnvironmentLearner",
    "SeedTask",
    "TrajectoryPair",
    "OptimizedTrajectory",
    "TaskGenerationContext",
    "LearningMetrics"
]