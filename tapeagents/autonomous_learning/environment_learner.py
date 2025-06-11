"""
Environment Learner - Main orchestrator for autonomous learning.

This module coordinates all components of the autonomous learning system,
implementing the complete learning pipeline from seed tasks to optimized performance.
"""

import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

from tapeagents.core import Tape
from tapeagents.agent import Agent
from tapeagents.llms import LLM

from .datatypes import (
    SeedTask, LearningSession, EnvironmentState, TaskGenerationContext,
    LearningMetrics, OptimizedTrajectory, TrajectoryPair, TaskCategory, TaskDifficulty
)
from .seed_task_manager import SeedTaskManager
from .trajectory_optimizer import TrajectoryOptimizer
from .memory_warmer import MemoryWarmer
from .task_generator import AutonomousTaskGenerator

logger = logging.getLogger(__name__)


class EnvironmentLearner:
    """
    Main orchestrator for autonomous learning in TapeAgents.
    
    The EnvironmentLearner coordinates the complete learning pipeline:
    1. Takes seed tasks from users
    2. Generates and optimizes trajectories using ETO methodology
    3. Warms up memory with optimized trajectories
    4. Generates new tasks based on learning progress
    5. Continuously improves performance through iterative learning
    
    This enables TapeAgents to learn and improve in environments with just
    data sources and tools, without explicit instructions.
    """
    
    def __init__(
        self,
        agent: Agent,
        llm: LLM,
        storage_path: Optional[str] = None,
        max_learning_rounds: int = 10,
        tasks_per_round: int = 5,
        optimization_rounds_per_cycle: int = 3,
        environment_executor: Optional[Callable] = None
    ):
        """
        Initialize the EnvironmentLearner.
        
        Args:
            agent: TapeAgent to train and improve
            llm: Language model for learning and generation
            storage_path: Path to store learning data
            max_learning_rounds: Maximum number of learning rounds
            tasks_per_round: Number of tasks to work on per round
            optimization_rounds_per_cycle: Number of optimization rounds per learning cycle
            environment_executor: Function to execute tasks in the environment
        """
        self.agent = agent
        self.llm = llm
        self.storage_path = Path(storage_path) if storage_path else Path("./autonomous_learning_data")
        self.max_learning_rounds = max_learning_rounds
        self.tasks_per_round = tasks_per_round
        self.optimization_rounds_per_cycle = optimization_rounds_per_cycle
        self.environment_executor = environment_executor or self._default_environment_executor
        
        # Initialize components
        self.seed_task_manager = SeedTaskManager(
            storage_path=str(self.storage_path / "seed_tasks.json")
        )
        self.trajectory_optimizer = TrajectoryOptimizer(
            agent=agent,
            llm=llm,
            max_exploration_rounds=optimization_rounds_per_cycle
        )
        self.memory_warmer = MemoryWarmer(agent=agent)
        self.task_generator = AutonomousTaskGenerator(llm=llm)
        
        # Learning state
        self.current_session: Optional[LearningSession] = None
        self.environment_state = EnvironmentState()
        self.learning_metrics = LearningMetrics()
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def start_learning_session(
        self,
        seed_task_descriptions: List[str],
        environment_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> LearningSession:
        """
        Start a new autonomous learning session.
        
        Args:
            seed_task_descriptions: Initial task descriptions from the user
            environment_context: Context about the environment (tools, data sources, etc.)
            session_id: Optional session ID (auto-generated if not provided)
            
        Returns:
            LearningSession object
        """
        session_id = session_id or f"session_{int(time.time())}"
        
        logger.info(f"Starting autonomous learning session: {session_id}")
        logger.info(f"Seed tasks: {len(seed_task_descriptions)}")
        
        # Create seed tasks
        seed_tasks = []
        for description in seed_task_descriptions:
            task = self.seed_task_manager.add_seed_task_from_description(
                description=description,
                environment_context=environment_context or {}
            )
            seed_tasks.append(task)
        
        # Analyze environment from seed tasks
        self.environment_state = self.seed_task_manager.analyze_environment_from_tasks()
        if environment_context:
            self.environment_state.current_context.update(environment_context)
        
        # Create learning session
        self.current_session = LearningSession(
            session_id=session_id,
            seed_tasks=seed_tasks,
            environment_state=self.environment_state,
            start_time=datetime.now().isoformat()
        )
        
        logger.info(f"Learning session started with {len(seed_tasks)} seed tasks")
        return self.current_session
    
    def run_autonomous_learning(
        self,
        max_rounds: Optional[int] = None,
        convergence_threshold: float = 0.95,
        save_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete autonomous learning pipeline.
        
        Args:
            max_rounds: Maximum learning rounds (uses default if None)
            convergence_threshold: Success rate threshold for convergence
            save_progress: Whether to save progress periodically
            
        Returns:
            Dictionary with learning results and statistics
        """
        if not self.current_session:
            raise ValueError("No active learning session. Call start_learning_session() first.")
        
        max_rounds = max_rounds or self.max_learning_rounds
        
        logger.info(f"Running autonomous learning for up to {max_rounds} rounds")
        
        learning_results = {
            "session_id": self.current_session.session_id,
            "rounds_completed": 0,
            "convergence_achieved": False,
            "final_metrics": {},
            "learning_insights": [],
            "generated_tasks": [],
            "optimization_summary": {}
        }
        
        for round_num in range(max_rounds):
            logger.info(f"Learning round {round_num + 1}/{max_rounds}")
            
            # Run one learning round
            round_results = self._run_learning_round(round_num)
            
            # Update session with round results
            self.current_session.optimization_rounds.append(round_results)
            
            # Check for convergence
            if self.learning_metrics.success_rate >= convergence_threshold:
                logger.info(f"Convergence achieved! Success rate: {self.learning_metrics.success_rate:.2f}")
                learning_results["convergence_achieved"] = True
                break
            
            # Save progress if requested
            if save_progress:
                self._save_session_progress()
            
            learning_results["rounds_completed"] = round_num + 1
        
        # Finalize learning session
        self.current_session.end_time = datetime.now().isoformat()
        self.current_session.metrics = self.learning_metrics
        
        # Compile final results
        learning_results.update({
            "final_metrics": self.learning_metrics.model_dump(),
            "learning_insights": self.current_session.insights,
            "generated_tasks": [task.model_dump() for task in self.current_session.generated_tasks],
            "optimization_summary": self.trajectory_optimizer.get_optimization_summary()
        })
        
        # Save final session
        if save_progress:
            self._save_final_session()
        
        logger.info(f"Autonomous learning completed: {learning_results['rounds_completed']} rounds")
        return learning_results
    
    def add_seed_tasks_during_learning(
        self,
        new_task_descriptions: List[str],
        environment_context: Optional[Dict[str, Any]] = None
    ) -> List[SeedTask]:
        """
        Add new seed tasks during an active learning session.
        
        Args:
            new_task_descriptions: New task descriptions to add
            environment_context: Additional environment context
            
        Returns:
            List of newly created seed tasks
        """
        if not self.current_session:
            raise ValueError("No active learning session")
        
        new_tasks = []
        for description in new_task_descriptions:
            task = self.seed_task_manager.add_seed_task_from_description(
                description=description,
                environment_context=environment_context or {}
            )
            new_tasks.append(task)
        
        # Add to current session
        self.current_session.seed_tasks.extend(new_tasks)
        
        # Update environment state
        updated_env_state = self.seed_task_manager.analyze_environment_from_tasks()
        self.environment_state = updated_env_state
        self.current_session.environment_state = updated_env_state
        
        logger.info(f"Added {len(new_tasks)} new seed tasks to active session")
        return new_tasks
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """
        Get current learning progress and statistics.
        
        Returns:
            Dictionary with current progress information
        """
        if not self.current_session:
            return {"error": "No active learning session"}
        
        return {
            "session_id": self.current_session.session_id,
            "rounds_completed": len(self.current_session.optimization_rounds),
            "current_metrics": self.learning_metrics.model_dump(),
            "environment_state": self.environment_state.model_dump(),
            "memory_statistics": self.memory_warmer.get_memory_statistics(),
            "task_generation_stats": self.task_generator.get_generation_statistics(),
            "optimization_progress": self.trajectory_optimizer.get_optimization_summary(),
            "recent_insights": self.current_session.insights[-10:]  # Last 10 insights
        }
    
    def generate_next_tasks(
        self,
        num_tasks: Optional[int] = None,
        focus_areas: Optional[List[str]] = None
    ) -> List[SeedTask]:
        """
        Generate next tasks based on current learning state.
        
        Args:
            num_tasks: Number of tasks to generate
            focus_areas: Specific areas to focus on
            
        Returns:
            List of generated tasks
        """
        if not self.current_session:
            raise ValueError("No active learning session")
        
        # Create task generation context
        context = self._create_task_generation_context(focus_areas)
        
        # Generate new tasks
        new_tasks = self.task_generator.generate_tasks(
            context=context,
            environment_state=self.environment_state,
            learning_metrics=self.learning_metrics,
            num_tasks=num_tasks
        )
        
        # Add to session
        self.current_session.generated_tasks.extend(new_tasks)
        
        logger.info(f"Generated {len(new_tasks)} new tasks")
        return new_tasks
    
    def _run_learning_round(self, round_num: int) -> Dict[str, Any]:
        """Run a single learning round."""
        logger.info(f"Starting learning round {round_num + 1}")
        
        round_start_time = time.time()
        
        # Step 1: Select tasks for this round
        current_tasks = self._select_tasks_for_round(round_num)
        
        # Step 2: Optimize trajectories for selected tasks
        logger.info("Optimizing trajectories...")
        optimized_trajectories = self.trajectory_optimizer.optimize_trajectories(
            seed_tasks=current_tasks,
            environment_executor=self.environment_executor,
            max_rounds=self.optimization_rounds_per_cycle
        )
        
        # Step 3: Warm up memory with optimized trajectories
        logger.info("Warming up memory...")
        memory_results = self.memory_warmer.warm_memory(
            optimized_trajectories=optimized_trajectories,
            trajectory_pairs=self.trajectory_optimizer.trajectory_pairs,
            seed_tasks=current_tasks
        )
        
        # Step 4: Generate new tasks based on learning
        logger.info("Generating new tasks...")
        context = self._create_task_generation_context()
        new_tasks = self.task_generator.generate_tasks(
            context=context,
            environment_state=self.environment_state,
            learning_metrics=self.learning_metrics,
            num_tasks=self.tasks_per_round
        )
        
        # Step 5: Update learning metrics and insights
        self._update_learning_metrics(optimized_trajectories, current_tasks)
        round_insights = self._extract_round_insights(optimized_trajectories)
        self.current_session.insights.extend(round_insights)
        
        # Step 6: Update environment state
        self._update_environment_state(optimized_trajectories, new_tasks)
        
        round_duration = time.time() - round_start_time
        
        round_results = {
            "round_number": round_num + 1,
            "duration_seconds": round_duration,
            "tasks_processed": len(current_tasks),
            "trajectories_optimized": len(optimized_trajectories),
            "new_tasks_generated": len(new_tasks),
            "memory_results": memory_results,
            "round_insights": round_insights,
            "metrics_snapshot": self.learning_metrics.model_dump()
        }
        
        logger.info(f"Learning round {round_num + 1} completed in {round_duration:.2f}s")
        return round_results
    
    def _select_tasks_for_round(self, round_num: int) -> List[SeedTask]:
        """Select tasks to work on for this round."""
        all_tasks = self.current_session.seed_tasks + self.current_session.generated_tasks
        
        # For early rounds, focus on seed tasks
        if round_num < 2:
            return self.current_session.seed_tasks[:self.tasks_per_round]
        
        # For later rounds, mix seed tasks and generated tasks
        seed_tasks = self.current_session.seed_tasks
        generated_tasks = self.current_session.generated_tasks
        
        # Select a mix
        num_seed = min(len(seed_tasks), self.tasks_per_round // 2)
        num_generated = min(len(generated_tasks), self.tasks_per_round - num_seed)
        
        selected_tasks = seed_tasks[:num_seed] + generated_tasks[-num_generated:]
        
        return selected_tasks[:self.tasks_per_round]
    
    def _create_task_generation_context(
        self,
        focus_areas: Optional[List[str]] = None
    ) -> TaskGenerationContext:
        """Create context for task generation."""
        # Extract successful and failure patterns from optimization history
        successful_patterns = []
        failure_patterns = []
        
        for insight in self.trajectory_optimizer.optimization_insights:
            if "success" in insight.lower():
                successful_patterns.append(insight)
            elif "failure" in insight.lower():
                failure_patterns.append(insight)
        
        # Identify learning gaps
        learning_gaps = focus_areas or self._identify_learning_gaps()
        
        # Determine current difficulty
        current_difficulty = self._determine_current_difficulty()
        
        return TaskGenerationContext(
            completed_tasks=[task.id for task in self.current_session.seed_tasks],
            successful_patterns=successful_patterns,
            failure_patterns=failure_patterns,
            environment_capabilities=self.environment_state.discovered_capabilities,
            learning_gaps=learning_gaps,
            current_difficulty=current_difficulty
        )
    
    def _identify_learning_gaps(self) -> List[str]:
        """Identify areas where more learning is needed."""
        gaps = []
        
        # Check tool usage gaps
        available_tools = set(self.environment_state.available_tools)
        discovered_tools = set(self.learning_metrics.tools_discovered)
        unexplored_tools = available_tools - discovered_tools
        
        for tool in unexplored_tools:
            gaps.append(f"tool usage: {tool}")
        
        # Check category gaps
        attempted_categories = set(self.learning_metrics.tasks_by_category.keys())
        all_categories = set(TaskCategory)
        
        for category in all_categories - attempted_categories:
            gaps.append(f"task category: {category.value}")
        
        return gaps
    
    def _determine_current_difficulty(self) -> TaskDifficulty:
        """Determine appropriate current difficulty level."""
        success_rate = self.learning_metrics.success_rate
        
        if success_rate > 0.8:
            return TaskDifficulty.ADVANCED
        elif success_rate > 0.6:
            return TaskDifficulty.INTERMEDIATE
        else:
            return TaskDifficulty.BEGINNER
    
    def _update_learning_metrics(
        self,
        trajectories: List[OptimizedTrajectory],
        tasks: List[SeedTask]
    ) -> None:
        """Update learning metrics based on round results."""
        for traj in trajectories:
            # Find corresponding task
            task = next((t for t in tasks if t.id == traj.original_task_id), None)
            if task:
                success = traj.outcome.value == "success"
                self.learning_metrics.add_task_attempt(
                    category=task.category,
                    difficulty=task.difficulty,
                    success=success
                )
        
        # Update average reward
        if trajectories:
            total_reward = sum(traj.reward for traj in trajectories)
            self.learning_metrics.average_reward = total_reward / len(trajectories)
    
    def _extract_round_insights(self, trajectories: List[OptimizedTrajectory]) -> List[str]:
        """Extract insights from the round's trajectories."""
        insights = []
        
        # Analyze success patterns
        successful_trajs = [t for t in trajectories if t.outcome.value == "success"]
        if successful_trajs:
            avg_success_reward = sum(t.reward for t in successful_trajs) / len(successful_trajs)
            insights.append(f"Successful trajectories achieved average reward of {avg_success_reward:.2f}")
        
        # Analyze failure patterns
        failed_trajs = [t for t in trajectories if t.outcome.value == "failure"]
        if failed_trajs:
            insights.append(f"Failed trajectories: {len(failed_trajs)} out of {len(trajectories)}")
        
        # Add optimization insights
        insights.extend(self.trajectory_optimizer.optimization_insights[-5:])  # Last 5 insights
        
        return insights
    
    def _update_environment_state(
        self,
        trajectories: List[OptimizedTrajectory],
        new_tasks: List[SeedTask]
    ) -> None:
        """Update environment state based on learning progress."""
        # Update discovered tools from trajectories
        for traj in trajectories:
            # Extract tools used in trajectory (simplified)
            for step in traj.tape.steps:
                if hasattr(step, 'content') and step.content:
                    # Simple tool detection - in practice, this would be more sophisticated
                    content_lower = step.content.lower()
                    for tool in self.environment_state.available_tools:
                        if tool.lower() in content_lower:
                            if tool not in self.learning_metrics.tools_discovered:
                                self.learning_metrics.tools_discovered.append(tool)
        
        # Update interaction history
        interaction_record = {
            "timestamp": datetime.now().isoformat(),
            "trajectories_count": len(trajectories),
            "new_tasks_count": len(new_tasks),
            "success_rate": self.learning_metrics.success_rate
        }
        self.environment_state.interaction_history.append(interaction_record)
    
    def _default_environment_executor(self, tape: Tape, task: SeedTask) -> Tape:
        """Default environment executor that just runs the agent."""
        try:
            result = self.agent.run(tape)
            return result.get_final_tape()
        except Exception as e:
            logger.warning(f"Environment execution failed for task {task.id}: {e}")
            return tape  # Return original tape if execution fails
    
    def _save_session_progress(self) -> None:
        """Save current session progress."""
        if not self.current_session:
            return
        
        progress_file = self.storage_path / f"session_{self.current_session.session_id}_progress.json"
        
        with open(progress_file, 'w') as f:
            json.dump(self.current_session.model_dump(), f, indent=2, default=str)
        
        logger.debug(f"Saved session progress to {progress_file}")
    
    def _save_final_session(self) -> None:
        """Save final session results."""
        if not self.current_session:
            return
        
        final_file = self.storage_path / f"session_{self.current_session.session_id}_final.json"
        
        with open(final_file, 'w') as f:
            json.dump(self.current_session.model_dump(), f, indent=2, default=str)
        
        # Also save seed tasks
        self.seed_task_manager.save_seed_tasks()
        
        logger.info(f"Saved final session to {final_file}")
    
    def load_session(self, session_id: str) -> Optional[LearningSession]:
        """
        Load a previous learning session.
        
        Args:
            session_id: ID of the session to load
            
        Returns:
            LearningSession object or None if not found
        """
        session_file = self.storage_path / f"session_{session_id}_final.json"
        
        if not session_file.exists():
            # Try progress file
            session_file = self.storage_path / f"session_{session_id}_progress.json"
        
        if not session_file.exists():
            logger.warning(f"Session file not found: {session_id}")
            return None
        
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            self.current_session = LearningSession.model_validate(session_data)
            self.environment_state = self.current_session.environment_state
            self.learning_metrics = self.current_session.metrics
            
            logger.info(f"Loaded session: {session_id}")
            return self.current_session
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current or last session."""
        if not self.current_session:
            return {"error": "No session available"}
        
        return {
            "session_id": self.current_session.session_id,
            "start_time": self.current_session.start_time,
            "end_time": self.current_session.end_time,
            "seed_tasks_count": len(self.current_session.seed_tasks),
            "generated_tasks_count": len(self.current_session.generated_tasks),
            "optimization_rounds": len(self.current_session.optimization_rounds),
            "total_insights": len(self.current_session.insights),
            "final_metrics": self.current_session.metrics.model_dump() if self.current_session.metrics else {},
            "environment_summary": {
                "available_tools": len(self.current_session.environment_state.available_tools),
                "data_sources": len(self.current_session.environment_state.data_sources),
                "interactions": len(self.current_session.environment_state.interaction_history)
            }
        }