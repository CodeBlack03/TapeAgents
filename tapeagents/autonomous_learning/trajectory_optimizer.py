"""
Trajectory Optimizer for autonomous learning.

This module implements ETO-style (Exploration-based Trajectory Optimization) 
trajectory optimization for TapeAgents, enabling learning from failure trajectories
and contrastive optimization.
"""

import logging
import random
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from tapeagents.core import Tape, Step
from tapeagents.agent import Agent
from tapeagents.llms import LLM

from .datatypes import (
    SeedTask, TrajectoryPair, OptimizedTrajectory, TrajectoryOutcome,
    LearningMetrics, TaskCategory, TaskDifficulty
)

logger = logging.getLogger(__name__)


class TrajectoryOptimizer:
    """
    Implements ETO-style trajectory optimization for TapeAgents.
    
    The TrajectoryOptimizer:
    - Generates multiple trajectory attempts for each task
    - Identifies successful and failed trajectories
    - Creates contrastive trajectory pairs for learning
    - Optimizes agent behavior through iterative improvement
    - Learns from both successes and failures
    """
    
    def __init__(
        self,
        agent: Agent,
        llm: LLM,
        max_exploration_rounds: int = 5,
        trajectories_per_round: int = 10,
        success_threshold: float = 0.7,
        learning_rate: float = 0.1
    ):
        """
        Initialize the TrajectoryOptimizer.
        
        Args:
            agent: TapeAgent to optimize
            llm: Language model for generating trajectories
            max_exploration_rounds: Maximum number of exploration rounds
            trajectories_per_round: Number of trajectories to generate per round
            success_threshold: Threshold for considering a trajectory successful
            learning_rate: Learning rate for optimization
        """
        self.agent = agent
        self.llm = llm
        self.max_exploration_rounds = max_exploration_rounds
        self.trajectories_per_round = trajectories_per_round
        self.success_threshold = success_threshold
        self.learning_rate = learning_rate
        
        # Storage for optimization data
        self.trajectory_history: List[OptimizedTrajectory] = []
        self.trajectory_pairs: List[TrajectoryPair] = []
        self.optimization_insights: List[str] = []
        
        # Metrics tracking
        self.metrics = LearningMetrics()
    
    def optimize_trajectories(
        self,
        seed_tasks: List[SeedTask],
        environment_executor: Any,  # Function or object that can execute tasks
        max_rounds: Optional[int] = None
    ) -> List[OptimizedTrajectory]:
        """
        Optimize trajectories for a set of seed tasks using ETO methodology.
        
        Args:
            seed_tasks: List of seed tasks to optimize for
            environment_executor: Function/object that can execute tasks in the environment
            max_rounds: Maximum optimization rounds (uses default if None)
            
        Returns:
            List of optimized trajectories
        """
        max_rounds = max_rounds or self.max_exploration_rounds
        optimized_trajectories = []
        
        logger.info(f"Starting trajectory optimization for {len(seed_tasks)} tasks")
        
        for round_num in range(max_rounds):
            logger.info(f"Optimization round {round_num + 1}/{max_rounds}")
            
            round_trajectories = []
            
            for task in seed_tasks:
                # Generate multiple trajectory attempts for this task
                task_trajectories = self._generate_task_trajectories(
                    task, environment_executor, round_num
                )
                round_trajectories.extend(task_trajectories)
            
            # Analyze trajectories and create pairs
            trajectory_pairs = self._create_trajectory_pairs(round_trajectories)
            self.trajectory_pairs.extend(trajectory_pairs)
            
            # Learn from trajectory pairs
            insights = self._learn_from_trajectory_pairs(trajectory_pairs)
            self.optimization_insights.extend(insights)
            
            # Select best trajectories for this round
            best_trajectories = self._select_best_trajectories(round_trajectories)
            optimized_trajectories.extend(best_trajectories)
            
            # Update metrics
            self._update_metrics(round_trajectories, round_num)
            
            logger.info(f"Round {round_num + 1} completed: {len(best_trajectories)} optimized trajectories")
        
        self.trajectory_history.extend(optimized_trajectories)
        logger.info(f"Trajectory optimization completed: {len(optimized_trajectories)} total optimized trajectories")
        
        return optimized_trajectories
    
    def _generate_task_trajectories(
        self,
        task: SeedTask,
        environment_executor: Any,
        round_num: int
    ) -> List[OptimizedTrajectory]:
        """
        Generate multiple trajectory attempts for a single task.
        
        Args:
            task: Task to generate trajectories for
            environment_executor: Environment execution function
            round_num: Current optimization round
            
        Returns:
            List of trajectory attempts
        """
        trajectories = []
        
        for attempt in range(self.trajectories_per_round):
            try:
                # Create initial tape for the task
                initial_tape = self._create_initial_tape(task)
                
                # Execute the task with some randomness for exploration
                execution_result = self._execute_task_with_exploration(
                    initial_tape, task, environment_executor, attempt
                )
                
                # Evaluate the trajectory
                outcome, reward = self._evaluate_trajectory(execution_result, task)
                
                # Create optimized trajectory object
                trajectory = OptimizedTrajectory(
                    id=f"{task.id}_round_{round_num}_attempt_{attempt}",
                    original_task_id=task.id,
                    tape=execution_result,
                    outcome=outcome,
                    reward=reward,
                    optimization_round=round_num,
                    improvements=[],
                    confidence_score=self._calculate_confidence_score(execution_result, reward)
                )
                
                trajectories.append(trajectory)
                
            except Exception as e:
                logger.warning(f"Failed to generate trajectory for task {task.id}, attempt {attempt}: {e}")
                continue
        
        return trajectories
    
    def _create_trajectory_pairs(
        self,
        trajectories: List[OptimizedTrajectory]
    ) -> List[TrajectoryPair]:
        """
        Create contrastive trajectory pairs from successful and failed trajectories.
        
        Args:
            trajectories: List of trajectories to pair
            
        Returns:
            List of trajectory pairs for contrastive learning
        """
        pairs = []
        
        # Group trajectories by task
        task_trajectories = defaultdict(list)
        for traj in trajectories:
            task_trajectories[traj.original_task_id].append(traj)
        
        # Create pairs within each task group
        for task_id, task_trajs in task_trajectories.items():
            # Separate successful and failed trajectories
            successful = [t for t in task_trajs if t.outcome == TrajectoryOutcome.SUCCESS]
            failed = [t for t in task_trajs if t.outcome == TrajectoryOutcome.FAILURE]
            
            # Create pairs between successful and failed trajectories
            for success_traj in successful:
                for fail_traj in failed:
                    # Analyze the contrast between success and failure
                    contrast_points = self._analyze_trajectory_contrast(
                        success_traj.tape, fail_traj.tape
                    )
                    
                    pair = TrajectoryPair(
                        id=f"pair_{success_traj.id}_{fail_traj.id}",
                        task_id=task_id,
                        success_trajectory=success_traj.tape,
                        failure_trajectory=fail_traj.tape,
                        success_reward=success_traj.reward,
                        failure_reward=fail_traj.reward,
                        contrast_points=contrast_points,
                        learning_insights=[]
                    )
                    
                    pairs.append(pair)
        
        logger.info(f"Created {len(pairs)} trajectory pairs for contrastive learning")
        return pairs
    
    def _learn_from_trajectory_pairs(
        self,
        trajectory_pairs: List[TrajectoryPair]
    ) -> List[str]:
        """
        Learn insights from trajectory pairs using contrastive analysis.
        
        Args:
            trajectory_pairs: Pairs of successful and failed trajectories
            
        Returns:
            List of learning insights
        """
        insights = []
        
        for pair in trajectory_pairs:
            # Analyze what made the successful trajectory work
            success_patterns = self._extract_success_patterns(pair.success_trajectory)
            failure_patterns = self._extract_failure_patterns(pair.failure_trajectory)
            
            # Generate insights from the contrast
            pair_insights = self._generate_contrastive_insights(
                success_patterns, failure_patterns, pair.contrast_points
            )
            
            pair.learning_insights = pair_insights
            insights.extend(pair_insights)
        
        # Deduplicate and prioritize insights
        unique_insights = list(set(insights))
        prioritized_insights = self._prioritize_insights(unique_insights)
        
        logger.info(f"Generated {len(prioritized_insights)} learning insights from trajectory pairs")
        return prioritized_insights
    
    def _select_best_trajectories(
        self,
        trajectories: List[OptimizedTrajectory]
    ) -> List[OptimizedTrajectory]:
        """
        Select the best trajectories from a round of optimization.
        
        Args:
            trajectories: All trajectories from the round
            
        Returns:
            Best trajectories selected for optimization
        """
        # Sort by reward and confidence score
        sorted_trajectories = sorted(
            trajectories,
            key=lambda t: (t.reward, t.confidence_score),
            reverse=True
        )
        
        # Select top trajectories, ensuring diversity
        selected = []
        task_counts = defaultdict(int)
        
        for traj in sorted_trajectories:
            # Limit trajectories per task to ensure diversity
            if task_counts[traj.original_task_id] < 2:  # Max 2 per task
                selected.append(traj)
                task_counts[traj.original_task_id] += 1
            
            # Stop when we have enough trajectories
            if len(selected) >= self.trajectories_per_round // 2:
                break
        
        return selected
    
    def _create_initial_tape(self, task: SeedTask) -> Tape:
        """Create an initial tape for a task."""
        # This is a simplified implementation - in practice, this would
        # create a proper initial tape based on the task and environment
        from tapeagents.dialog_tape import DialogTape, UserStep
        
        initial_step = UserStep(content=task.description)
        return DialogTape(steps=[initial_step])
    
    def _execute_task_with_exploration(
        self,
        initial_tape: Tape,
        task: SeedTask,
        environment_executor: Any,
        attempt: int
    ) -> Tape:
        """
        Execute a task with exploration randomness.
        
        Args:
            initial_tape: Initial tape for the task
            task: Task to execute
            environment_executor: Environment execution function
            attempt: Attempt number (for randomness seeding)
            
        Returns:
            Resulting tape after execution
        """
        # Add some randomness for exploration
        exploration_factor = 0.1 + (attempt * 0.05)  # Increase randomness with attempts
        
        # Modify agent parameters for exploration
        original_params = self.agent.llm.parameters.copy()
        exploration_params = original_params.copy()
        exploration_params["temperature"] = min(1.0, 
            exploration_params.get("temperature", 0.7) + exploration_factor)
        
        # Temporarily update agent parameters
        self.agent.llm.parameters = exploration_params
        
        try:
            # Execute the task
            result = self.agent.run(initial_tape)
            final_tape = result.get_final_tape()
            
            return final_tape
            
        finally:
            # Restore original parameters
            self.agent.llm.parameters = original_params
    
    def _evaluate_trajectory(self, tape: Tape, task: SeedTask) -> Tuple[TrajectoryOutcome, float]:
        """
        Evaluate a trajectory to determine its outcome and reward.
        
        Args:
            tape: Trajectory tape to evaluate
            task: Original task
            
        Returns:
            Tuple of (outcome, reward)
        """
        # Simple evaluation based on tape completion and success criteria
        # In practice, this would be more sophisticated
        
        if not tape.steps:
            return TrajectoryOutcome.ERROR, 0.0
        
        # Check if the trajectory completed successfully
        has_error = any(hasattr(step, 'error') and step.error for step in tape.steps)
        if has_error:
            return TrajectoryOutcome.ERROR, 0.1
        
        # Simple reward calculation based on number of steps and completion
        num_steps = len(tape.steps)
        
        # Reward based on task completion (simplified)
        if num_steps >= 2:  # At least user input and assistant response
            # Check if success criteria are met (simplified)
            if self._check_success_criteria(tape, task):
                return TrajectoryOutcome.SUCCESS, 0.8 + (0.2 * min(1.0, num_steps / 10))
            else:
                return TrajectoryOutcome.PARTIAL, 0.4 + (0.1 * min(1.0, num_steps / 10))
        else:
            return TrajectoryOutcome.FAILURE, 0.1
    
    def _check_success_criteria(self, tape: Tape, task: SeedTask) -> bool:
        """Check if a trajectory meets the task's success criteria."""
        # Simplified success checking - in practice, this would be more sophisticated
        if not task.success_criteria:
            return True  # No specific criteria, assume success if completed
        
        # Check if any success criteria keywords appear in the tape
        tape_content = " ".join([
            step.content for step in tape.steps 
            if hasattr(step, 'content') and step.content
        ]).lower()
        
        criteria_met = 0
        for criterion in task.success_criteria:
            if any(keyword.lower() in tape_content for keyword in criterion.split()):
                criteria_met += 1
        
        # Consider successful if at least half the criteria are met
        return criteria_met >= len(task.success_criteria) / 2
    
    def _calculate_confidence_score(self, tape: Tape, reward: float) -> float:
        """Calculate confidence score for a trajectory."""
        # Simple confidence calculation based on reward and tape quality
        base_confidence = reward
        
        # Adjust based on tape characteristics
        if tape.steps:
            # Longer tapes might indicate more thorough exploration
            length_factor = min(1.0, len(tape.steps) / 5)
            base_confidence += 0.1 * length_factor
        
        return min(1.0, base_confidence)
    
    def _analyze_trajectory_contrast(self, success_tape: Tape, failure_tape: Tape) -> List[str]:
        """Analyze the contrast between successful and failed trajectories."""
        contrasts = []
        
        # Compare tape lengths
        if len(success_tape.steps) != len(failure_tape.steps):
            contrasts.append(f"Different trajectory lengths: success={len(success_tape.steps)}, failure={len(failure_tape.steps)}")
        
        # Compare step types
        success_step_types = [type(step).__name__ for step in success_tape.steps]
        failure_step_types = [type(step).__name__ for step in failure_tape.steps]
        
        if success_step_types != failure_step_types:
            contrasts.append(f"Different step patterns: success={success_step_types}, failure={failure_step_types}")
        
        # Compare content patterns (simplified)
        success_content = " ".join([
            step.content for step in success_tape.steps 
            if hasattr(step, 'content') and step.content
        ])
        failure_content = " ".join([
            step.content for step in failure_tape.steps 
            if hasattr(step, 'content') and step.content
        ])
        
        if len(success_content) > len(failure_content) * 1.5:
            contrasts.append("Successful trajectory has significantly more content")
        elif len(failure_content) > len(success_content) * 1.5:
            contrasts.append("Failed trajectory has significantly more content")
        
        return contrasts
    
    def _extract_success_patterns(self, tape: Tape) -> List[str]:
        """Extract patterns that led to success."""
        patterns = []
        
        # Analyze step sequence patterns
        step_types = [type(step).__name__ for step in tape.steps]
        patterns.append(f"Step sequence: {' -> '.join(step_types)}")
        
        # Analyze content patterns
        if tape.steps:
            patterns.append(f"Total steps: {len(tape.steps)}")
            
            # Look for specific successful patterns
            content_words = []
            for step in tape.steps:
                if hasattr(step, 'content') and step.content:
                    content_words.extend(step.content.lower().split())
            
            if content_words:
                patterns.append(f"Content length: {len(content_words)} words")
        
        return patterns
    
    def _extract_failure_patterns(self, tape: Tape) -> List[str]:
        """Extract patterns that led to failure."""
        patterns = []
        
        # Similar to success patterns but focused on failure indicators
        step_types = [type(step).__name__ for step in tape.steps]
        patterns.append(f"Failed step sequence: {' -> '.join(step_types)}")
        
        # Look for error indicators
        has_errors = any(hasattr(step, 'error') and step.error for step in tape.steps)
        if has_errors:
            patterns.append("Contains error steps")
        
        if len(tape.steps) < 2:
            patterns.append("Trajectory too short")
        
        return patterns
    
    def _generate_contrastive_insights(
        self,
        success_patterns: List[str],
        failure_patterns: List[str],
        contrast_points: List[str]
    ) -> List[str]:
        """Generate insights from contrastive analysis."""
        insights = []
        
        # Generate insights from patterns
        for pattern in success_patterns:
            insights.append(f"Success factor: {pattern}")
        
        for pattern in failure_patterns:
            insights.append(f"Failure factor: {pattern}")
        
        # Generate insights from contrasts
        for contrast in contrast_points:
            insights.append(f"Key difference: {contrast}")
        
        return insights
    
    def _prioritize_insights(self, insights: List[str]) -> List[str]:
        """Prioritize insights by importance."""
        # Simple prioritization - in practice, this could be more sophisticated
        priority_keywords = ["error", "success", "failure", "length", "pattern"]
        
        prioritized = []
        for keyword in priority_keywords:
            for insight in insights:
                if keyword in insight.lower() and insight not in prioritized:
                    prioritized.append(insight)
        
        # Add remaining insights
        for insight in insights:
            if insight not in prioritized:
                prioritized.append(insight)
        
        return prioritized
    
    def _update_metrics(self, trajectories: List[OptimizedTrajectory], round_num: int) -> None:
        """Update optimization metrics."""
        self.metrics.optimization_rounds = round_num + 1
        self.metrics.trajectory_pairs_generated += len(self.trajectory_pairs)
        
        # Update task metrics
        for traj in trajectories:
            # This is simplified - in practice, we'd need to map back to task categories/difficulties
            success = traj.outcome == TrajectoryOutcome.SUCCESS
            self.metrics.add_task_attempt(TaskCategory.EXPLORATION, TaskDifficulty.BEGINNER, success)
        
        # Update average reward
        if trajectories:
            total_reward = sum(traj.reward for traj in trajectories)
            self.metrics.average_reward = total_reward / len(trajectories)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of the optimization process."""
        return {
            "total_trajectories": len(self.trajectory_history),
            "trajectory_pairs": len(self.trajectory_pairs),
            "optimization_insights": len(self.optimization_insights),
            "metrics": self.metrics.model_dump(),
            "top_insights": self.optimization_insights[:10]  # Top 10 insights
        }