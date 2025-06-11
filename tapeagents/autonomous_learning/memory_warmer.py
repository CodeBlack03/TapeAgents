"""
Memory Warmer for autonomous learning.

This module handles warming up the Tape memory with optimized trajectories,
creating a knowledge base that the agent can leverage for future tasks.
"""

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

from tapeagents.core import Tape, Step
from tapeagents.agent import Agent

from .datatypes import (
    OptimizedTrajectory, TrajectoryPair, SeedTask, TaskCategory, 
    TaskDifficulty, LearningMetrics
)

logger = logging.getLogger(__name__)


class MemoryWarmer:
    """
    Warms up agent memory with optimized trajectories.
    
    The MemoryWarmer:
    - Organizes optimized trajectories by task type and difficulty
    - Creates memory templates from successful trajectories
    - Builds a knowledge base of successful patterns
    - Provides memory retrieval for similar tasks
    - Maintains memory quality and relevance
    """
    
    def __init__(
        self,
        agent: Agent,
        max_memory_size: int = 1000,
        similarity_threshold: float = 0.7,
        memory_decay_factor: float = 0.95
    ):
        """
        Initialize the MemoryWarmer.
        
        Args:
            agent: TapeAgent to warm memory for
            max_memory_size: Maximum number of trajectories to keep in memory
            similarity_threshold: Threshold for considering trajectories similar
            memory_decay_factor: Factor for decaying old memory importance
        """
        self.agent = agent
        self.max_memory_size = max_memory_size
        self.similarity_threshold = similarity_threshold
        self.memory_decay_factor = memory_decay_factor
        
        # Memory storage
        self.memory_bank: Dict[str, List[OptimizedTrajectory]] = defaultdict(list)
        self.pattern_library: Dict[str, List[str]] = defaultdict(list)
        self.success_templates: Dict[TaskCategory, List[Tape]] = defaultdict(list)
        
        # Memory metadata
        self.memory_usage_stats: Dict[str, int] = defaultdict(int)
        self.memory_quality_scores: Dict[str, float] = {}
        
        # Metrics
        self.warmup_metrics = LearningMetrics()
    
    def warm_memory(
        self,
        optimized_trajectories: List[OptimizedTrajectory],
        trajectory_pairs: List[TrajectoryPair],
        seed_tasks: List[SeedTask]
    ) -> Dict[str, Any]:
        """
        Warm up the agent's memory with optimized trajectories.
        
        Args:
            optimized_trajectories: List of optimized trajectories
            trajectory_pairs: List of trajectory pairs for contrastive learning
            seed_tasks: Original seed tasks for context
            
        Returns:
            Dictionary with warmup results and statistics
        """
        logger.info(f"Starting memory warmup with {len(optimized_trajectories)} trajectories")
        
        # Organize trajectories by task characteristics
        self._organize_trajectories_by_task(optimized_trajectories, seed_tasks)
        
        # Extract successful patterns from trajectory pairs
        self._extract_success_patterns(trajectory_pairs)
        
        # Create success templates for different task categories
        self._create_success_templates(optimized_trajectories, seed_tasks)
        
        # Build pattern library for quick retrieval
        self._build_pattern_library(optimized_trajectories)
        
        # Optimize memory storage
        self._optimize_memory_storage()
        
        # Update metrics
        self._update_warmup_metrics(optimized_trajectories)
        
        warmup_results = {
            "trajectories_stored": len(optimized_trajectories),
            "memory_categories": len(self.memory_bank),
            "success_templates": sum(len(templates) for templates in self.success_templates.values()),
            "pattern_library_size": sum(len(patterns) for patterns in self.pattern_library.values()),
            "memory_quality_average": self._calculate_average_memory_quality(),
            "warmup_metrics": self.warmup_metrics.model_dump()
        }
        
        logger.info(f"Memory warmup completed: {warmup_results}")
        return warmup_results
    
    def retrieve_relevant_memory(
        self,
        task_description: str,
        task_category: Optional[TaskCategory] = None,
        task_difficulty: Optional[TaskDifficulty] = None,
        max_results: int = 5
    ) -> List[OptimizedTrajectory]:
        """
        Retrieve relevant memory trajectories for a given task.
        
        Args:
            task_description: Description of the task to find memory for
            task_category: Category of the task (optional)
            task_difficulty: Difficulty of the task (optional)
            max_results: Maximum number of trajectories to return
            
        Returns:
            List of relevant optimized trajectories
        """
        relevant_trajectories = []
        
        # Search by category first if provided
        if task_category:
            category_key = f"category_{task_category.value}"
            if category_key in self.memory_bank:
                relevant_trajectories.extend(self.memory_bank[category_key])
        
        # Search by difficulty if provided
        if task_difficulty:
            difficulty_key = f"difficulty_{task_difficulty.value}"
            if difficulty_key in self.memory_bank:
                relevant_trajectories.extend(self.memory_bank[difficulty_key])
        
        # Search by content similarity
        content_matches = self._find_content_similar_trajectories(task_description)
        relevant_trajectories.extend(content_matches)
        
        # Remove duplicates and rank by relevance
        unique_trajectories = self._deduplicate_trajectories(relevant_trajectories)
        ranked_trajectories = self._rank_trajectories_by_relevance(
            unique_trajectories, task_description, task_category, task_difficulty
        )
        
        # Update usage statistics
        for traj in ranked_trajectories[:max_results]:
            self.memory_usage_stats[traj.id] += 1
        
        logger.info(f"Retrieved {len(ranked_trajectories[:max_results])} relevant trajectories for task")
        return ranked_trajectories[:max_results]
    
    def get_success_template(
        self,
        task_category: TaskCategory,
        task_difficulty: Optional[TaskDifficulty] = None
    ) -> Optional[Tape]:
        """
        Get a success template for a specific task category.
        
        Args:
            task_category: Category of task to get template for
            task_difficulty: Difficulty level (optional)
            
        Returns:
            Success template tape or None if not found
        """
        if task_category not in self.success_templates:
            return None
        
        templates = self.success_templates[task_category]
        if not templates:
            return None
        
        # For now, return the first template
        # In practice, could select based on difficulty or other criteria
        return templates[0]
    
    def get_pattern_suggestions(
        self,
        task_description: str,
        max_suggestions: int = 10
    ) -> List[str]:
        """
        Get pattern suggestions based on task description.
        
        Args:
            task_description: Description of the task
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of pattern suggestions
        """
        suggestions = []
        task_words = set(task_description.lower().split())
        
        # Find patterns that match task keywords
        for pattern_type, patterns in self.pattern_library.items():
            for pattern in patterns:
                pattern_words = set(pattern.lower().split())
                # Check for word overlap
                if task_words.intersection(pattern_words):
                    suggestions.append(f"{pattern_type}: {pattern}")
        
        # Sort by relevance (simplified)
        suggestions.sort(key=lambda x: len(set(x.lower().split()).intersection(task_words)), reverse=True)
        
        return suggestions[:max_suggestions]
    
    def update_memory_quality(
        self,
        trajectory_id: str,
        success_feedback: bool,
        reward_feedback: Optional[float] = None
    ) -> None:
        """
        Update memory quality based on usage feedback.
        
        Args:
            trajectory_id: ID of the trajectory to update
            success_feedback: Whether the trajectory led to success
            reward_feedback: Optional reward feedback
        """
        current_quality = self.memory_quality_scores.get(trajectory_id, 0.5)
        
        # Update quality based on feedback
        if success_feedback:
            new_quality = min(1.0, current_quality + 0.1)
        else:
            new_quality = max(0.0, current_quality - 0.1)
        
        # Incorporate reward feedback if provided
        if reward_feedback is not None:
            new_quality = (new_quality + reward_feedback) / 2
        
        self.memory_quality_scores[trajectory_id] = new_quality
        
        logger.debug(f"Updated memory quality for {trajectory_id}: {current_quality} -> {new_quality}")
    
    def cleanup_memory(self, force_cleanup: bool = False) -> Dict[str, int]:
        """
        Clean up memory by removing low-quality or unused trajectories.
        
        Args:
            force_cleanup: Whether to force cleanup even if under memory limit
            
        Returns:
            Dictionary with cleanup statistics
        """
        total_trajectories = sum(len(trajs) for trajs in self.memory_bank.values())
        
        if not force_cleanup and total_trajectories <= self.max_memory_size:
            return {"removed": 0, "remaining": total_trajectories}
        
        removed_count = 0
        
        # Remove low-quality trajectories
        for category, trajectories in self.memory_bank.items():
            to_remove = []
            for traj in trajectories:
                quality = self.memory_quality_scores.get(traj.id, 0.5)
                usage = self.memory_usage_stats.get(traj.id, 0)
                
                # Remove if quality is low and usage is minimal
                if quality < 0.3 and usage < 2:
                    to_remove.append(traj)
            
            for traj in to_remove:
                trajectories.remove(traj)
                removed_count += 1
        
        # If still over limit, remove least used trajectories
        if total_trajectories - removed_count > self.max_memory_size:
            all_trajectories = []
            for category, trajectories in self.memory_bank.items():
                for traj in trajectories:
                    all_trajectories.append((category, traj))
            
            # Sort by usage and quality
            all_trajectories.sort(
                key=lambda x: (
                    self.memory_usage_stats.get(x[1].id, 0),
                    self.memory_quality_scores.get(x[1].id, 0.5)
                )
            )
            
            # Remove least valuable trajectories
            target_size = self.max_memory_size
            while len(all_trajectories) > target_size:
                category, traj = all_trajectories.pop(0)
                self.memory_bank[category].remove(traj)
                removed_count += 1
        
        remaining_count = sum(len(trajs) for trajs in self.memory_bank.values())
        
        logger.info(f"Memory cleanup completed: removed {removed_count}, remaining {remaining_count}")
        return {"removed": removed_count, "remaining": remaining_count}
    
    def _organize_trajectories_by_task(
        self,
        trajectories: List[OptimizedTrajectory],
        seed_tasks: List[SeedTask]
    ) -> None:
        """Organize trajectories by task characteristics."""
        # Create task lookup
        task_lookup = {task.id: task for task in seed_tasks}
        
        for traj in trajectories:
            # Store by task ID
            self.memory_bank[f"task_{traj.original_task_id}"].append(traj)
            
            # Store by task category if available
            if traj.original_task_id in task_lookup:
                task = task_lookup[traj.original_task_id]
                self.memory_bank[f"category_{task.category.value}"].append(traj)
                self.memory_bank[f"difficulty_{task.difficulty.value}"].append(traj)
            
            # Store by outcome
            self.memory_bank[f"outcome_{traj.outcome.value}"].append(traj)
            
            # Store by optimization round
            self.memory_bank[f"round_{traj.optimization_round}"].append(traj)
    
    def _extract_success_patterns(self, trajectory_pairs: List[TrajectoryPair]) -> None:
        """Extract success patterns from trajectory pairs."""
        for pair in trajectory_pairs:
            # Extract patterns from successful trajectories
            success_patterns = self._analyze_trajectory_patterns(pair.success_trajectory)
            
            for pattern_type, patterns in success_patterns.items():
                self.pattern_library[f"success_{pattern_type}"].extend(patterns)
            
            # Store learning insights as patterns
            self.pattern_library["insights"].extend(pair.learning_insights)
    
    def _create_success_templates(
        self,
        trajectories: List[OptimizedTrajectory],
        seed_tasks: List[SeedTask]
    ) -> None:
        """Create success templates for different task categories."""
        task_lookup = {task.id: task for task in seed_tasks}
        
        # Group successful trajectories by category
        category_successes = defaultdict(list)
        
        for traj in trajectories:
            if traj.outcome.value == "success" and traj.original_task_id in task_lookup:
                task = task_lookup[traj.original_task_id]
                category_successes[task.category].append(traj.tape)
        
        # Create templates from best trajectories in each category
        for category, tapes in category_successes.items():
            # Sort by quality and take the best ones
            sorted_tapes = sorted(tapes, key=lambda t: len(t.steps), reverse=True)
            self.success_templates[category] = sorted_tapes[:3]  # Keep top 3
    
    def _build_pattern_library(self, trajectories: List[OptimizedTrajectory]) -> None:
        """Build a pattern library from trajectories."""
        for traj in trajectories:
            patterns = self._analyze_trajectory_patterns(traj.tape)
            
            for pattern_type, pattern_list in patterns.items():
                self.pattern_library[pattern_type].extend(pattern_list)
        
        # Deduplicate patterns
        for pattern_type in self.pattern_library:
            self.pattern_library[pattern_type] = list(set(self.pattern_library[pattern_type]))
    
    def _analyze_trajectory_patterns(self, tape: Tape) -> Dict[str, List[str]]:
        """Analyze patterns in a trajectory tape."""
        patterns = defaultdict(list)
        
        # Step sequence patterns
        step_types = [type(step).__name__ for step in tape.steps]
        patterns["step_sequence"].append(" -> ".join(step_types))
        
        # Content patterns
        for step in tape.steps:
            if hasattr(step, 'content') and step.content:
                # Extract keywords
                words = step.content.lower().split()
                if len(words) > 2:
                    patterns["content_keywords"].extend(words[:5])  # First 5 words
        
        # Length patterns
        patterns["length"].append(f"steps_{len(tape.steps)}")
        
        return patterns
    
    def _find_content_similar_trajectories(self, task_description: str) -> List[OptimizedTrajectory]:
        """Find trajectories with similar content to the task description."""
        similar_trajectories = []
        task_words = set(task_description.lower().split())
        
        for trajectories in self.memory_bank.values():
            for traj in trajectories:
                # Extract content from trajectory
                traj_content = []
                for step in traj.tape.steps:
                    if hasattr(step, 'content') and step.content:
                        traj_content.extend(step.content.lower().split())
                
                traj_words = set(traj_content)
                
                # Calculate similarity
                if task_words and traj_words:
                    similarity = len(task_words.intersection(traj_words)) / len(task_words.union(traj_words))
                    if similarity >= self.similarity_threshold:
                        similar_trajectories.append(traj)
        
        return similar_trajectories
    
    def _deduplicate_trajectories(self, trajectories: List[OptimizedTrajectory]) -> List[OptimizedTrajectory]:
        """Remove duplicate trajectories."""
        seen_ids = set()
        unique_trajectories = []
        
        for traj in trajectories:
            if traj.id not in seen_ids:
                unique_trajectories.append(traj)
                seen_ids.add(traj.id)
        
        return unique_trajectories
    
    def _rank_trajectories_by_relevance(
        self,
        trajectories: List[OptimizedTrajectory],
        task_description: str,
        task_category: Optional[TaskCategory],
        task_difficulty: Optional[TaskDifficulty]
    ) -> List[OptimizedTrajectory]:
        """Rank trajectories by relevance to the current task."""
        def relevance_score(traj: OptimizedTrajectory) -> float:
            score = 0.0
            
            # Base score from trajectory quality
            score += traj.reward * 0.3
            score += traj.confidence_score * 0.2
            
            # Usage-based score
            usage = self.memory_usage_stats.get(traj.id, 0)
            score += min(0.2, usage * 0.05)
            
            # Quality-based score
            quality = self.memory_quality_scores.get(traj.id, 0.5)
            score += quality * 0.3
            
            return score
        
        # Sort by relevance score
        ranked_trajectories = sorted(trajectories, key=relevance_score, reverse=True)
        return ranked_trajectories
    
    def _optimize_memory_storage(self) -> None:
        """Optimize memory storage by removing redundant or low-quality entries."""
        # Apply memory decay to older trajectories
        for trajectories in self.memory_bank.values():
            for traj in trajectories:
                if traj.id in self.memory_quality_scores:
                    self.memory_quality_scores[traj.id] *= self.memory_decay_factor
    
    def _update_warmup_metrics(self, trajectories: List[OptimizedTrajectory]) -> None:
        """Update warmup metrics."""
        self.warmup_metrics.memory_warmup_completions += 1
        
        # Count successful trajectories
        successful_count = sum(1 for traj in trajectories if traj.outcome.value == "success")
        self.warmup_metrics.total_tasks_attempted += len(trajectories)
        self.warmup_metrics.total_tasks_completed += successful_count
        self.warmup_metrics.update_success_rate()
    
    def _calculate_average_memory_quality(self) -> float:
        """Calculate average memory quality."""
        if not self.memory_quality_scores:
            return 0.0
        
        return sum(self.memory_quality_scores.values()) / len(self.memory_quality_scores)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        total_trajectories = sum(len(trajs) for trajs in self.memory_bank.values())
        
        return {
            "total_trajectories": total_trajectories,
            "memory_categories": len(self.memory_bank),
            "pattern_library_size": sum(len(patterns) for patterns in self.pattern_library.values()),
            "success_templates": {
                category.value: len(templates) 
                for category, templates in self.success_templates.items()
            },
            "average_quality": self._calculate_average_memory_quality(),
            "most_used_trajectories": sorted(
                self.memory_usage_stats.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            "memory_utilization": min(1.0, total_trajectories / self.max_memory_size),
            "warmup_metrics": self.warmup_metrics.model_dump()
        }