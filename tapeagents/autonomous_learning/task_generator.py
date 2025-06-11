"""
Autonomous Task Generator for learning environments.

This module generates new tasks based on past interactions and learning progress,
enabling continuous improvement and exploration of the environment.
"""

import logging
import random
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict

from tapeagents.core import Tape
from tapeagents.llms import LLM

from .datatypes import (
    SeedTask, TaskCategory, TaskDifficulty, TaskGenerationContext,
    OptimizedTrajectory, LearningMetrics, EnvironmentState
)

logger = logging.getLogger(__name__)


class AutonomousTaskGenerator:
    """
    Generates new tasks autonomously based on learning progress and environment exploration.
    
    The AutonomousTaskGenerator:
    - Analyzes successful and failed patterns from past tasks
    - Identifies learning gaps and unexplored areas
    - Generates progressively challenging tasks
    - Creates tasks that explore new environment capabilities
    - Adapts task generation based on agent performance
    """
    
    def __init__(
        self,
        llm: LLM,
        max_tasks_per_generation: int = 10,
        difficulty_progression_rate: float = 0.1,
        exploration_bias: float = 0.3,
        creativity_factor: float = 0.2
    ):
        """
        Initialize the AutonomousTaskGenerator.
        
        Args:
            llm: Language model for generating task descriptions
            max_tasks_per_generation: Maximum tasks to generate per round
            difficulty_progression_rate: Rate at which to increase difficulty
            exploration_bias: Bias towards exploration vs exploitation
            creativity_factor: Factor for generating creative/novel tasks
        """
        self.llm = llm
        self.max_tasks_per_generation = max_tasks_per_generation
        self.difficulty_progression_rate = difficulty_progression_rate
        self.exploration_bias = exploration_bias
        self.creativity_factor = creativity_factor
        
        # Task generation history
        self.generated_tasks: List[SeedTask] = []
        self.generation_history: List[Dict[str, Any]] = []
        
        # Learning state
        self.known_patterns: Set[str] = set()
        self.successful_strategies: List[str] = []
        self.failed_strategies: List[str] = []
        
        # Environment understanding
        self.discovered_capabilities: Dict[str, Any] = {}
        self.unexplored_areas: List[str] = []
    
    def generate_tasks(
        self,
        context: TaskGenerationContext,
        environment_state: EnvironmentState,
        learning_metrics: LearningMetrics,
        num_tasks: Optional[int] = None
    ) -> List[SeedTask]:
        """
        Generate new tasks based on current learning context.
        
        Args:
            context: Current task generation context
            environment_state: Current state of the environment
            learning_metrics: Current learning metrics
            num_tasks: Number of tasks to generate (uses default if None)
            
        Returns:
            List of newly generated tasks
        """
        num_tasks = num_tasks or self.max_tasks_per_generation
        
        logger.info(f"Generating {num_tasks} new tasks based on learning context")
        
        # Update internal state
        self._update_learning_state(context, environment_state, learning_metrics)
        
        # Determine task generation strategy
        generation_strategy = self._determine_generation_strategy(context, learning_metrics)
        
        # Generate tasks based on strategy
        new_tasks = []
        
        # Generate different types of tasks
        exploration_tasks = self._generate_exploration_tasks(
            environment_state, int(num_tasks * self.exploration_bias)
        )
        new_tasks.extend(exploration_tasks)
        
        # Generate skill-building tasks
        skill_tasks = self._generate_skill_building_tasks(
            context, int(num_tasks * 0.4)
        )
        new_tasks.extend(skill_tasks)
        
        # Generate integration tasks
        integration_tasks = self._generate_integration_tasks(
            context, environment_state, int(num_tasks * 0.2)
        )
        new_tasks.extend(integration_tasks)
        
        # Generate creative/novel tasks
        creative_tasks = self._generate_creative_tasks(
            environment_state, int(num_tasks * self.creativity_factor)
        )
        new_tasks.extend(creative_tasks)
        
        # Fill remaining slots with adaptive tasks
        remaining_slots = num_tasks - len(new_tasks)
        if remaining_slots > 0:
            adaptive_tasks = self._generate_adaptive_tasks(
                context, learning_metrics, remaining_slots
            )
            new_tasks.extend(adaptive_tasks)
        
        # Trim to requested number and add to history
        final_tasks = new_tasks[:num_tasks]
        self.generated_tasks.extend(final_tasks)
        
        # Record generation round
        generation_record = {
            "round": len(self.generation_history) + 1,
            "strategy": generation_strategy,
            "tasks_generated": len(final_tasks),
            "task_categories": self._count_task_categories(final_tasks),
            "context_summary": self._summarize_context(context)
        }
        self.generation_history.append(generation_record)
        
        logger.info(f"Generated {len(final_tasks)} tasks using {generation_strategy} strategy")
        return final_tasks
    
    def generate_progressive_tasks(
        self,
        base_task: SeedTask,
        progression_steps: int = 5
    ) -> List[SeedTask]:
        """
        Generate a progressive sequence of tasks building on a base task.
        
        Args:
            base_task: Base task to build progression from
            progression_steps: Number of progression steps
            
        Returns:
            List of progressively challenging tasks
        """
        progressive_tasks = []
        current_difficulty = base_task.difficulty
        
        for step in range(progression_steps):
            # Increase difficulty gradually
            if step > 0:
                current_difficulty = self._get_next_difficulty(current_difficulty)
            
            # Generate task description with increased complexity
            task_description = self._create_progressive_description(
                base_task.description, step, progression_steps
            )
            
            # Create progressive task
            progressive_task = SeedTask(
                id=f"{base_task.id}_prog_{step + 1}",
                description=task_description,
                category=base_task.category,
                difficulty=current_difficulty,
                environment_context=base_task.environment_context.copy(),
                success_criteria=self._adapt_success_criteria(base_task.success_criteria, step),
                examples=base_task.examples.copy(),
                metadata={
                    "auto_generated": True,
                    "source": "progressive_generation",
                    "base_task_id": base_task.id,
                    "progression_step": step + 1,
                    "total_steps": progression_steps
                }
            )
            
            progressive_tasks.append(progressive_task)
        
        logger.info(f"Generated {len(progressive_tasks)} progressive tasks from {base_task.id}")
        return progressive_tasks
    
    def generate_exploration_tasks_for_tools(
        self,
        available_tools: List[str],
        unexplored_tools: List[str]
    ) -> List[SeedTask]:
        """
        Generate tasks specifically for exploring tools and capabilities.
        
        Args:
            available_tools: List of available tools in the environment
            unexplored_tools: List of tools that haven't been explored yet
            
        Returns:
            List of tool exploration tasks
        """
        exploration_tasks = []
        
        # Prioritize unexplored tools
        tools_to_explore = unexplored_tools + [
            tool for tool in available_tools if tool not in unexplored_tools
        ]
        
        for i, tool in enumerate(tools_to_explore[:self.max_tasks_per_generation]):
            # Generate exploration task for this tool
            task_description = self._create_tool_exploration_description(tool)
            
            exploration_task = SeedTask(
                id=f"explore_tool_{tool}_{i + 1}",
                description=task_description,
                category=TaskCategory.TOOL_USAGE,
                difficulty=TaskDifficulty.BEGINNER,
                environment_context={"target_tool": tool, "exploration_focus": True},
                success_criteria=[f"Successfully use {tool}", f"Understand {tool} capabilities"],
                examples=[f"Use {tool} to accomplish a simple task"],
                metadata={
                    "auto_generated": True,
                    "source": "tool_exploration",
                    "target_tool": tool,
                    "exploration_type": "tool_discovery"
                }
            )
            
            exploration_tasks.append(exploration_task)
        
        logger.info(f"Generated {len(exploration_tasks)} tool exploration tasks")
        return exploration_tasks
    
    def adapt_tasks_based_on_performance(
        self,
        recent_tasks: List[SeedTask],
        performance_metrics: Dict[str, float],
        optimization_trajectories: List[OptimizedTrajectory]
    ) -> List[SeedTask]:
        """
        Adapt task generation based on recent performance.
        
        Args:
            recent_tasks: Recently completed tasks
            performance_metrics: Performance metrics for recent tasks
            optimization_trajectories: Recent optimization trajectories
            
        Returns:
            List of adapted tasks
        """
        adapted_tasks = []
        
        # Analyze performance patterns
        success_rate = performance_metrics.get("success_rate", 0.5)
        average_reward = performance_metrics.get("average_reward", 0.5)
        
        # Determine adaptation strategy
        if success_rate > 0.8:
            # High success rate - increase difficulty
            adaptation_strategy = "increase_difficulty"
            target_difficulty = self._get_higher_difficulty_target()
        elif success_rate < 0.3:
            # Low success rate - decrease difficulty or provide more support
            adaptation_strategy = "decrease_difficulty"
            target_difficulty = self._get_lower_difficulty_target()
        else:
            # Moderate success rate - maintain current level with variations
            adaptation_strategy = "maintain_with_variation"
            target_difficulty = None
        
        # Generate adapted tasks based on strategy
        for i in range(min(len(recent_tasks), self.max_tasks_per_generation)):
            base_task = recent_tasks[i % len(recent_tasks)]
            
            adapted_task = self._create_adapted_task(
                base_task, adaptation_strategy, target_difficulty, i
            )
            adapted_tasks.append(adapted_task)
        
        logger.info(f"Generated {len(adapted_tasks)} adapted tasks using {adaptation_strategy} strategy")
        return adapted_tasks
    
    def _update_learning_state(
        self,
        context: TaskGenerationContext,
        environment_state: EnvironmentState,
        metrics: LearningMetrics
    ) -> None:
        """Update internal learning state based on context."""
        # Update known patterns
        self.known_patterns.update(context.successful_patterns)
        
        # Update strategies
        self.successful_strategies.extend(context.successful_patterns)
        self.failed_strategies.extend(context.failure_patterns)
        
        # Update environment understanding
        self.discovered_capabilities.update(environment_state.discovered_capabilities)
        
        # Identify unexplored areas
        all_tools = set(environment_state.available_tools)
        explored_tools = set(metrics.tools_discovered)
        self.unexplored_areas = list(all_tools - explored_tools)
    
    def _determine_generation_strategy(
        self,
        context: TaskGenerationContext,
        metrics: LearningMetrics
    ) -> str:
        """Determine the best task generation strategy."""
        success_rate = metrics.success_rate
        
        if success_rate > 0.8:
            return "challenge_focused"
        elif success_rate < 0.3:
            return "support_focused"
        elif len(self.unexplored_areas) > 3:
            return "exploration_focused"
        else:
            return "balanced"
    
    def _generate_exploration_tasks(
        self,
        environment_state: EnvironmentState,
        num_tasks: int
    ) -> List[SeedTask]:
        """Generate tasks focused on environment exploration."""
        exploration_tasks = []
        
        # Explore unexplored tools
        for i, area in enumerate(self.unexplored_areas[:num_tasks]):
            task_description = f"Explore and understand the capabilities of {area}"
            
            task = SeedTask(
                id=f"explore_{area}_{len(self.generated_tasks) + i + 1}",
                description=task_description,
                category=TaskCategory.EXPLORATION,
                difficulty=TaskDifficulty.BEGINNER,
                environment_context={"exploration_target": area},
                success_criteria=[f"Successfully interact with {area}"],
                examples=[],
                metadata={
                    "auto_generated": True,
                    "source": "exploration_generation",
                    "exploration_target": area
                }
            )
            
            exploration_tasks.append(task)
        
        return exploration_tasks
    
    def _generate_skill_building_tasks(
        self,
        context: TaskGenerationContext,
        num_tasks: int
    ) -> List[SeedTask]:
        """Generate tasks focused on building specific skills."""
        skill_tasks = []
        
        # Identify skill gaps from learning gaps
        for i, gap in enumerate(context.learning_gaps[:num_tasks]):
            task_description = f"Practice and improve skills in {gap}"
            
            task = SeedTask(
                id=f"skill_{gap.replace(' ', '_')}_{len(self.generated_tasks) + i + 1}",
                description=task_description,
                category=TaskCategory.PROBLEM_SOLVING,
                difficulty=context.current_difficulty,
                environment_context={"skill_focus": gap},
                success_criteria=[f"Demonstrate competency in {gap}"],
                examples=[],
                metadata={
                    "auto_generated": True,
                    "source": "skill_building",
                    "skill_focus": gap
                }
            )
            
            skill_tasks.append(task)
        
        return skill_tasks
    
    def _generate_integration_tasks(
        self,
        context: TaskGenerationContext,
        environment_state: EnvironmentState,
        num_tasks: int
    ) -> List[SeedTask]:
        """Generate tasks that integrate multiple capabilities."""
        integration_tasks = []
        
        # Combine successful patterns into integration tasks
        if len(context.successful_patterns) >= 2:
            for i in range(num_tasks):
                # Select random patterns to combine
                patterns = random.sample(
                    context.successful_patterns,
                    min(2, len(context.successful_patterns))
                )
                
                task_description = f"Combine and integrate: {' and '.join(patterns)}"
                
                task = SeedTask(
                    id=f"integrate_{i + 1}_{len(self.generated_tasks) + 1}",
                    description=task_description,
                    category=TaskCategory.INTEGRATION,
                    difficulty=self._get_next_difficulty(context.current_difficulty),
                    environment_context={"integration_patterns": patterns},
                    success_criteria=[f"Successfully combine {len(patterns)} different approaches"],
                    examples=[],
                    metadata={
                        "auto_generated": True,
                        "source": "integration_generation",
                        "combined_patterns": patterns
                    }
                )
                
                integration_tasks.append(task)
        
        return integration_tasks
    
    def _generate_creative_tasks(
        self,
        environment_state: EnvironmentState,
        num_tasks: int
    ) -> List[SeedTask]:
        """Generate creative and novel tasks."""
        creative_tasks = []
        
        # Generate creative combinations of available tools
        tools = environment_state.available_tools
        if len(tools) >= 2:
            for i in range(num_tasks):
                # Select random tools to combine creatively
                selected_tools = random.sample(tools, min(2, len(tools)))
                
                task_description = f"Create something innovative using {' and '.join(selected_tools)}"
                
                task = SeedTask(
                    id=f"creative_{i + 1}_{len(self.generated_tasks) + 1}",
                    description=task_description,
                    category=TaskCategory.CREATIVE,
                    difficulty=TaskDifficulty.INTERMEDIATE,
                    environment_context={"creative_tools": selected_tools},
                    success_criteria=["Create something novel and useful"],
                    examples=[],
                    metadata={
                        "auto_generated": True,
                        "source": "creative_generation",
                        "creative_tools": selected_tools
                    }
                )
                
                creative_tasks.append(task)
        
        return creative_tasks
    
    def _generate_adaptive_tasks(
        self,
        context: TaskGenerationContext,
        metrics: LearningMetrics,
        num_tasks: int
    ) -> List[SeedTask]:
        """Generate adaptive tasks based on current performance."""
        adaptive_tasks = []
        
        # Adapt based on success rate
        if metrics.success_rate > 0.7:
            # Generate harder tasks
            target_difficulty = self._get_next_difficulty(context.current_difficulty)
            task_prefix = "Advanced challenge"
        else:
            # Generate supportive tasks
            target_difficulty = context.current_difficulty
            task_prefix = "Practice and reinforce"
        
        for i in range(num_tasks):
            task_description = f"{task_prefix}: Task {i + 1} adapted to current performance level"
            
            task = SeedTask(
                id=f"adaptive_{i + 1}_{len(self.generated_tasks) + 1}",
                description=task_description,
                category=TaskCategory.PROBLEM_SOLVING,
                difficulty=target_difficulty,
                environment_context={"adaptive_generation": True},
                success_criteria=["Meet performance expectations"],
                examples=[],
                metadata={
                    "auto_generated": True,
                    "source": "adaptive_generation",
                    "performance_based": True
                }
            )
            
            adaptive_tasks.append(task)
        
        return adaptive_tasks
    
    def _create_progressive_description(
        self,
        base_description: str,
        step: int,
        total_steps: int
    ) -> str:
        """Create a progressive task description."""
        complexity_modifiers = [
            "Simple version:",
            "Intermediate version:",
            "Advanced version:",
            "Complex version:",
            "Expert-level version:"
        ]
        
        modifier_index = min(step, len(complexity_modifiers) - 1)
        modifier = complexity_modifiers[modifier_index]
        
        return f"{modifier} {base_description}"
    
    def _adapt_success_criteria(self, base_criteria: List[str], step: int) -> List[str]:
        """Adapt success criteria for progressive tasks."""
        if not base_criteria:
            return [f"Complete task at progression level {step + 1}"]
        
        adapted_criteria = []
        for criterion in base_criteria:
            if step == 0:
                adapted_criteria.append(f"Basic: {criterion}")
            elif step < 3:
                adapted_criteria.append(f"Intermediate: {criterion}")
            else:
                adapted_criteria.append(f"Advanced: {criterion}")
        
        return adapted_criteria
    
    def _create_tool_exploration_description(self, tool: str) -> str:
        """Create a description for tool exploration."""
        exploration_templates = [
            f"Discover what {tool} can do and how to use it effectively",
            f"Explore the capabilities and features of {tool}",
            f"Learn to use {tool} for various tasks and scenarios",
            f"Investigate the potential applications of {tool}",
            f"Master the basic and advanced features of {tool}"
        ]
        
        return random.choice(exploration_templates)
    
    def _create_adapted_task(
        self,
        base_task: SeedTask,
        strategy: str,
        target_difficulty: Optional[TaskDifficulty],
        index: int
    ) -> SeedTask:
        """Create an adapted task based on performance strategy."""
        if strategy == "increase_difficulty":
            description = f"Advanced challenge: {base_task.description}"
            difficulty = target_difficulty or self._get_next_difficulty(base_task.difficulty)
        elif strategy == "decrease_difficulty":
            description = f"Simplified approach: {base_task.description}"
            difficulty = target_difficulty or self._get_previous_difficulty(base_task.difficulty)
        else:
            description = f"Alternative approach: {base_task.description}"
            difficulty = base_task.difficulty
        
        return SeedTask(
            id=f"adapted_{base_task.id}_{strategy}_{index + 1}",
            description=description,
            category=base_task.category,
            difficulty=difficulty,
            environment_context=base_task.environment_context.copy(),
            success_criteria=base_task.success_criteria.copy(),
            examples=base_task.examples.copy(),
            metadata={
                "auto_generated": True,
                "source": "performance_adaptation",
                "adaptation_strategy": strategy,
                "base_task_id": base_task.id
            }
        )
    
    def _get_next_difficulty(self, current: TaskDifficulty) -> TaskDifficulty:
        """Get the next difficulty level."""
        difficulties = [TaskDifficulty.BEGINNER, TaskDifficulty.INTERMEDIATE, 
                      TaskDifficulty.ADVANCED, TaskDifficulty.EXPERT]
        current_index = difficulties.index(current)
        return difficulties[min(len(difficulties) - 1, current_index + 1)]
    
    def _get_previous_difficulty(self, current: TaskDifficulty) -> TaskDifficulty:
        """Get the previous difficulty level."""
        difficulties = [TaskDifficulty.BEGINNER, TaskDifficulty.INTERMEDIATE, 
                      TaskDifficulty.ADVANCED, TaskDifficulty.EXPERT]
        current_index = difficulties.index(current)
        return difficulties[max(0, current_index - 1)]
    
    def _get_higher_difficulty_target(self) -> TaskDifficulty:
        """Get a higher difficulty target for adaptation."""
        return TaskDifficulty.ADVANCED
    
    def _get_lower_difficulty_target(self) -> TaskDifficulty:
        """Get a lower difficulty target for adaptation."""
        return TaskDifficulty.BEGINNER
    
    def _count_task_categories(self, tasks: List[SeedTask]) -> Dict[str, int]:
        """Count tasks by category."""
        category_counts = defaultdict(int)
        for task in tasks:
            category_counts[task.category.value] += 1
        return dict(category_counts)
    
    def _summarize_context(self, context: TaskGenerationContext) -> Dict[str, Any]:
        """Summarize the task generation context."""
        return {
            "completed_tasks": len(context.completed_tasks),
            "successful_patterns": len(context.successful_patterns),
            "failure_patterns": len(context.failure_patterns),
            "learning_gaps": len(context.learning_gaps),
            "current_difficulty": context.current_difficulty.value
        }
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive task generation statistics."""
        return {
            "total_generated_tasks": len(self.generated_tasks),
            "generation_rounds": len(self.generation_history),
            "task_categories": self._count_task_categories(self.generated_tasks),
            "known_patterns": len(self.known_patterns),
            "successful_strategies": len(self.successful_strategies),
            "failed_strategies": len(self.failed_strategies),
            "unexplored_areas": len(self.unexplored_areas),
            "generation_history": self.generation_history[-5:]  # Last 5 rounds
        }