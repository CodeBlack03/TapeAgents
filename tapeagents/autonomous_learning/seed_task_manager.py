"""
Seed Task Manager for autonomous learning.

This module handles the initial task input from users and manages the seed tasks
that bootstrap the autonomous learning process.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .datatypes import SeedTask, TaskCategory, TaskDifficulty, EnvironmentState

logger = logging.getLogger(__name__)


class SeedTaskManager:
    """
    Manages seed tasks for autonomous learning.
    
    The SeedTaskManager handles:
    - Taking initial task descriptions from users
    - Converting them into structured SeedTask objects
    - Analyzing environment context from task descriptions
    - Generating variations of seed tasks
    - Persisting and loading seed tasks
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the SeedTaskManager.
        
        Args:
            storage_path: Path to store seed tasks (optional)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.seed_tasks: List[SeedTask] = []
        self.environment_context: Dict[str, Any] = {}
        
        if self.storage_path and self.storage_path.exists():
            self.load_seed_tasks()
    
    def add_seed_task_from_description(
        self,
        description: str,
        category: Optional[TaskCategory] = None,
        difficulty: Optional[TaskDifficulty] = None,
        environment_context: Optional[Dict[str, Any]] = None,
        success_criteria: Optional[List[str]] = None,
        examples: Optional[List[str]] = None
    ) -> SeedTask:
        """
        Create a seed task from a natural language description.
        
        Args:
            description: Natural language description of the task
            category: Task category (auto-detected if not provided)
            difficulty: Task difficulty (auto-detected if not provided)
            environment_context: Context about the environment
            success_criteria: Criteria for task success
            examples: Example inputs or scenarios
            
        Returns:
            Created SeedTask object
        """
        # Auto-detect category if not provided
        if category is None:
            category = self._detect_task_category(description)
        
        # Auto-detect difficulty if not provided
        if difficulty is None:
            difficulty = self._detect_task_difficulty(description)
        
        # Generate task ID
        task_id = f"seed_{len(self.seed_tasks) + 1}_{category.value}"
        
        # Create the seed task
        seed_task = SeedTask(
            id=task_id,
            description=description,
            category=category,
            difficulty=difficulty,
            environment_context=environment_context or {},
            success_criteria=success_criteria or [],
            examples=examples or [],
            metadata={"auto_generated": False, "source": "user_input"}
        )
        
        self.seed_tasks.append(seed_task)
        self._update_environment_context(seed_task)
        
        logger.info(f"Added seed task: {task_id} ({category.value}, {difficulty.value})")
        return seed_task
    
    def add_multiple_seed_tasks(self, task_descriptions: List[str]) -> List[SeedTask]:
        """
        Add multiple seed tasks from a list of descriptions.
        
        Args:
            task_descriptions: List of task descriptions
            
        Returns:
            List of created SeedTask objects
        """
        created_tasks = []
        for description in task_descriptions:
            task = self.add_seed_task_from_description(description)
            created_tasks.append(task)
        
        logger.info(f"Added {len(created_tasks)} seed tasks")
        return created_tasks
    
    def generate_task_variations(self, base_task: SeedTask, num_variations: int = 3) -> List[SeedTask]:
        """
        Generate variations of a base seed task.
        
        Args:
            base_task: Base task to create variations from
            num_variations: Number of variations to generate
            
        Returns:
            List of task variations
        """
        variations = []
        
        for i in range(num_variations):
            variation_id = f"{base_task.id}_var_{i + 1}"
            
            # Create variations by modifying difficulty, context, or criteria
            if i == 0:
                # Easier variation
                new_difficulty = self._get_easier_difficulty(base_task.difficulty)
                variation_desc = f"Simplified version: {base_task.description}"
            elif i == 1:
                # Harder variation
                new_difficulty = self._get_harder_difficulty(base_task.difficulty)
                variation_desc = f"Advanced version: {base_task.description}"
            else:
                # Different context variation
                new_difficulty = base_task.difficulty
                variation_desc = f"Alternative approach: {base_task.description}"
            
            variation = SeedTask(
                id=variation_id,
                description=variation_desc,
                category=base_task.category,
                difficulty=new_difficulty,
                environment_context=base_task.environment_context.copy(),
                success_criteria=base_task.success_criteria.copy(),
                examples=base_task.examples.copy(),
                metadata={
                    "auto_generated": True,
                    "source": "variation",
                    "base_task_id": base_task.id,
                    "variation_type": ["easier", "harder", "alternative"][i]
                }
            )
            
            variations.append(variation)
        
        self.seed_tasks.extend(variations)
        logger.info(f"Generated {len(variations)} variations for task {base_task.id}")
        return variations
    
    def analyze_environment_from_tasks(self) -> EnvironmentState:
        """
        Analyze the environment based on all seed tasks.
        
        Returns:
            EnvironmentState object with discovered information
        """
        tools = set()
        data_sources = set()
        capabilities = {}
        
        for task in self.seed_tasks:
            # Extract tools mentioned in task descriptions
            task_tools = self._extract_tools_from_description(task.description)
            tools.update(task_tools)
            
            # Extract data sources
            task_data_sources = self._extract_data_sources_from_description(task.description)
            data_sources.update(task_data_sources)
            
            # Extract capabilities from environment context
            if task.environment_context:
                for key, value in task.environment_context.items():
                    if key not in capabilities:
                        capabilities[key] = []
                    if isinstance(value, list):
                        capabilities[key].extend(value)
                    else:
                        capabilities[key].append(value)
        
        environment_state = EnvironmentState(
            available_tools=list(tools),
            data_sources=list(data_sources),
            discovered_capabilities=capabilities
        )
        
        logger.info(f"Analyzed environment: {len(tools)} tools, {len(data_sources)} data sources")
        return environment_state
    
    def get_tasks_by_category(self, category: TaskCategory) -> List[SeedTask]:
        """Get all seed tasks of a specific category."""
        return [task for task in self.seed_tasks if task.category == category]
    
    def get_tasks_by_difficulty(self, difficulty: TaskDifficulty) -> List[SeedTask]:
        """Get all seed tasks of a specific difficulty."""
        return [task for task in self.seed_tasks if task.difficulty == difficulty]
    
    def save_seed_tasks(self, path: Optional[str] = None) -> None:
        """
        Save seed tasks to file.
        
        Args:
            path: Path to save to (uses default storage_path if not provided)
        """
        save_path = Path(path) if path else self.storage_path
        if not save_path:
            raise ValueError("No storage path provided")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "seed_tasks": [task.model_dump() for task in self.seed_tasks],
            "environment_context": self.environment_context
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.seed_tasks)} seed tasks to {save_path}")
    
    def load_seed_tasks(self, path: Optional[str] = None) -> None:
        """
        Load seed tasks from file.
        
        Args:
            path: Path to load from (uses default storage_path if not provided)
        """
        load_path = Path(path) if path else self.storage_path
        if not load_path or not load_path.exists():
            logger.warning(f"Seed tasks file not found: {load_path}")
            return
        
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        self.seed_tasks = [SeedTask.model_validate(task_data) for task_data in data["seed_tasks"]]
        self.environment_context = data.get("environment_context", {})
        
        logger.info(f"Loaded {len(self.seed_tasks)} seed tasks from {load_path}")
    
    def _detect_task_category(self, description: str) -> TaskCategory:
        """Auto-detect task category from description."""
        description_lower = description.lower()
        
        # Simple keyword-based detection
        if any(word in description_lower for word in ["explore", "find", "discover", "search"]):
            return TaskCategory.EXPLORATION
        elif any(word in description_lower for word in ["analyze", "data", "statistics", "report"]):
            return TaskCategory.DATA_ANALYSIS
        elif any(word in description_lower for word in ["tool", "use", "execute", "run"]):
            return TaskCategory.TOOL_USAGE
        elif any(word in description_lower for word in ["solve", "problem", "fix", "debug"]):
            return TaskCategory.PROBLEM_SOLVING
        elif any(word in description_lower for word in ["create", "generate", "design", "build"]):
            return TaskCategory.CREATIVE
        elif any(word in description_lower for word in ["combine", "integrate", "connect", "merge"]):
            return TaskCategory.INTEGRATION
        else:
            return TaskCategory.EXPLORATION  # Default
    
    def _detect_task_difficulty(self, description: str) -> TaskDifficulty:
        """Auto-detect task difficulty from description."""
        description_lower = description.lower()
        
        # Simple keyword-based detection
        if any(word in description_lower for word in ["simple", "basic", "easy", "beginner"]):
            return TaskDifficulty.BEGINNER
        elif any(word in description_lower for word in ["complex", "advanced", "difficult", "expert"]):
            return TaskDifficulty.ADVANCED
        elif any(word in description_lower for word in ["intermediate", "moderate"]):
            return TaskDifficulty.INTERMEDIATE
        else:
            return TaskDifficulty.BEGINNER  # Default to beginner
    
    def _get_easier_difficulty(self, current: TaskDifficulty) -> TaskDifficulty:
        """Get an easier difficulty level."""
        difficulty_order = [TaskDifficulty.BEGINNER, TaskDifficulty.INTERMEDIATE, 
                          TaskDifficulty.ADVANCED, TaskDifficulty.EXPERT]
        current_index = difficulty_order.index(current)
        return difficulty_order[max(0, current_index - 1)]
    
    def _get_harder_difficulty(self, current: TaskDifficulty) -> TaskDifficulty:
        """Get a harder difficulty level."""
        difficulty_order = [TaskDifficulty.BEGINNER, TaskDifficulty.INTERMEDIATE, 
                          TaskDifficulty.ADVANCED, TaskDifficulty.EXPERT]
        current_index = difficulty_order.index(current)
        return difficulty_order[min(len(difficulty_order) - 1, current_index + 1)]
    
    def _extract_tools_from_description(self, description: str) -> List[str]:
        """Extract potential tool names from task description."""
        # Simple extraction - in practice, this could be more sophisticated
        tools = []
        description_lower = description.lower()
        
        # Common tool keywords
        tool_keywords = [
            "calculator", "browser", "search", "database", "api", "file", "editor",
            "terminal", "command", "script", "query", "request", "download", "upload"
        ]
        
        for keyword in tool_keywords:
            if keyword in description_lower:
                tools.append(keyword)
        
        return tools
    
    def _extract_data_sources_from_description(self, description: str) -> List[str]:
        """Extract potential data sources from task description."""
        data_sources = []
        description_lower = description.lower()
        
        # Common data source keywords
        data_keywords = [
            "database", "file", "csv", "json", "xml", "api", "web", "document",
            "spreadsheet", "table", "dataset", "log", "report"
        ]
        
        for keyword in data_keywords:
            if keyword in description_lower:
                data_sources.append(keyword)
        
        return data_sources
    
    def _update_environment_context(self, task: SeedTask) -> None:
        """Update global environment context based on a new task."""
        if task.environment_context:
            for key, value in task.environment_context.items():
                if key not in self.environment_context:
                    self.environment_context[key] = []
                
                if isinstance(value, list):
                    self.environment_context[key].extend(value)
                else:
                    self.environment_context[key].append(value)