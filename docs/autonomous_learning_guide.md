# Autonomous Learning Guide for TapeAgents

This guide explains how to use the autonomous learning system in TapeAgents, which enables agents to learn and improve in environments without explicit instructions, using only data sources and tools available in the environment.

## Overview

The autonomous learning system implements an ETO-inspired (Exploration-based Trajectory Optimization) approach that allows TapeAgents to:

1. **Learn from Seed Tasks**: Start with minimal user-provided task descriptions
2. **Generate and Optimize Trajectories**: Use exploration and contrastive learning to improve performance
3. **Warm Memory**: Build a knowledge base from successful patterns
4. **Generate New Tasks**: Autonomously create increasingly challenging tasks
5. **Continuous Improvement**: Iteratively improve through trial and error

## Key Components

### 1. Environment Learner
The main orchestrator that coordinates the entire learning pipeline.

```python
from tapeagents.autonomous_learning import EnvironmentLearner

learner = EnvironmentLearner(
    agent=your_agent,
    llm=your_llm,
    storage_path="./learning_data",
    max_learning_rounds=10,
    tasks_per_round=5
)
```

### 2. Seed Task Manager
Handles initial task input and converts user descriptions into structured tasks.

```python
from tapeagents.autonomous_learning import SeedTaskManager

seed_manager = SeedTaskManager()
task = seed_manager.add_seed_task_from_description(
    description="Learn to analyze data files and extract insights",
    category=TaskCategory.DATA_ANALYSIS,
    difficulty=TaskDifficulty.BEGINNER
)
```

### 3. Trajectory Optimizer
Implements ETO-style optimization using contrastive learning from success/failure pairs.

```python
from tapeagents.autonomous_learning import TrajectoryOptimizer

optimizer = TrajectoryOptimizer(
    agent=agent,
    llm=llm,
    max_exploration_rounds=5,
    trajectories_per_round=10
)
```

### 4. Memory Warmer
Builds and manages a knowledge base from optimized trajectories.

```python
from tapeagents.autonomous_learning import MemoryWarmer

memory_warmer = MemoryWarmer(
    agent=agent,
    max_memory_size=1000,
    similarity_threshold=0.7
)
```

### 5. Task Generator
Autonomously generates new tasks based on learning progress.

```python
from tapeagents.autonomous_learning import AutonomousTaskGenerator

task_generator = AutonomousTaskGenerator(
    llm=llm,
    max_tasks_per_generation=10,
    exploration_bias=0.3
)
```

## Quick Start

### Basic Usage

```python
import os
from tapeagents.agent import Agent
from tapeagents.llms import LiteLLM
from tapeagents.autonomous_learning import EnvironmentLearner

# Set up your LLM and agent
llm = LiteLLM(model_name="gpt-4o-mini")
agent = Agent.create(llm, nodes=[YourNode()])

# Create environment learner
learner = EnvironmentLearner(
    agent=agent,
    llm=llm,
    storage_path="./learning_data"
)

# Define seed tasks (what you want the agent to learn)
seed_tasks = [
    "Explore available data sources and understand their structure",
    "Learn to use file reading tools to process documents",
    "Practice web searching and information extraction",
    "Develop data analysis and visualization skills"
]

# Start learning session
session = learner.start_learning_session(
    seed_task_descriptions=seed_tasks,
    environment_context={
        "available_tools": ["file_reader", "web_search", "data_analyzer"],
        "data_sources": ["local_files", "web_content", "databases"]
    }
)

# Run autonomous learning
results = learner.run_autonomous_learning(
    max_rounds=5,
    convergence_threshold=0.8
)

print(f"Learning completed! Success rate: {results['final_metrics']['success_rate']}")
```

### Advanced Usage with Custom Environment

```python
def custom_environment_executor(tape, task):
    """Custom environment execution function."""
    # Your environment interaction logic here
    # This could interact with real tools, APIs, databases, etc.
    
    # Example: File system interaction
    if "file" in task.description.lower():
        # Simulate file operations
        result = perform_file_operations(tape, task)
        return result
    
    # Example: Web interaction
    elif "web" in task.description.lower():
        # Simulate web operations
        result = perform_web_operations(tape, task)
        return result
    
    # Default: Run agent normally
    return agent.run(tape).get_final_tape()

# Use custom executor
learner = EnvironmentLearner(
    agent=agent,
    llm=llm,
    environment_executor=custom_environment_executor
)
```

## Configuration

### Using Configuration Files

```yaml
# autonomous_learning.yaml
environment_learner:
  max_learning_rounds: 10
  tasks_per_round: 5
  optimization_rounds_per_cycle: 3

trajectory_optimizer:
  max_exploration_rounds: 5
  trajectories_per_round: 10
  success_threshold: 0.7

memory_warmer:
  max_memory_size: 1000
  similarity_threshold: 0.7

task_generator:
  max_tasks_per_generation: 10
  exploration_bias: 0.3
```

```python
from hydra import compose, initialize
from hydra.utils import instantiate

with initialize(config_path="../conf"):
    cfg = compose(config_name="autonomous_learning")
    
    # Create components from config
    learner = EnvironmentLearner(
        agent=agent,
        llm=llm,
        **cfg.environment_learner
    )
```

## Learning Process

### 1. Seed Task Phase
- User provides initial task descriptions
- System analyzes environment context
- Tasks are categorized and structured

### 2. Trajectory Generation Phase
- Agent attempts tasks multiple times with exploration
- Both successful and failed trajectories are collected
- Trajectories are evaluated and scored

### 3. Optimization Phase (ETO-inspired)
- Successful and failed trajectories are paired
- Contrastive learning identifies key differences
- Agent policy is updated based on insights

### 4. Memory Warming Phase
- Optimized trajectories are stored in memory
- Success patterns are extracted and indexed
- Memory templates are created for future use

### 5. Task Generation Phase
- New tasks are generated based on learning progress
- Tasks target identified learning gaps
- Difficulty is progressively increased

### 6. Iteration
- Process repeats with new tasks
- Performance metrics guide adaptation
- Convergence is monitored

## Task Categories

The system supports several task categories:

- **EXPLORATION**: Discovering environment capabilities
- **DATA_ANALYSIS**: Processing and analyzing data
- **TOOL_USAGE**: Learning to use specific tools
- **PROBLEM_SOLVING**: Solving complex problems
- **CREATIVE**: Creative and innovative tasks
- **INTEGRATION**: Combining multiple capabilities

## Difficulty Levels

Tasks are automatically assigned difficulty levels:

- **BEGINNER**: Simple, foundational tasks
- **INTERMEDIATE**: Moderate complexity tasks
- **ADVANCED**: Complex, multi-step tasks
- **EXPERT**: Highly challenging tasks

## Monitoring and Metrics

### Learning Metrics

```python
# Get current learning progress
progress = learner.get_learning_progress()

print(f"Success rate: {progress['current_metrics']['success_rate']}")
print(f"Average reward: {progress['current_metrics']['average_reward']}")
print(f"Tools discovered: {progress['current_metrics']['tools_discovered']}")
print(f"Patterns learned: {progress['current_metrics']['patterns_learned']}")
```

### Session Management

```python
# Save session progress
learner._save_session_progress()

# Load previous session
previous_session = learner.load_session("session_123")

# Get session summary
summary = learner.get_session_summary()
```

## Best Practices

### 1. Seed Task Design
- Provide diverse initial tasks covering different aspects
- Include both simple and moderately complex tasks
- Specify clear success criteria when possible
- Include environment context information

### 2. Environment Setup
- Clearly define available tools and capabilities
- Implement robust environment execution functions
- Handle errors gracefully in environment interactions
- Provide meaningful feedback for agent actions

### 3. Learning Configuration
- Start with conservative learning parameters
- Monitor convergence and adjust thresholds
- Use caching to speed up repeated experiments
- Save learning progress regularly

### 4. Performance Optimization
- Use appropriate LLM models for your use case
- Enable caching for development and testing
- Monitor memory usage and clean up periodically
- Implement efficient environment execution

## Examples

### Data Science Environment

```python
# Seed tasks for data science learning
data_science_tasks = [
    "Load and explore a CSV dataset to understand its structure",
    "Perform basic statistical analysis on numerical columns",
    "Create visualizations to identify patterns in the data",
    "Clean and preprocess data for analysis",
    "Build a simple predictive model",
    "Evaluate model performance and interpret results"
]

# Environment context
data_science_context = {
    "available_tools": [
        "pandas_reader", "numpy_calculator", "matplotlib_plotter",
        "sklearn_modeler", "data_cleaner", "statistics_analyzer"
    ],
    "data_sources": ["csv_files", "json_files", "database_tables"],
    "capabilities": {
        "data_loading": ["csv", "json", "sql"],
        "analysis": ["statistics", "correlation", "distribution"],
        "visualization": ["plots", "charts", "graphs"],
        "modeling": ["regression", "classification", "clustering"]
    }
}
```

### Web Automation Environment

```python
# Seed tasks for web automation learning
web_automation_tasks = [
    "Navigate to a website and extract basic information",
    "Fill out forms and submit data",
    "Search for specific content and collect results",
    "Download files and resources from websites",
    "Monitor websites for changes",
    "Interact with web APIs and services"
]

# Environment context
web_context = {
    "available_tools": [
        "browser", "form_filler", "web_scraper",
        "api_client", "file_downloader", "content_monitor"
    ],
    "data_sources": ["web_pages", "apis", "forms", "downloads"],
    "capabilities": {
        "navigation": ["click", "scroll", "navigate"],
        "interaction": ["type", "select", "submit"],
        "extraction": ["text", "links", "images", "data"],
        "automation": ["workflows", "monitoring", "scheduling"]
    }
}
```

## Troubleshooting

### Common Issues

1. **Low Success Rate**
   - Check environment execution function
   - Verify tool availability and functionality
   - Simplify initial seed tasks
   - Increase exploration rounds

2. **Memory Issues**
   - Reduce max_memory_size
   - Enable automatic cleanup
   - Monitor memory usage patterns
   - Implement memory decay

3. **Slow Learning**
   - Increase trajectories_per_round
   - Adjust learning_rate
   - Use more diverse seed tasks
   - Check LLM response quality

4. **Task Generation Problems**
   - Verify learning context is being updated
   - Check pattern extraction logic
   - Increase creativity_factor
   - Review environment capabilities

### Debugging

```python
# Enable detailed logging
import logging
logging.getLogger("tapeagents.autonomous_learning").setLevel(logging.DEBUG)

# Check learning state
progress = learner.get_learning_progress()
print(f"Learning state: {progress}")

# Examine memory contents
memory_stats = learner.memory_warmer.get_memory_statistics()
print(f"Memory statistics: {memory_stats}")

# Review generated tasks
task_stats = learner.task_generator.get_generation_statistics()
print(f"Task generation: {task_stats}")
```

## Advanced Features

### Progressive Learning

```python
# Generate progressive task sequences
progressive_tasks = task_generator.generate_progressive_tasks(
    base_task=seed_task,
    progression_steps=5
)
```

### Tool-Specific Exploration

```python
# Generate tasks for specific tools
tool_tasks = task_generator.generate_exploration_tasks_for_tools(
    available_tools=["database_query", "api_client"],
    unexplored_tools=["database_query"]
)
```

### Performance-Based Adaptation

```python
# Adapt tasks based on performance
adapted_tasks = task_generator.adapt_tasks_based_on_performance(
    recent_tasks=recent_tasks,
    performance_metrics={"success_rate": 0.6, "average_reward": 0.4},
    optimization_trajectories=trajectories
)
```

## Integration with Existing TapeAgents

The autonomous learning system is designed to work with existing TapeAgents:

```python
# Use with existing agent
existing_agent = your_existing_agent

# Wrap with autonomous learning
learner = EnvironmentLearner(
    agent=existing_agent,
    llm=existing_agent.llm
)

# The agent will learn and improve while maintaining its existing capabilities
```

## Future Enhancements

The autonomous learning system is designed to be extensible:

- **Multi-Agent Learning**: Coordinate learning across multiple agents
- **Cross-Session Memory**: Persist learning across sessions
- **Meta-Learning**: Learn how to learn more effectively
- **Curriculum Learning**: Structured learning progressions
- **Transfer Learning**: Apply learning to new environments

## Support and Resources

- **Examples**: See `examples/autonomous_learning_example.py`
- **Configuration**: See `conf/autonomous_learning.yaml`
- **API Documentation**: See module docstrings
- **GitHub Issues**: Report bugs and feature requests

The autonomous learning system enables TapeAgents to become truly autonomous, learning and improving in any environment with minimal human guidance. This opens up new possibilities for AI agents that can adapt and excel in real-world scenarios.