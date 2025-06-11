# Autonomous Learning for TapeAgents

This document provides a comprehensive overview of the autonomous learning functionality added to TapeAgents, which enables agents to learn and improve in environments without explicit instructions.

## 🎯 Overview

The autonomous learning system implements an ETO-inspired (Exploration-based Trajectory Optimization) approach that allows TapeAgents to:

1. **Start with Seed Tasks**: Take minimal user-provided task descriptions about the environment
2. **Generate & Optimize Trajectories**: Use exploration and contrastive learning to improve performance  
3. **Warm Memory**: Build a knowledge base from successful patterns and optimized trajectories
4. **Generate New Tasks**: Autonomously create increasingly challenging tasks based on past interactions
5. **Learn Environments**: Discover and master environments with just data sources and tools

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Environment Learner                         │
│                   (Main Orchestrator)                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
┌─────────┐    ┌─────────────┐    ┌─────────────┐
│  Seed   │    │ Trajectory  │    │   Memory    │
│  Task   │    │ Optimizer   │    │   Warmer    │
│Manager  │    │   (ETO)     │    │             │
└─────────┘    └─────────────┘    └─────────────┘
    │                 │                 │
    └─────────────────┼─────────────────┘
                      │
                      ▼
              ┌─────────────┐
              │    Task     │
              │ Generator   │
              │             │
              └─────────────┘
```

## 🚀 Quick Start

### Basic Usage

```python
from tapeagents.agent import Agent
from tapeagents.llms import LiteLLM
from tapeagents.autonomous_learning import EnvironmentLearner

# Set up your agent
llm = LiteLLM(model_name="gpt-4o-mini")
agent = Agent.create(llm, nodes=[YourNode()])

# Create environment learner
learner = EnvironmentLearner(
    agent=agent,
    llm=llm,
    storage_path="./learning_data"
)

# Define what you want the agent to learn
seed_tasks = [
    "Explore available data sources and understand their structure",
    "Learn to use file reading tools to process documents", 
    "Practice web searching and information extraction",
    "Develop data analysis and visualization skills"
]

# Start autonomous learning
session = learner.start_learning_session(
    seed_task_descriptions=seed_tasks,
    environment_context={
        "available_tools": ["file_reader", "web_search", "data_analyzer"],
        "data_sources": ["local_files", "web_content", "databases"]
    }
)

# Run the learning process
results = learner.run_autonomous_learning(
    max_rounds=5,
    convergence_threshold=0.8
)

print(f"Learning completed! Success rate: {results['final_metrics']['success_rate']}")
```

### Advanced Usage with Custom Environment

```python
def custom_environment_executor(tape, task):
    """Custom environment that interacts with real tools and data."""
    
    # File system operations
    if "file" in task.description.lower():
        return handle_file_operations(tape, task)
    
    # Web operations  
    elif "web" in task.description.lower():
        return handle_web_operations(tape, task)
    
    # Database operations
    elif "database" in task.description.lower():
        return handle_database_operations(tape, task)
    
    # Default: run agent normally
    return agent.run(tape).get_final_tape()

# Use custom environment
learner = EnvironmentLearner(
    agent=agent,
    llm=llm,
    environment_executor=custom_environment_executor
)
```

## 📋 Key Components

### 1. SeedTaskManager
Handles initial task input and converts user descriptions into structured tasks.

```python
from tapeagents.autonomous_learning import SeedTaskManager, TaskCategory

manager = SeedTaskManager()
task = manager.add_seed_task_from_description(
    description="Learn to analyze CSV files and extract insights",
    category=TaskCategory.DATA_ANALYSIS,
    environment_context={"tools": ["pandas", "matplotlib"]},
    success_criteria=["Load CSV", "Generate summary statistics"]
)
```

### 2. TrajectoryOptimizer  
Implements ETO-style optimization using contrastive learning from success/failure pairs.

```python
from tapeagents.autonomous_learning import TrajectoryOptimizer

optimizer = TrajectoryOptimizer(
    agent=agent,
    llm=llm,
    max_exploration_rounds=5,
    trajectories_per_round=10,
    success_threshold=0.7
)

# Optimize trajectories for tasks
optimized_trajectories = optimizer.optimize_trajectories(
    seed_tasks=tasks,
    environment_executor=environment_executor
)
```

### 3. MemoryWarmer
Builds and manages a knowledge base from optimized trajectories.

```python
from tapeagents.autonomous_learning import MemoryWarmer

memory_warmer = MemoryWarmer(
    agent=agent,
    max_memory_size=1000,
    similarity_threshold=0.7
)

# Warm memory with optimized trajectories
memory_warmer.warm_memory(
    optimized_trajectories=trajectories,
    trajectory_pairs=pairs,
    seed_tasks=tasks
)

# Retrieve relevant memory for new tasks
relevant_memory = memory_warmer.retrieve_relevant_memory(
    task_description="analyze data files",
    max_results=5
)
```

### 4. AutonomousTaskGenerator
Generates new tasks based on learning progress and environment exploration.

```python
from tapeagents.autonomous_learning import AutonomousTaskGenerator

task_generator = AutonomousTaskGenerator(
    llm=llm,
    max_tasks_per_generation=10,
    exploration_bias=0.3,
    creativity_factor=0.2
)

# Generate new tasks based on context
new_tasks = task_generator.generate_tasks(
    context=generation_context,
    environment_state=env_state,
    learning_metrics=metrics
)
```

### 5. EnvironmentLearner
Main orchestrator that coordinates the complete learning pipeline.

```python
from tapeagents.autonomous_learning import EnvironmentLearner

learner = EnvironmentLearner(
    agent=agent,
    llm=llm,
    max_learning_rounds=10,
    tasks_per_round=5,
    optimization_rounds_per_cycle=3
)
```

## 🔄 Learning Process

### Phase 1: Seed Task Analysis
- User provides initial task descriptions
- System analyzes environment context  
- Tasks are categorized and structured
- Environment capabilities are discovered

### Phase 2: Trajectory Generation & Optimization
- Agent attempts tasks multiple times with exploration
- Both successful and failed trajectories are collected
- Trajectories are evaluated and scored
- Contrastive learning identifies success/failure patterns

### Phase 3: Memory Warming
- Optimized trajectories are stored in memory
- Success patterns are extracted and indexed
- Memory templates are created for future use
- Knowledge base is built for quick retrieval

### Phase 4: Autonomous Task Generation
- New tasks are generated based on learning progress
- Tasks target identified learning gaps
- Difficulty is progressively increased
- Creative combinations are explored

### Phase 5: Iteration & Improvement
- Process repeats with new tasks
- Performance metrics guide adaptation
- Convergence is monitored
- Continuous improvement through trial and error

## 📊 Monitoring & Metrics

### Learning Progress

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

## 🎯 Use Cases

### 1. Data Science Environment
```python
data_science_tasks = [
    "Load and explore CSV datasets",
    "Perform statistical analysis", 
    "Create data visualizations",
    "Build predictive models",
    "Evaluate model performance"
]

data_science_context = {
    "available_tools": ["pandas", "numpy", "matplotlib", "sklearn"],
    "data_sources": ["csv_files", "databases", "apis"],
    "capabilities": {
        "data_loading": ["csv", "json", "sql"],
        "analysis": ["statistics", "correlation"],
        "visualization": ["plots", "charts"],
        "modeling": ["regression", "classification"]
    }
}
```

### 2. Web Automation Environment
```python
web_automation_tasks = [
    "Navigate websites and extract information",
    "Fill out forms and submit data",
    "Search for content and collect results", 
    "Download files and resources",
    "Monitor websites for changes"
]

web_context = {
    "available_tools": ["browser", "scraper", "form_filler", "downloader"],
    "data_sources": ["web_pages", "apis", "forms"],
    "capabilities": {
        "navigation": ["click", "scroll", "navigate"],
        "interaction": ["type", "select", "submit"],
        "extraction": ["text", "links", "images"]
    }
}
```

### 3. File Processing Environment
```python
file_processing_tasks = [
    "Read and parse different file formats",
    "Extract text and metadata from documents",
    "Convert between file formats",
    "Organize and categorize files",
    "Search and filter file contents"
]

file_context = {
    "available_tools": ["file_reader", "text_extractor", "converter", "organizer"],
    "data_sources": ["documents", "images", "archives"],
    "capabilities": {
        "reading": ["pdf", "docx", "txt", "csv"],
        "extraction": ["text", "metadata", "images"],
        "conversion": ["format_conversion", "encoding"]
    }
}
```

## ⚙️ Configuration

### Using Configuration Files

```yaml
# autonomous_learning.yaml
environment_learner:
  max_learning_rounds: 10
  tasks_per_round: 5
  optimization_rounds_per_cycle: 3
  convergence_threshold: 0.9

trajectory_optimizer:
  max_exploration_rounds: 5
  trajectories_per_round: 10
  success_threshold: 0.7
  learning_rate: 0.1

memory_warmer:
  max_memory_size: 1000
  similarity_threshold: 0.7
  memory_decay_factor: 0.95

task_generator:
  max_tasks_per_generation: 10
  exploration_bias: 0.3
  creativity_factor: 0.2
```

### Loading Configuration

```python
from hydra import compose, initialize
from hydra.utils import instantiate

with initialize(config_path="../conf"):
    cfg = compose(config_name="autonomous_learning")
    
    learner = EnvironmentLearner(
        agent=agent,
        llm=llm,
        **cfg.environment_learner
    )
```

## 🔧 Best Practices

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

## 🐛 Troubleshooting

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

## 📁 Files Added

### Core Module Files
- `tapeagents/autonomous_learning/__init__.py` - Module initialization
- `tapeagents/autonomous_learning/datatypes.py` - Data structures and types
- `tapeagents/autonomous_learning/seed_task_manager.py` - Seed task management
- `tapeagents/autonomous_learning/trajectory_optimizer.py` - ETO-style optimization
- `tapeagents/autonomous_learning/memory_warmer.py` - Memory management
- `tapeagents/autonomous_learning/task_generator.py` - Autonomous task generation
- `tapeagents/autonomous_learning/environment_learner.py` - Main orchestrator

### Examples and Documentation
- `examples/autonomous_learning_example.py` - Comprehensive examples
- `docs/autonomous_learning_guide.md` - Detailed documentation
- `conf/autonomous_learning.yaml` - Configuration template

### Testing
- `tests/test_autonomous_learning.py` - Comprehensive test suite

## 🔮 Future Enhancements

The autonomous learning system is designed to be extensible:

- **Multi-Agent Learning**: Coordinate learning across multiple agents
- **Cross-Session Memory**: Persist learning across sessions  
- **Meta-Learning**: Learn how to learn more effectively
- **Curriculum Learning**: Structured learning progressions
- **Transfer Learning**: Apply learning to new environments
- **Reinforcement Learning**: Advanced reward-based optimization

## 🤝 Integration with Existing TapeAgents

The autonomous learning system seamlessly integrates with existing TapeAgents:

```python
# Use with existing agent
existing_agent = your_existing_agent

# Wrap with autonomous learning
learner = EnvironmentLearner(
    agent=existing_agent,
    llm=existing_agent.llm
)

# The agent will learn and improve while maintaining existing capabilities
```

## 📚 Resources

- **Examples**: `examples/autonomous_learning_example.py`
- **Configuration**: `conf/autonomous_learning.yaml`  
- **Documentation**: `docs/autonomous_learning_guide.md`
- **Tests**: `tests/test_autonomous_learning.py`
- **API Reference**: Module docstrings and type hints

## 🎉 Getting Started

1. **Install TapeAgents** with the autonomous learning module
2. **Run the example**: `python examples/autonomous_learning_example.py`
3. **Read the guide**: `docs/autonomous_learning_guide.md`
4. **Customize for your environment** using the configuration templates
5. **Start learning!** Define your seed tasks and let the agent explore

The autonomous learning system enables TapeAgents to become truly autonomous, learning and improving in any environment with minimal human guidance. This opens up new possibilities for AI agents that can adapt and excel in real-world scenarios.