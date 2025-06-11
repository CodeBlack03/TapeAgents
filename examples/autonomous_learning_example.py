#!/usr/bin/env python3
"""
Autonomous Learning Example for TapeAgents

This example demonstrates how to use the autonomous learning system that:
1. Takes seed tasks from users about the environment
2. Generates and optimizes trajectories using ETO methodology
3. Warms up memory with optimized trajectories
4. Generates new tasks based on past interactions
5. Learns to operate in environments with just data sources and tools

Usage:
    python autonomous_learning_example.py
"""

import os
import logging
from typing import List, Dict, Any

from tapeagents.agent import Agent, Node
from tapeagents.core import Prompt, SetNextNode
from tapeagents.dialog_tape import AssistantStep, DialogTape, UserStep
from tapeagents.llms import LiteLLM, LLMStream
from tapeagents.prompting import tape_to_messages

# Import autonomous learning components
from tapeagents.autonomous_learning import (
    EnvironmentLearner,
    SeedTaskManager,
    TaskCategory,
    TaskDifficulty
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutonomousLearningNode(Node):
    """
    A node designed for autonomous learning that can adapt its behavior
    based on the learning context and environment.
    """
    name: str = "autonomous_learning"

    def make_prompt(self, agent: Agent, tape: DialogTape) -> Prompt:
        """Create a prompt that includes learning context."""
        messages = tape_to_messages(tape)
        
        # Add system message with learning context if available
        if hasattr(agent, 'learning_context'):
            system_message = {
                "role": "system",
                "content": f"You are an autonomous learning agent. Context: {agent.learning_context}"
            }
            messages.insert(0, system_message)
        
        return Prompt(messages=messages)

    def generate_steps(self, agent: Agent, tape: DialogTape, llm_stream: LLMStream):
        """Generate steps with learning awareness."""
        content = llm_stream.get_text()
        
        # Add learning metadata to the step
        step = AssistantStep(content=content)
        if hasattr(agent, 'current_task_category'):
            step.metadata = step.metadata or {}
            step.metadata['task_category'] = agent.current_task_category
        
        yield step
        yield SetNextNode(next_node="autonomous_learning")


def create_autonomous_agent() -> Agent:
    """Create a TapeAgent configured for autonomous learning."""
    # Set up LLM (using OpenAI by default, but could be Azure OpenAI)
    llm = LiteLLM(
        model_name="gpt-4o-mini",
        parameters={
            "temperature": 0.7,
            "max_tokens": 1000
        },
        use_cache=True  # Enable caching for efficiency during learning
    )
    
    # Create agent with autonomous learning node
    agent = Agent[DialogTape].create(llm, nodes=[AutonomousLearningNode()])
    
    # Add learning context attribute
    agent.learning_context = "Autonomous learning mode - adapt based on environment feedback"
    
    return agent


def setup_environment_context() -> Dict[str, Any]:
    """
    Set up environment context with available tools and data sources.
    In a real scenario, this would be discovered dynamically.
    """
    return {
        "available_tools": [
            "file_reader", "web_search", "calculator", "data_analyzer", 
            "code_executor", "database_query", "api_client"
        ],
        "data_sources": [
            "local_files", "web_content", "databases", "apis", "user_input"
        ],
        "capabilities": {
            "text_processing": ["read", "write", "analyze", "summarize"],
            "data_analysis": ["statistics", "visualization", "modeling"],
            "web_interaction": ["search", "scrape", "api_calls"],
            "computation": ["math", "logic", "algorithms"]
        }
    }


def create_seed_tasks() -> List[str]:
    """
    Create initial seed tasks that a user might provide.
    These represent the starting point for autonomous learning.
    """
    return [
        "Explore the available data sources and understand what information is accessible",
        "Learn to use the file reading tools to access and process local documents",
        "Discover how to perform web searches and extract relevant information",
        "Practice using the calculator and data analysis tools for computational tasks",
        "Experiment with combining different tools to solve complex problems",
        "Understand how to query databases and retrieve specific information",
        "Learn to make API calls and integrate external services",
        "Develop strategies for handling errors and unexpected situations"
    ]


def environment_executor(tape: DialogTape, task: Any) -> DialogTape:
    """
    Simulate environment execution for tasks.
    In a real implementation, this would interact with actual tools and data sources.
    """
    # Simulate some environment interaction
    last_step = tape.steps[-1] if tape.steps else None
    
    if last_step and hasattr(last_step, 'content'):
        content = last_step.content.lower()
        
        # Simulate different environment responses based on content
        if "file" in content or "read" in content:
            response = "File reading simulation: Successfully accessed local file system. Found 15 documents."
        elif "web" in content or "search" in content:
            response = "Web search simulation: Found 8 relevant results from web search."
        elif "calculator" in content or "math" in content:
            response = "Calculator simulation: Mathematical computation completed successfully."
        elif "database" in content or "query" in content:
            response = "Database simulation: Query executed, returned 23 records."
        elif "api" in content:
            response = "API simulation: External API call successful, received JSON response."
        else:
            response = "Environment simulation: Task executed with standard response."
        
        # Add environment response to tape
        environment_step = UserStep(content=f"Environment Response: {response}")
        new_tape = DialogTape(steps=tape.steps + [environment_step])
        return new_tape
    
    return tape


def run_basic_autonomous_learning():
    """Run a basic autonomous learning example."""
    print("=== Basic Autonomous Learning Example ===")
    
    # Create agent and environment
    agent = create_autonomous_agent()
    environment_context = setup_environment_context()
    seed_task_descriptions = create_seed_tasks()
    
    # Initialize environment learner
    learner = EnvironmentLearner(
        agent=agent,
        llm=agent.llm,
        storage_path="./autonomous_learning_data",
        max_learning_rounds=3,  # Reduced for demo
        tasks_per_round=3,
        optimization_rounds_per_cycle=2,
        environment_executor=environment_executor
    )
    
    # Start learning session
    print(f"Starting learning session with {len(seed_task_descriptions)} seed tasks...")
    session = learner.start_learning_session(
        seed_task_descriptions=seed_task_descriptions,
        environment_context=environment_context,
        session_id="demo_session_1"
    )
    
    print(f"Session started: {session.session_id}")
    print(f"Environment tools: {len(session.environment_state.available_tools)}")
    print(f"Seed tasks: {len(session.seed_tasks)}")
    
    # Run autonomous learning
    print("\nRunning autonomous learning...")
    results = learner.run_autonomous_learning(
        max_rounds=3,
        convergence_threshold=0.8,
        save_progress=True
    )
    
    # Display results
    print(f"\n=== Learning Results ===")
    print(f"Rounds completed: {results['rounds_completed']}")
    print(f"Convergence achieved: {results['convergence_achieved']}")
    print(f"Final success rate: {results['final_metrics']['success_rate']:.2f}")
    print(f"Average reward: {results['final_metrics']['average_reward']:.2f}")
    print(f"Tasks attempted: {results['final_metrics']['total_tasks_attempted']}")
    print(f"Tasks completed: {results['final_metrics']['total_tasks_completed']}")
    
    # Show learning insights
    print(f"\n=== Learning Insights ===")
    for i, insight in enumerate(results['learning_insights'][-5:], 1):
        print(f"{i}. {insight}")
    
    # Show generated tasks
    print(f"\n=== Generated Tasks ===")
    for i, task in enumerate(results['generated_tasks'][-3:], 1):
        print(f"{i}. {task['description']} (Category: {task['category']}, Difficulty: {task['difficulty']})")
    
    return learner, results


def run_progressive_learning_example():
    """Run an example showing progressive task generation."""
    print("\n=== Progressive Learning Example ===")
    
    # Create a simple seed task manager
    seed_manager = SeedTaskManager()
    
    # Add a base task
    base_task = seed_manager.add_seed_task_from_description(
        description="Learn to analyze data files and extract insights",
        category=TaskCategory.DATA_ANALYSIS,
        difficulty=TaskDifficulty.BEGINNER,
        environment_context={"tools": ["file_reader", "data_analyzer"]},
        success_criteria=["Successfully read data file", "Extract basic statistics"]
    )
    
    print(f"Base task: {base_task.description}")
    
    # Generate progressive variations
    variations = seed_manager.generate_task_variations(base_task, num_variations=3)
    
    print(f"\nGenerated {len(variations)} variations:")
    for i, var in enumerate(variations, 1):
        print(f"{i}. {var.description} (Difficulty: {var.difficulty.value})")
    
    return seed_manager


def run_memory_warming_example():
    """Run an example showing memory warming with optimized trajectories."""
    print("\n=== Memory Warming Example ===")
    
    # Create agent and memory warmer
    agent = create_autonomous_agent()
    
    from tapeagents.autonomous_learning.memory_warmer import MemoryWarmer
    memory_warmer = MemoryWarmer(agent=agent)
    
    # Create some mock optimized trajectories
    from tapeagents.autonomous_learning.datatypes import OptimizedTrajectory, TrajectoryOutcome
    
    mock_trajectories = []
    for i in range(5):
        # Create a simple tape
        tape = DialogTape(steps=[
            UserStep(content=f"Task {i + 1}: Analyze data"),
            AssistantStep(content=f"Analysis complete for task {i + 1}")
        ])
        
        trajectory = OptimizedTrajectory(
            id=f"traj_{i + 1}",
            original_task_id=f"task_{i + 1}",
            tape=tape,
            outcome=TrajectoryOutcome.SUCCESS if i % 2 == 0 else TrajectoryOutcome.FAILURE,
            reward=0.8 if i % 2 == 0 else 0.3,
            optimization_round=1,
            improvements=[f"Improvement {i + 1}"],
            confidence_score=0.7
        )
        mock_trajectories.append(trajectory)
    
    # Create mock seed tasks
    seed_tasks = []
    for i in range(3):
        from tapeagents.autonomous_learning.datatypes import SeedTask
        task = SeedTask(
            id=f"task_{i + 1}",
            description=f"Mock task {i + 1}",
            category=TaskCategory.DATA_ANALYSIS,
            difficulty=TaskDifficulty.BEGINNER
        )
        seed_tasks.append(task)
    
    # Warm up memory
    print("Warming up memory with optimized trajectories...")
    warmup_results = memory_warmer.warm_memory(
        optimized_trajectories=mock_trajectories,
        trajectory_pairs=[],  # Empty for this demo
        seed_tasks=seed_tasks
    )
    
    print(f"Memory warmup results:")
    print(f"- Trajectories stored: {warmup_results['trajectories_stored']}")
    print(f"- Memory categories: {warmup_results['memory_categories']}")
    print(f"- Success templates: {warmup_results['success_templates']}")
    print(f"- Pattern library size: {warmup_results['pattern_library_size']}")
    
    # Test memory retrieval
    print("\nTesting memory retrieval...")
    relevant_memory = memory_warmer.retrieve_relevant_memory(
        task_description="Analyze data and extract insights",
        task_category=TaskCategory.DATA_ANALYSIS,
        max_results=3
    )
    
    print(f"Retrieved {len(relevant_memory)} relevant trajectories")
    for i, traj in enumerate(relevant_memory, 1):
        print(f"{i}. {traj.id} (Reward: {traj.reward:.2f}, Outcome: {traj.outcome.value})")
    
    return memory_warmer


def run_task_generation_example():
    """Run an example showing autonomous task generation."""
    print("\n=== Task Generation Example ===")
    
    # Create task generator
    llm = LiteLLM(model_name="gpt-4o-mini", parameters={"temperature": 0.8})
    
    from tapeagents.autonomous_learning.task_generator import AutonomousTaskGenerator
    from tapeagents.autonomous_learning.datatypes import (
        TaskGenerationContext, EnvironmentState, LearningMetrics
    )
    
    task_generator = AutonomousTaskGenerator(llm=llm)
    
    # Create context for task generation
    context = TaskGenerationContext(
        completed_tasks=["task_1", "task_2", "task_3"],
        successful_patterns=["file reading", "data analysis", "web search"],
        failure_patterns=["complex calculations", "API timeouts"],
        environment_capabilities={"tools": ["file_reader", "web_search", "calculator"]},
        learning_gaps=["database queries", "advanced analytics"],
        current_difficulty=TaskDifficulty.INTERMEDIATE
    )
    
    environment_state = EnvironmentState(
        available_tools=["file_reader", "web_search", "calculator", "database_query"],
        data_sources=["files", "web", "database"],
        discovered_capabilities={"text_processing": True, "basic_math": True}
    )
    
    metrics = LearningMetrics(
        total_tasks_attempted=10,
        total_tasks_completed=7,
        success_rate=0.7,
        average_reward=0.65
    )
    
    # Generate new tasks
    print("Generating new tasks based on learning context...")
    new_tasks = task_generator.generate_tasks(
        context=context,
        environment_state=environment_state,
        learning_metrics=metrics,
        num_tasks=5
    )
    
    print(f"Generated {len(new_tasks)} new tasks:")
    for i, task in enumerate(new_tasks, 1):
        print(f"{i}. {task.description}")
        print(f"   Category: {task.category.value}, Difficulty: {task.difficulty.value}")
        print(f"   Metadata: {task.metadata}")
        print()
    
    return task_generator


def main():
    """Run all autonomous learning examples."""
    print("TapeAgents Autonomous Learning Examples")
    print("=" * 50)
    
    # Set up OpenAI API key (required for LLM calls)
    if "OPENAI_API_KEY" not in os.environ:
        print("Warning: OPENAI_API_KEY not set. Using mock responses.")
        os.environ["OPENAI_API_KEY"] = "mock-key-for-demo"
    
    try:
        # Run basic autonomous learning
        learner, results = run_basic_autonomous_learning()
        
        # Run progressive learning example
        seed_manager = run_progressive_learning_example()
        
        # Run memory warming example
        memory_warmer = run_memory_warming_example()
        
        # Run task generation example
        task_generator = run_task_generation_example()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
        # Show final learning progress
        if learner:
            progress = learner.get_learning_progress()
            print(f"\nFinal Learning Progress:")
            print(f"- Session: {progress['session_id']}")
            print(f"- Rounds completed: {progress['rounds_completed']}")
            print(f"- Success rate: {progress['current_metrics']['success_rate']:.2f}")
            print(f"- Tools discovered: {len(progress['current_metrics']['tools_discovered'])}")
            print(f"- Patterns learned: {len(progress['current_metrics']['patterns_learned'])}")
        
        print(f"\nData saved to: ./autonomous_learning_data/")
        print("You can examine the saved session data to see detailed learning progress.")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"Error running examples: {e}")
        print("This might be due to missing API keys or other configuration issues.")


if __name__ == "__main__":
    main()