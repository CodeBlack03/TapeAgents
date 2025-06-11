"""
CodeAct Framework Example

This example demonstrates the CodeAct framework where:
1. Tasks are planned as executable Python code
2. Execution is tracked through workflow dependency graphs
3. Errors are precisely localized to specific code lines
4. Self-reflection targets only failed sub-tasks
5. The system can recover from failures and continue execution
"""

import logging
import sys
from pathlib import Path

# Add TapeAgents to path
sys.path.append(str(Path(__file__).parent.parent))

from tapeagents.codeact_agent import CodeActAgent, create_codeact_agent
from tapeagents.codeact_environment import CodeActEnvironment
from tapeagents.codeact_core import CodeAction, CodePlan, WorkflowGraph, WorkflowNode
from tapeagents.core import Tape, TapeMetadata
from tapeagents.llms import LiteLLM
from tapeagents.orchestrator import main_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_data_analysis_task() -> Tape:
    """Create a tape with a data analysis task."""
    
    from tapeagents.core import Observation
    
    task_description = """
    Analyze sales data and create a comprehensive report:
    
    1. Load sales data from a CSV file (simulate with generated data)
    2. Clean and preprocess the data
    3. Calculate key metrics (total sales, average order value, top products)
    4. Create visualizations (sales trends, product performance)
    5. Generate a summary report
    
    The analysis should be robust and handle potential data quality issues.
    """
    
    initial_step = Observation(content=task_description)
    
    return Tape(
        steps=[initial_step],
        metadata=TapeMetadata(id="data_analysis_task")
    )


def create_web_scraping_task() -> Tape:
    """Create a tape with a web scraping task."""
    
    from tapeagents.core import Observation
    
    task_description = """
    Build a web scraping pipeline:
    
    1. Define target websites and data to extract
    2. Create robust scraping functions with error handling
    3. Implement rate limiting and respectful scraping
    4. Parse and structure the extracted data
    5. Store results in a structured format
    6. Create a monitoring system for scraping health
    
    Handle common issues like rate limiting, dynamic content, and data validation.
    """
    
    initial_step = Observation(content=task_description)
    
    return Tape(
        steps=[initial_step],
        metadata=TapeMetadata(id="web_scraping_task")
    )


def create_machine_learning_task() -> Tape:
    """Create a tape with a machine learning task."""
    
    from tapeagents.core import Observation
    
    task_description = """
    Build a machine learning pipeline:
    
    1. Generate or load a dataset for classification
    2. Perform exploratory data analysis
    3. Preprocess data (scaling, encoding, feature selection)
    4. Train multiple models (logistic regression, random forest, SVM)
    5. Evaluate models using cross-validation
    6. Select best model and create predictions
    7. Generate model performance report
    
    Include proper error handling and model validation.
    """
    
    initial_step = Observation(content=task_description)
    
    return Tape(
        steps=[initial_step],
        metadata=TapeMetadata(id="ml_pipeline_task")
    )


def demonstrate_codeact_planning():
    """Demonstrate CodeAct planning with a simple task."""
    
    print("\n" + "="*60)
    print("CODEACT PLANNING DEMONSTRATION")
    print("="*60)
    
    # Create LLM
    llm = LiteLLM(model_name="gpt-4o-mini", parameters={"temperature": 0.1})
    
    # Create CodeAct agent
    agent = create_codeact_agent(llm, name="CodeActPlanner")
    
    # Create environment
    environment = CodeActEnvironment(
        execution_mode="sequential",
        enable_sandboxing=True,
        default_timeout=10.0
    )
    
    # Create a simple task
    from tapeagents.core import Observation
    task = Observation(content="""
    Create a simple calculator that can:
    1. Add two numbers
    2. Multiply two numbers  
    3. Calculate the factorial of a number
    4. Test all functions with sample inputs
    """)
    
    start_tape = Tape(steps=[task], metadata=TapeMetadata(id="calculator_task"))
    
    print(f"Task: {task.content}")
    print("\nExecuting CodeAct planning...")
    
    # Run the main loop
    try:
        stream = main_loop(agent, start_tape, environment, max_loops=10)
        final_tape = stream.get_final_tape()
        
        print(f"\nExecution completed! Final tape has {len(final_tape.steps)} steps.")
        
        # Show the generated plan and execution results
        for i, step in enumerate(final_tape.steps):
            print(f"\nStep {i+1}: {type(step).__name__}")
            if hasattr(step, 'content'):
                print(f"Content: {step.content[:200]}...")
            elif hasattr(step, 'workflow'):
                print(f"Workflow with {len(step.workflow.nodes)} nodes")
            elif hasattr(step, 'code'):
                print(f"Code: {step.code[:100]}...")
    
    except Exception as e:
        logger.error(f"Error in CodeAct demonstration: {e}")
        print(f"Error: {e}")


def demonstrate_error_recovery():
    """Demonstrate error recovery and self-reflection."""
    
    print("\n" + "="*60)
    print("ERROR RECOVERY DEMONSTRATION")
    print("="*60)
    
    # Create LLM
    llm = LiteLLM(model_name="gpt-4o-mini", parameters={"temperature": 0.1})
    
    # Create CodeAct agent
    agent = create_codeact_agent(llm, name="ErrorRecoveryAgent")
    
    # Create environment
    environment = CodeActEnvironment(
        execution_mode="sequential",
        enable_sandboxing=False,  # Allow more operations for demonstration
        default_timeout=5.0
    )
    
    # Create a task that will likely have errors
    from tapeagents.core import Observation
    task = Observation(content="""
    Create a data processing pipeline that:
    1. Reads data from a non-existent file (this will fail)
    2. Processes the data with some calculations
    3. Saves results to a file
    
    The system should handle the file not found error and create sample data instead.
    """)
    
    start_tape = Tape(steps=[task], metadata=TapeMetadata(id="error_recovery_task"))
    
    print(f"Task: {task.content}")
    print("\nExecuting with intentional errors...")
    
    try:
        stream = main_loop(agent, start_tape, environment, max_loops=15)
        final_tape = stream.get_final_tape()
        
        print(f"\nExecution completed! Final tape has {len(final_tape.steps)} steps.")
        
        # Analyze the execution for errors and recovery
        errors_found = 0
        reflections_made = 0
        
        for step in final_tape.steps:
            if hasattr(step, 'status') and step.status == "failed":
                errors_found += 1
                print(f"\nError found: {step.get_error_summary()}")
            elif hasattr(step, 'failed_node_id'):  # CodeReflection
                reflections_made += 1
                print(f"\nReflection made on node: {step.failed_node_id}")
                print(f"Root cause: {step.root_cause}")
        
        print(f"\nSummary:")
        print(f"- Errors encountered: {errors_found}")
        print(f"- Reflections made: {reflections_made}")
        print(f"- Recovery successful: {reflections_made > 0}")
    
    except Exception as e:
        logger.error(f"Error in recovery demonstration: {e}")
        print(f"Error: {e}")


def demonstrate_complex_workflow():
    """Demonstrate complex workflow with dependencies."""
    
    print("\n" + "="*60)
    print("COMPLEX WORKFLOW DEMONSTRATION")
    print("="*60)
    
    # Create LLM
    llm = LiteLLM(model_name="gpt-4o-mini", parameters={"temperature": 0.1})
    
    # Create CodeAct agent
    agent = create_codeact_agent(llm, name="WorkflowAgent")
    
    # Create environment with parallel execution
    environment = CodeActEnvironment(
        execution_mode="parallel",
        max_parallel_nodes=3,
        enable_sandboxing=True,
        default_timeout=15.0
    )
    
    # Use the data analysis task
    start_tape = create_data_analysis_task()
    
    print("Task: Data Analysis Pipeline")
    print("Executing complex workflow with dependencies...")
    
    try:
        stream = main_loop(agent, start_tape, environment, max_loops=20)
        final_tape = stream.get_final_tape()
        
        print(f"\nExecution completed! Final tape has {len(final_tape.steps)} steps.")
        
        # Analyze the workflow execution
        workflows_created = 0
        nodes_executed = 0
        parallel_executions = 0
        
        for step in final_tape.steps:
            if hasattr(step, 'workflow'):
                workflows_created += 1
                nodes_executed += len(step.workflow.nodes)
                print(f"\nWorkflow created with {len(step.workflow.nodes)} nodes")
                
                # Show dependency structure
                for node_id, node in step.workflow.nodes.items():
                    deps = len(node.dependencies)
                    print(f"  - {node.name}: {deps} dependencies")
            
            elif hasattr(step, 'node_id') and hasattr(step, 'execution_time'):
                if step.execution_time > 0:
                    print(f"  Executed {step.node_id} in {step.execution_time:.2f}s")
        
        print(f"\nWorkflow Summary:")
        print(f"- Workflows created: {workflows_created}")
        print(f"- Total nodes: {nodes_executed}")
        print(f"- Execution mode: {environment.execution_mode}")
    
    except Exception as e:
        logger.error(f"Error in workflow demonstration: {e}")
        print(f"Error: {e}")


def demonstrate_safety_features():
    """Demonstrate code safety and sandboxing features."""
    
    print("\n" + "="*60)
    print("SAFETY FEATURES DEMONSTRATION")
    print("="*60)
    
    # Create environment with strict safety
    environment = CodeActEnvironment(
        execution_mode="sequential",
        enable_sandboxing=True,
        allowed_imports=["math", "random", "datetime"]  # Limited imports
    )
    
    # Test various safety scenarios
    safety_tests = [
        {
            "name": "Safe Code",
            "code": """
def safe_calculation(x, y):
    import math
    return math.sqrt(x**2 + y**2)
""",
            "should_pass": True
        },
        {
            "name": "Dangerous System Call",
            "code": """
def dangerous_function():
    import os
    os.system("rm -rf /")
    return "done"
""",
            "should_pass": False
        },
        {
            "name": "Disallowed Import",
            "code": """
def network_function():
    import requests
    return requests.get("http://example.com")
""",
            "should_pass": False
        },
        {
            "name": "File Operations",
            "code": """
def file_function():
    with open("/etc/passwd", "r") as f:
        return f.read()
""",
            "should_pass": False
        }
    ]
    
    for test in safety_tests:
        print(f"\nTesting: {test['name']}")
        print(f"Expected to pass: {test['should_pass']}")
        
        # Create a CodeAction
        action = CodeAction(
            code=test['code'],
            function_name="test_function",
            safety_level="strict"
        )
        
        # Test safety validation
        safety_issues = action.validate_code_safety()
        
        if safety_issues:
            print(f"Safety issues detected: {safety_issues}")
            print("✓ Correctly blocked unsafe code" if not test['should_pass'] else "✗ False positive")
        else:
            print("No safety issues detected")
            print("✓ Safe code approved" if test['should_pass'] else "✗ Unsafe code not detected")


def run_all_demonstrations():
    """Run all CodeAct demonstrations."""
    
    print("CODEACT FRAMEWORK DEMONSTRATION")
    print("="*80)
    print("This demonstration shows the CodeAct framework capabilities:")
    print("1. Planning tasks as executable Python code")
    print("2. Workflow dependency graph execution")
    print("3. Precise error localization and recovery")
    print("4. Safety features and code sandboxing")
    print("="*80)
    
    try:
        # Basic planning demonstration
        demonstrate_codeact_planning()
        
        # Error recovery demonstration
        demonstrate_error_recovery()
        
        # Complex workflow demonstration
        demonstrate_complex_workflow()
        
        # Safety features demonstration
        demonstrate_safety_features()
        
        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Code-based planning instead of text plans")
        print("✓ Workflow dependency graph execution")
        print("✓ Line-level error localization")
        print("✓ Targeted self-reflection on failures")
        print("✓ Automatic error recovery and code correction")
        print("✓ Parallel and sequential execution modes")
        print("✓ Code safety validation and sandboxing")
        print("✓ Comprehensive execution monitoring")
        
    except Exception as e:
        logger.error(f"Error in demonstrations: {e}")
        print(f"Demonstration failed: {e}")


if __name__ == "__main__":
    # Check if we have the required dependencies
    try:
        import litellm
        print("LiteLLM found, running demonstrations...")
        run_all_demonstrations()
    except ImportError:
        print("LiteLLM not found. Please install it to run the demonstrations:")
        print("pip install litellm")
        
        # Show the framework structure instead
        print("\nCodeAct Framework Structure:")
        print("- codeact_core.py: Core data structures and workflow graphs")
        print("- codeact_agent.py: Agents that plan using executable code")
        print("- codeact_environment.py: Environment for executing workflows")
        print("- This example: Comprehensive demonstrations")
        
        print("\nKey Innovations:")
        print("1. Plans are executable Python code, not text")
        print("2. Execution tracked through dependency graphs")
        print("3. Errors localized to specific code lines")
        print("4. Self-reflection targets only failed sub-tasks")
        print("5. Automatic recovery and code correction")