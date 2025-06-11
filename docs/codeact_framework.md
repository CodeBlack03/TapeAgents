# CodeAct Framework for TapeAgents

The CodeAct framework transforms TapeAgents to use **executable Python code as the planning language** instead of text-based plans. This enables precise error localization, targeted self-reflection, and robust error recovery.

## 🎯 Key Innovations

### 1. **Code-Based Planning**
- Plans are generated as executable Python functions
- Each function has clear inputs, outputs, and dependencies
- Functions are organized in workflow dependency graphs

### 2. **Workflow Dependency Graphs**
- Execution flow tracked as a directed acyclic graph (DAG)
- Automatic dependency resolution and parallel execution
- Impact analysis when nodes fail

### 3. **Precise Error Localization**
- Errors mapped to specific code lines and functions
- Context-aware error reporting with surrounding code
- Detailed traceback analysis

### 4. **Targeted Self-Reflection**
- Reflection focuses only on failed sub-tasks
- Root cause analysis for specific function failures
- Automatic code correction and retry mechanisms

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CodeAct Framework                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ CodeAct     │ │ Workflow    │ │ CodeAct     │
│ Agent       │ │ Graph       │ │ Environment │
│             │ │ Executor    │ │             │
└─────────────┘ └─────────────┘ └─────────────┘
    │                 │                 │
    ▼                 ▼                 ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Planning    │ │ Dependency  │ │ Error       │
│ Nodes       │ │ Resolution  │ │ Analysis    │
└─────────────┘ └─────────────┘ └─────────────┘
```

## 📋 Core Components

### 1. **CodeAction**
Represents an executable code action with safety validation:

```python
from tapeagents import CodeAction

action = CodeAction(
    code="""
def analyze_data(data_path):
    import pandas as pd
    df = pd.read_csv(data_path)
    return df.describe()
""",
    function_name="analyze_data",
    expected_outputs=["summary_stats"],
    safety_level="safe",
    timeout=30.0
)
```

### 2. **WorkflowGraph**
Manages execution dependencies and flow:

```python
from tapeagents import WorkflowGraph, WorkflowNode, DependencyType

# Create workflow
workflow = WorkflowGraph()

# Add nodes
load_node = WorkflowNode(
    name="load_data",
    code="def load_data(): return pd.read_csv('data.csv')",
    function_name="load_data",
    outputs=["raw_data"]
)

process_node = WorkflowNode(
    name="process_data", 
    code="def process_data(raw_data): return raw_data.dropna()",
    function_name="process_data",
    inputs=["raw_data"],
    outputs=["clean_data"]
)

# Add to workflow
workflow.add_node(load_node)
workflow.add_node(process_node)

# Add dependency
workflow.add_dependency(load_node.id, process_node.id, DependencyType.DATA_DEPENDENCY)
```

### 3. **CodeActAgent**
Agent that plans using executable code:

```python
from tapeagents import create_codeact_agent, LiteLLM

llm = LiteLLM(model_name="gpt-4o")
agent = create_codeact_agent(llm, name="DataAnalyst")

# Agent automatically:
# 1. Generates code plans
# 2. Creates workflow graphs
# 3. Handles execution errors
# 4. Performs targeted reflection
```

### 4. **CodeActEnvironment**
Environment for executing workflows with error tracking:

```python
from tapeagents import CodeActEnvironment

env = CodeActEnvironment(
    execution_mode="parallel",  # or "sequential", "adaptive"
    max_parallel_nodes=4,
    enable_sandboxing=True,
    default_timeout=30.0
)

# Environment provides:
# - Safe code execution
# - Dependency resolution
# - Error localization
# - Performance monitoring
```

## 🔄 Planning Process

### 1. **Code Planning Node**
Generates executable code plans:

```python
class CodePlanningNode(Node):
    def make_prompt(self, agent, tape):
        return Prompt(messages=[{
            "role": "system", 
            "content": """
            Create an executable Python plan with these requirements:
            
            1. Break task into small, testable functions
            2. Define clear inputs/outputs for each function
            3. Specify dependencies between functions
            4. Include error handling and validation
            
            Format:
            ```python
            def step_1_function(inputs):
                \"\"\"
                Description: What this does
                Inputs: input parameters
                Outputs: output variables  
                Dependencies: []
                \"\"\"
                # Implementation
                return result
            ```
            """
        }])
```

### 2. **Workflow Execution**
Executes code with dependency tracking:

```python
# Sequential execution
results = []
for node_id in workflow.get_execution_path():
    node = workflow.nodes[node_id]
    if node.is_ready_to_execute(completed_nodes):
        result = executor.execute_node(node)
        results.append(result)

# Parallel execution  
ready_nodes = workflow.get_ready_nodes()
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(execute_node, node) for node in ready_nodes]
    results = [future.result() for future in futures]
```

### 3. **Error Analysis and Recovery**
Precise error localization and targeted reflection:

```python
class CodeReflectionNode(Node):
    def analyze_failure(self, failed_node):
        error = failed_node.error
        
        analysis = {
            "error_type": error.error_type,
            "failed_function": error.location.function_name,
            "failed_line": error.location.line_number,
            "context_lines": error.context_lines,
            "root_cause": self.identify_root_cause(error),
            "suggested_fixes": self.generate_fixes(error)
        }
        
        return analysis
```

## 🛠️ Usage Examples

### Basic Data Analysis Task

```python
from tapeagents import create_codeact_agent, CodeActEnvironment, LiteLLM
from tapeagents.core import Tape, TapeMetadata, Observation
from tapeagents.orchestrator import main_loop

# Setup
llm = LiteLLM(model_name="gpt-4o")
agent = create_codeact_agent(llm)
environment = CodeActEnvironment(execution_mode="sequential")

# Create task
task = Observation(content="""
Analyze sales data:
1. Load CSV data
2. Calculate total sales
3. Find top products
4. Create visualization
""")

start_tape = Tape(steps=[task], metadata=TapeMetadata(id="sales_analysis"))

# Execute
stream = main_loop(agent, start_tape, environment)
final_tape = stream.get_final_tape()

# Results include:
# - Generated code plan
# - Workflow execution results  
# - Error analysis (if any)
# - Performance metrics
```

### Error Recovery Example

```python
# Task with intentional error
task = Observation(content="""
Process data file:
1. Read from non-existent file (will fail)
2. Process the data
3. Save results

Handle file not found error gracefully.
""")

# Agent will:
# 1. Generate initial plan
# 2. Execute and encounter file error
# 3. Reflect on specific failure
# 4. Generate corrected code
# 5. Retry execution
# 6. Complete successfully
```

### Parallel Workflow Execution

```python
environment = CodeActEnvironment(
    execution_mode="parallel",
    max_parallel_nodes=4
)

# Task with independent subtasks
task = Observation(content="""
Multi-source data analysis:
1. Load data from API (independent)
2. Load data from database (independent)  
3. Load data from file (independent)
4. Merge all data sources (depends on 1,2,3)
5. Analyze merged data (depends on 4)
""")

# Nodes 1,2,3 execute in parallel
# Node 4 waits for 1,2,3 completion
# Node 5 waits for 4 completion
```

## 🔒 Safety Features

### Code Sandboxing
```python
environment = CodeActEnvironment(
    enable_sandboxing=True,
    allowed_imports=["pandas", "numpy", "matplotlib"],
    # Blocks: os.system, subprocess, eval, exec, file operations
)
```

### Safety Validation
```python
action = CodeAction(
    code="import os; os.system('rm -rf /')",  # Dangerous
    safety_level="strict"
)

issues = action.validate_code_safety()
# Returns: ["System command execution: os.system"]
```

### Execution Limits
```python
environment = CodeActEnvironment(
    default_timeout=30.0,  # 30 second timeout
    max_parallel_nodes=4,   # Limit parallelism
)
```

## 📊 Error Analysis Features

### Line-Level Error Localization
```python
error = CodeError(
    error_type="NameError",
    error_message="name 'undefined_var' is not defined",
    location=CodeLocation(
        function_name="process_data",
        line_number=15,
        file_name="<workflow_node>"
    ),
    context_lines=[
        "    13: def process_data(df):",
        "    14:     # Process the dataframe", 
        ">>> 15:     result = undefined_var + df.sum()",  # Error line
        "    16:     return result",
        "    17: "
    ]
)
```

### Failure Impact Analysis
```python
impact = workflow.analyze_failure_impact(failed_node_id)
# Returns:
{
    "failed_node": "process_data_node",
    "directly_affected": ["visualize_node", "report_node"],
    "total_affected": ["visualize_node", "report_node", "summary_node"],
    "can_continue": False,
    "alternative_paths": []
}
```

### Targeted Reflection
```python
reflection = CodeReflection(
    failed_node_id="process_data_node",
    root_cause="Undefined variable 'undefined_var' in line 15",
    suggested_fixes=[
        "Define 'undefined_var' before use",
        "Remove reference to 'undefined_var'", 
        "Use existing variable instead"
    ],
    alternative_approaches=[
        "Use different processing algorithm",
        "Add input validation"
    ]
)
```

## 🚀 Advanced Features

### Adaptive Execution Mode
```python
environment = CodeActEnvironment(execution_mode="adaptive")

# Automatically chooses:
# - Parallel for independent tasks
# - Sequential for dependent tasks
# - Based on workflow structure analysis
```

### Custom Dependency Types
```python
workflow.add_dependency(
    node_a.id, node_b.id, 
    DependencyType.RESOURCE_DEPENDENCY  # B needs resources from A
)

workflow.add_dependency(
    node_c.id, node_d.id,
    DependencyType.CONTROL_DEPENDENCY   # D only runs if C succeeds
)
```

### Performance Monitoring
```python
summary = environment.get_execution_summary()
# Returns:
{
    "total_executions": 15,
    "successful_executions": 12,
    "failed_executions": 3,
    "total_execution_time": 45.2,
    "average_execution_time": 3.01
}
```

## 🔧 Integration with Existing TapeAgents

The CodeAct framework seamlessly integrates with existing TapeAgents:

```python
# Use with existing agents
from tapeagents import Agent
from tapeagents.codeact_agent import CodePlanningNode, CodeExecutionNode

class HybridAgent(Agent):
    def __init__(self, **kwargs):
        nodes = [
            TextPlanningNode(),      # Traditional text planning
            CodePlanningNode(),      # CodeAct planning
            CodeExecutionNode(),     # Code execution
            ReflectionNode()         # Traditional reflection
        ]
        super().__init__(nodes=nodes, **kwargs)
```

## 📈 Benefits Over Traditional Planning

| Aspect | Traditional Planning | CodeAct Framework |
|--------|---------------------|-------------------|
| **Plan Format** | Natural language text | Executable Python code |
| **Error Localization** | General failure description | Line-level precision |
| **Self-Reflection** | Entire plan re-evaluation | Targeted sub-task analysis |
| **Execution Tracking** | Linear step sequence | Dependency graph |
| **Parallelization** | Manual coordination | Automatic dependency resolution |
| **Error Recovery** | Full plan regeneration | Surgical code fixes |
| **Validation** | Manual review | Automated safety checks |
| **Debugging** | Trial and error | Systematic error analysis |

## 🎯 Best Practices

### 1. **Function Design**
- Keep functions small and focused
- Use clear, descriptive names
- Include comprehensive docstrings
- Add input validation

### 2. **Dependency Management**
- Minimize dependencies between functions
- Use data dependencies over control dependencies
- Design for parallel execution when possible

### 3. **Error Handling**
- Include try-catch blocks in generated code
- Validate inputs before processing
- Provide meaningful error messages

### 4. **Safety Considerations**
- Enable sandboxing for untrusted code
- Limit allowed imports and operations
- Set appropriate execution timeouts

### 5. **Performance Optimization**
- Use parallel execution for independent tasks
- Monitor execution times and optimize bottlenecks
- Cache expensive computations

The CodeAct framework represents a significant advancement in LLM agent planning, providing the precision and reliability needed for complex, real-world tasks while maintaining the flexibility and adaptability of traditional approaches.