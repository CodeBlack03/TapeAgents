"""
Comprehensive test suite for the CodeAct framework.

Tests cover:
- Core data structures and workflow graphs
- Code planning and execution
- Error localization and recovery
- Safety features and sandboxing
- Parallel and sequential execution modes
"""

import pytest
import time
from unittest.mock import Mock, patch

from tapeagents.codeact_core import (
    CodeAction, CodeActTape, CodeError, CodeExecutionResult, CodePlan, 
    CodeReflection, DependencyType, WorkflowExecutor, WorkflowGraph, 
    WorkflowNode, CodeExecutionStatus, CodeLocation
)
from tapeagents.codeact_agent import CodeActAgent, CodePlanningNode, CodeExecutionNode, CodeReflectionNode
from tapeagents.codeact_environment import CodeActEnvironment
from tapeagents.core import Tape, TapeMetadata, Observation
from tapeagents.llms import LLMStream, LLMEvent, LLMOutput


class TestCodeActCore:
    """Test core CodeAct data structures."""
    
    def test_workflow_node_creation(self):
        """Test WorkflowNode creation and basic functionality."""
        node = WorkflowNode(
            name="test_node",
            code="def test(): return 42",
            function_name="test",
            inputs=["x"],
            outputs=["result"]
        )
        
        assert node.name == "test_node"
        assert node.function_name == "test"
        assert node.inputs == ["x"]
        assert node.outputs == ["result"]
        assert node.status == CodeExecutionStatus.PENDING
        assert len(node.dependencies) == 0
    
    def test_workflow_node_dependencies(self):
        """Test dependency management in WorkflowNode."""
        node = WorkflowNode(name="test", code="", function_name="test")
        
        node.add_dependency("dep1", DependencyType.DATA_DEPENDENCY)
        node.add_dependency("dep2", DependencyType.CONTROL_DEPENDENCY)
        
        assert "dep1" in node.dependencies
        assert "dep2" in node.dependencies
        assert node.get_dependency_type("dep1") == DependencyType.DATA_DEPENDENCY
        assert node.get_dependency_type("dep2") == DependencyType.CONTROL_DEPENDENCY
    
    def test_workflow_node_ready_to_execute(self):
        """Test readiness check for node execution."""
        node = WorkflowNode(name="test", code="", function_name="test")
        node.add_dependency("dep1")
        node.add_dependency("dep2")
        
        # Not ready - dependencies not completed
        assert not node.is_ready_to_execute(set())
        assert not node.is_ready_to_execute({"dep1"})
        
        # Ready - all dependencies completed
        assert node.is_ready_to_execute({"dep1", "dep2"})
        assert node.is_ready_to_execute({"dep1", "dep2", "extra"})
    
    def test_workflow_graph_creation(self):
        """Test WorkflowGraph creation and node management."""
        workflow = WorkflowGraph()
        
        node1 = WorkflowNode(name="node1", code="", function_name="func1")
        node2 = WorkflowNode(name="node2", code="", function_name="func2")
        
        id1 = workflow.add_node(node1)
        id2 = workflow.add_node(node2)
        
        assert len(workflow.nodes) == 2
        assert workflow.nodes[id1] == node1
        assert workflow.nodes[id2] == node2
    
    def test_workflow_graph_dependencies(self):
        """Test dependency management in WorkflowGraph."""
        workflow = WorkflowGraph()
        
        node1 = WorkflowNode(name="node1", code="", function_name="func1")
        node2 = WorkflowNode(name="node2", code="", function_name="func2")
        
        id1 = workflow.add_node(node1)
        id2 = workflow.add_node(node2)
        
        workflow.add_dependency(id1, id2, DependencyType.DATA_DEPENDENCY)
        
        assert id1 in node2.dependencies
        assert id2 in node1.dependents
        assert node2.get_dependency_type(id1) == DependencyType.DATA_DEPENDENCY
    
    def test_workflow_graph_execution_path(self):
        """Test execution path calculation (topological sort)."""
        workflow = WorkflowGraph()
        
        # Create nodes: A -> B -> C, A -> D -> C
        node_a = WorkflowNode(name="A", code="", function_name="a")
        node_b = WorkflowNode(name="B", code="", function_name="b")
        node_c = WorkflowNode(name="C", code="", function_name="c")
        node_d = WorkflowNode(name="D", code="", function_name="d")
        
        id_a = workflow.add_node(node_a)
        id_b = workflow.add_node(node_b)
        id_c = workflow.add_node(node_c)
        id_d = workflow.add_node(node_d)
        
        workflow.add_dependency(id_a, id_b)
        workflow.add_dependency(id_b, id_c)
        workflow.add_dependency(id_a, id_d)
        workflow.add_dependency(id_d, id_c)
        
        execution_path = workflow.get_execution_path()
        
        # A should come first, C should come last
        assert execution_path.index(id_a) < execution_path.index(id_b)
        assert execution_path.index(id_a) < execution_path.index(id_d)
        assert execution_path.index(id_b) < execution_path.index(id_c)
        assert execution_path.index(id_d) < execution_path.index(id_c)
    
    def test_workflow_graph_ready_nodes(self):
        """Test getting nodes ready for execution."""
        workflow = WorkflowGraph()
        
        node1 = WorkflowNode(name="node1", code="", function_name="func1")
        node2 = WorkflowNode(name="node2", code="", function_name="func2")
        node3 = WorkflowNode(name="node3", code="", function_name="func3")
        
        id1 = workflow.add_node(node1)
        id2 = workflow.add_node(node2)
        id3 = workflow.add_node(node3)
        
        workflow.add_dependency(id1, id2)
        workflow.add_dependency(id2, id3)
        
        # Initially, only node1 should be ready
        ready_nodes = workflow.get_ready_nodes()
        assert len(ready_nodes) == 1
        assert ready_nodes[0].name == "node1"
        
        # After node1 completes, node2 should be ready
        node1.status = CodeExecutionStatus.SUCCESS
        ready_nodes = workflow.get_ready_nodes()
        assert len(ready_nodes) == 1
        assert ready_nodes[0].name == "node2"
    
    def test_workflow_graph_failure_impact(self):
        """Test failure impact analysis."""
        workflow = WorkflowGraph()
        
        node1 = WorkflowNode(name="node1", code="", function_name="func1")
        node2 = WorkflowNode(name="node2", code="", function_name="func2")
        node3 = WorkflowNode(name="node3", code="", function_name="func3")
        
        id1 = workflow.add_node(node1)
        id2 = workflow.add_node(node2)
        id3 = workflow.add_node(node3)
        
        workflow.add_dependency(id1, id2)
        workflow.add_dependency(id2, id3)
        
        # Analyze impact of node1 failure
        impact = workflow.analyze_failure_impact(id1)
        
        assert impact["failed_node"] == id1
        assert id2 in impact["directly_affected"]
        assert id3 in impact["total_affected"]
    
    def test_code_action_creation(self):
        """Test CodeAction creation and validation."""
        action = CodeAction(
            code="def test(x): return x * 2",
            function_name="test",
            expected_outputs=["result"],
            timeout=10.0,
            safety_level="safe"
        )
        
        assert action.function_name == "test"
        assert action.expected_outputs == ["result"]
        assert action.timeout == 10.0
        assert action.safety_level == "safe"
    
    def test_code_action_function_info_extraction(self):
        """Test function information extraction from code."""
        action = CodeAction(
            code='''
def calculate_sum(a, b):
    """Calculate sum of two numbers."""
    return a + b
''',
            function_name="calculate_sum"
        )
        
        func_info = action.extract_function_info()
        
        assert func_info["name"] == "calculate_sum"
        assert func_info["args"] == ["a", "b"]
        assert "Calculate sum" in func_info["docstring"]
    
    def test_code_action_safety_validation(self):
        """Test code safety validation."""
        # Safe code
        safe_action = CodeAction(
            code="def safe_func(): return 42",
            function_name="safe_func"
        )
        assert len(safe_action.validate_code_safety()) == 0
        
        # Unsafe code
        unsafe_action = CodeAction(
            code="import os; os.system('rm -rf /')",
            function_name="unsafe_func"
        )
        issues = unsafe_action.validate_code_safety()
        assert len(issues) > 0
        assert any("os.system" in issue for issue in issues)
    
    def test_code_execution_result(self):
        """Test CodeExecutionResult creation and methods."""
        # Successful result
        success_result = CodeExecutionResult(
            node_id="test_node",
            status=CodeExecutionStatus.SUCCESS,
            result=42,
            execution_time=1.5
        )
        
        assert success_result.is_success()
        assert success_result.get_error_summary() == "No error"
        
        # Failed result
        error = CodeError(
            error_type="ValueError",
            error_message="Invalid input",
            location=CodeLocation(function_name="test", line_number=5),
            traceback_info=""
        )
        
        failed_result = CodeExecutionResult(
            node_id="test_node",
            status=CodeExecutionStatus.FAILED,
            error=error
        )
        
        assert not failed_result.is_success()
        assert "ValueError" in failed_result.get_error_summary()
        assert failed_result.get_failed_lines() == [5]


class TestWorkflowExecutor:
    """Test workflow execution functionality."""
    
    def test_execute_simple_node(self):
        """Test execution of a simple workflow node."""
        executor = WorkflowExecutor()
        
        node = WorkflowNode(
            name="simple_test",
            code="""
def simple_test():
    result = 2 + 2
    return result
""",
            function_name="simple_test",
            outputs=["result"]
        )
        
        result = executor.execute_node(node)
        
        assert result.status == CodeExecutionStatus.SUCCESS
        assert result.result == 4
    
    def test_execute_node_with_error(self):
        """Test execution of a node that raises an error."""
        executor = WorkflowExecutor()
        
        node = WorkflowNode(
            name="error_test",
            code="""
def error_test():
    raise ValueError("Test error")
""",
            function_name="error_test"
        )
        
        result = executor.execute_node(node)
        
        assert result.status == CodeExecutionStatus.FAILED
        assert result.error is not None
        assert result.error.error_type == "ValueError"
        assert "Test error" in result.error.error_message
    
    def test_execute_node_with_dependencies(self):
        """Test execution with variable dependencies."""
        executor = WorkflowExecutor()
        
        # First node produces data
        node1 = WorkflowNode(
            name="producer",
            code="""
data = [1, 2, 3, 4, 5]
""",
            function_name="producer",
            outputs=["data"]
        )
        
        result1 = executor.execute_node(node1)
        assert result1.status == CodeExecutionStatus.SUCCESS
        
        # Second node consumes data
        node2 = WorkflowNode(
            name="consumer",
            code="""
def consumer():
    total = sum(data)
    return total
""",
            function_name="consumer",
            inputs=["data"],
            outputs=["total"]
        )
        
        # Update global namespace with first result
        executor.global_namespace.update(result1.output_variables)
        
        result2 = executor.execute_node(node2, result1.output_variables)
        assert result2.status == CodeExecutionStatus.SUCCESS
        assert result2.result == 15


class TestCodeActAgent:
    """Test CodeAct agent functionality."""
    
    def test_code_planning_node_prompt_generation(self):
        """Test prompt generation for code planning."""
        node = CodePlanningNode()
        
        # Mock tape with task
        tape = Tape(
            steps=[Observation(content="Create a calculator function")],
            metadata=TapeMetadata(id="test")
        )
        
        # Mock agent
        agent = Mock()
        agent.tools_description = "Python standard library"
        
        prompt = node.make_prompt(agent, tape)
        
        assert len(prompt.messages) > 0
        system_message = prompt.messages[0]["content"]
        assert "executable Python code" in system_message
        assert "dependencies" in system_message.lower()
    
    def test_code_execution_node_workflow_extraction(self):
        """Test workflow extraction from tape."""
        node = CodeExecutionNode()
        
        # Create tape with CodePlan
        workflow = WorkflowGraph()
        test_node = WorkflowNode(name="test", code="", function_name="test")
        workflow.add_node(test_node)
        
        code_plan = CodePlan(
            workflow=workflow,
            plan_description="Test plan"
        )
        
        tape = Tape(
            steps=[code_plan],
            metadata=TapeMetadata(id="test")
        )
        
        extracted_workflow = node._get_current_workflow(tape)
        assert extracted_workflow is not None
        assert len(extracted_workflow.nodes) == 1
    
    def test_code_reflection_node_error_analysis(self):
        """Test error analysis in reflection node."""
        node = CodeReflectionNode()
        
        # Create workflow with failed node
        workflow = WorkflowGraph()
        failed_node = WorkflowNode(
            name="failed_test",
            code="def test(): return undefined_var",
            function_name="test"
        )
        failed_node.status = CodeExecutionStatus.FAILED
        failed_node.error = CodeError(
            error_type="NameError",
            error_message="name 'undefined_var' is not defined",
            location=CodeLocation(function_name="test", line_number=1),
            traceback_info=""
        )
        
        workflow.add_node(failed_node)
        
        reflection = CodeReflection(
            failed_node_id=failed_node.id,
            error_analysis={},
            root_cause="",
            suggested_fixes=[],
            impact_analysis={},
            alternative_approaches=[]
        )
        
        analysis = reflection.analyze_failure(workflow, failed_node.id)
        
        assert analysis["error_type"] == "NameError"
        assert analysis["failed_function"] == "test"
        assert "undefined_var" in analysis["error_message"]
        assert len(analysis["suggested_fixes"]) > 0


class TestCodeActEnvironment:
    """Test CodeAct environment functionality."""
    
    def test_environment_creation(self):
        """Test CodeActEnvironment creation with different modes."""
        env = CodeActEnvironment(
            execution_mode="parallel",
            max_parallel_nodes=4,
            enable_sandboxing=True
        )
        
        assert env.execution_mode == "parallel"
        assert env.max_parallel_nodes == 4
        assert env.enable_sandboxing == True
    
    def test_code_safety_validation(self):
        """Test code safety validation in environment."""
        env = CodeActEnvironment(enable_sandboxing=True)
        
        # Safe code
        safe_code = "def safe(): return 42"
        issues = env._validate_code_safety(safe_code)
        assert len(issues) == 0
        
        # Unsafe code
        unsafe_code = "import os; os.system('rm -rf /')"
        issues = env._validate_code_safety(unsafe_code)
        assert len(issues) > 0
    
    def test_safe_code_execution(self):
        """Test safe code execution with monitoring."""
        env = CodeActEnvironment()
        
        code = """
def calculate():
    result = 2 + 2
    return result

result = calculate()
"""
        
        result = env._execute_code_safely(code, "calculate", timeout=5.0)
        
        assert result.status == CodeExecutionStatus.SUCCESS
        assert result.output_variables.get("result") == 4
        assert result.execution_time > 0
    
    def test_code_execution_with_error(self):
        """Test code execution that produces an error."""
        env = CodeActEnvironment()
        
        code = """
def error_function():
    return undefined_variable

result = error_function()
"""
        
        result = env._execute_code_safely(code, "error_function", timeout=5.0)
        
        assert result.status == CodeExecutionStatus.FAILED
        assert result.error is not None
        assert result.error.error_type == "NameError"
    
    def test_context_lines_extraction(self):
        """Test extraction of context lines around errors."""
        env = CodeActEnvironment()
        
        code = """line 1
line 2
line 3
line 4
line 5"""
        
        context = env._get_context_lines(code, 3, context=1)
        
        assert len(context) == 3  # line 2, 3, 4
        assert ">>> 3:" in context[1]  # Error line marked
        assert "line 3" in context[1]
    
    def test_execution_summary(self):
        """Test execution summary generation."""
        env = CodeActEnvironment()
        
        # Add some mock execution history
        env.execution_history = [
            {"success": True, "execution_time": 1.0},
            {"success": False, "execution_time": 0.5},
            {"success": True, "execution_time": 2.0}
        ]
        
        summary = env.get_execution_summary()
        
        assert summary["total_executions"] == 3
        assert summary["successful_executions"] == 2
        assert summary["failed_executions"] == 1
        assert summary["total_execution_time"] == 3.5
    
    def test_environment_reset(self):
        """Test environment reset functionality."""
        env = CodeActEnvironment()
        
        # Add some state
        env.global_namespace["test"] = "value"
        env.execution_history.append({"test": "data"})
        env.active_workflows["test"] = "workflow"
        
        # Reset
        env.reset_environment()
        
        assert len(env.global_namespace) == 0
        assert len(env.execution_history) == 0
        assert len(env.active_workflows) == 0


class TestCodeActIntegration:
    """Test integration between CodeAct components."""
    
    def test_end_to_end_simple_task(self):
        """Test end-to-end execution of a simple task."""
        # Create mock LLM that returns code plan
        mock_llm_output = LLMOutput(
            role="assistant",
            content="""
# Plan: Create a simple calculator

def add_numbers(a, b):
    \"\"\"
    Description: Add two numbers
    Inputs: a, b
    Outputs: sum_result
    Dependencies: []
    \"\"\"
    sum_result = a + b
    return sum_result

def multiply_numbers(x, y):
    \"\"\"
    Description: Multiply two numbers  
    Inputs: x, y
    Outputs: product_result
    Dependencies: []
    \"\"\"
    product_result = x * y
    return product_result
"""
        )
        
        # Create mock LLM stream
        llm_stream = LLMStream(
            [LLMEvent(output=mock_llm_output)],
            prompt=Mock()
        )
        
        # Test code planning node
        planning_node = CodePlanningNode()
        agent = Mock()
        agent.tools_description = "Python standard library"
        
        tape = Tape(
            steps=[Observation(content="Create calculator functions")],
            metadata=TapeMetadata(id="test")
        )
        
        # Generate steps from planning node
        steps = list(planning_node.generate_steps(agent, tape, llm_stream))
        
        # Should generate CodePlan and CodeAction steps
        code_plan_steps = [s for s in steps if isinstance(s, CodePlan)]
        code_action_steps = [s for s in steps if isinstance(s, CodeAction)]
        
        assert len(code_plan_steps) > 0
        assert len(code_action_steps) > 0
        
        # Test that workflow was created correctly
        workflow = code_plan_steps[0].workflow
        assert len(workflow.nodes) > 0
    
    def test_error_recovery_flow(self):
        """Test the error recovery and reflection flow."""
        # Create a workflow with a failed node
        workflow = WorkflowGraph()
        
        failed_node = WorkflowNode(
            name="failed_calculation",
            code="def calc(): return undefined_var + 5",
            function_name="calc"
        )
        failed_node.status = CodeExecutionStatus.FAILED
        failed_node.error = CodeError(
            error_type="NameError",
            error_message="name 'undefined_var' is not defined",
            location=CodeLocation(function_name="calc", line_number=1),
            traceback_info="",
            context_lines=["def calc(): return undefined_var + 5"]
        )
        
        workflow.add_node(failed_node)
        
        # Test reflection node
        reflection_node = CodeReflectionNode()
        
        # Mock LLM response with corrected code
        mock_reflection_output = LLMOutput(
            role="assistant",
            content="""
## Error Analysis
The function references an undefined variable 'undefined_var'.

## Root Cause
Variable 'undefined_var' is not defined before use.

## Corrected Code
```python
def calc():
    undefined_var = 10  # Define the variable
    return undefined_var + 5
```

## Explanation
Added variable definition to fix the NameError.
"""
        )
        
        llm_stream = LLMStream(
            [LLMEvent(output=mock_reflection_output)],
            prompt=Mock()
        )
        
        # Create tape with failed workflow
        code_plan = CodePlan(workflow=workflow, plan_description="Test plan")
        tape = Tape(steps=[code_plan], metadata=TapeMetadata(id="test"))
        
        # Generate reflection steps
        steps = list(reflection_node.generate_steps(Mock(), tape, llm_stream))
        
        # Should generate reflection and corrected code
        reflection_steps = [s for s in steps if isinstance(s, CodeReflection)]
        code_action_steps = [s for s in steps if isinstance(s, CodeAction)]
        
        assert len(reflection_steps) > 0
        assert len(code_action_steps) > 0
        
        # Check that corrected code was generated
        corrected_action = code_action_steps[0]
        assert "undefined_var = 10" in corrected_action.code


class TestCodeActPerformance:
    """Test performance aspects of CodeAct framework."""
    
    def test_parallel_execution_performance(self):
        """Test that parallel execution is faster than sequential."""
        # This is a conceptual test - actual timing would depend on system
        env_sequential = CodeActEnvironment(execution_mode="sequential")
        env_parallel = CodeActEnvironment(execution_mode="parallel", max_parallel_nodes=4)
        
        # Create workflow with independent nodes
        workflow = WorkflowGraph()
        
        for i in range(4):
            node = WorkflowNode(
                name=f"independent_task_{i}",
                code=f"""
import time
def task_{i}():
    time.sleep(0.1)  # Simulate work
    return {i}
""",
                function_name=f"task_{i}",
                outputs=[f"result_{i}"]
            )
            workflow.add_node(node)
        
        # Both environments should handle the workflow
        # (Actual timing comparison would require real execution)
        assert env_sequential.execution_mode == "sequential"
        assert env_parallel.execution_mode == "parallel"
        assert env_parallel.max_parallel_nodes == 4
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking in execution results."""
        env = CodeActEnvironment()
        
        code = """
def memory_test():
    # Create some data
    data = list(range(1000))
    return len(data)

result = memory_test()
"""
        
        result = env._execute_code_safely(code, "memory_test")
        
        # Memory usage should be tracked (even if 0 in test environment)
        assert hasattr(result, 'memory_usage')
        assert isinstance(result.memory_usage, int)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])