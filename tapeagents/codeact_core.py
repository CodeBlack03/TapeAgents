"""
CodeAct Core Data Structures for TapeAgents

This module implements the core data structures for CodeAct-style planning where:
- Plans are represented as executable Python code
- Execution flow is tracked as a workflow dependency graph
- Errors are localized to specific code lines and functions
- Self-reflection targets only failed sub-tasks
"""

from __future__ import annotations

import ast
import inspect
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4
from enum import Enum
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from typing_extensions import Self

from .core import Step, Action, Observation, Thought


class CodeExecutionStatus(str, Enum):
    """Status of code execution."""
    PENDING = "pending"
    RUNNING = "running" 
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class DependencyType(str, Enum):
    """Types of dependencies between code blocks."""
    DATA_DEPENDENCY = "data"      # Output of A is input to B
    CONTROL_DEPENDENCY = "control"  # B executes only if A succeeds
    RESOURCE_DEPENDENCY = "resource"  # B needs resources from A
    TEMPORAL_DEPENDENCY = "temporal"  # B must execute after A


@dataclass
class CodeLocation:
    """Represents a location in code."""
    function_name: str
    line_number: int
    column_number: int = 0
    file_name: str = "<generated>"
    
    def __str__(self) -> str:
        return f"{self.file_name}:{self.function_name}:{self.line_number}:{self.column_number}"


@dataclass
class CodeError:
    """Represents an error in code execution."""
    error_type: str
    error_message: str
    location: CodeLocation
    traceback_info: str
    context_lines: List[str] = field(default_factory=list)
    suggested_fixes: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"{self.error_type} at {self.location}: {self.error_message}"


class WorkflowNode(BaseModel):
    """A node in the workflow dependency graph."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description="Human-readable name for this node")
    code: str = Field(description="Python code for this node")
    function_name: str = Field(description="Name of the main function in this node")
    inputs: List[str] = Field(default_factory=list, description="Input variable names")
    outputs: List[str] = Field(default_factory=list, description="Output variable names")
    dependencies: Set[str] = Field(default_factory=set, description="IDs of nodes this depends on")
    dependents: Set[str] = Field(default_factory=set, description="IDs of nodes that depend on this")
    status: CodeExecutionStatus = CodeExecutionStatus.PENDING
    execution_time: float = 0.0
    memory_usage: int = 0
    error: Optional[CodeError] = None
    result: Any = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_dependency(self, node_id: str, dependency_type: DependencyType = DependencyType.DATA_DEPENDENCY):
        """Add a dependency to another node."""
        self.dependencies.add(node_id)
        self.metadata.setdefault("dependency_types", {})[node_id] = dependency_type
    
    def is_ready_to_execute(self, completed_nodes: Set[str]) -> bool:
        """Check if this node is ready to execute based on dependencies."""
        return self.dependencies.issubset(completed_nodes)
    
    def get_dependency_type(self, node_id: str) -> DependencyType:
        """Get the type of dependency on another node."""
        return self.metadata.get("dependency_types", {}).get(node_id, DependencyType.DATA_DEPENDENCY)


class WorkflowGraph(BaseModel):
    """Workflow dependency graph for code execution."""
    
    nodes: Dict[str, WorkflowNode] = Field(default_factory=dict)
    execution_order: List[str] = Field(default_factory=list)
    global_variables: Dict[str, Any] = Field(default_factory=dict)
    imports: List[str] = Field(default_factory=list)
    
    def add_node(self, node: WorkflowNode) -> str:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        return node.id
    
    def add_dependency(self, from_node_id: str, to_node_id: str, 
                      dependency_type: DependencyType = DependencyType.DATA_DEPENDENCY):
        """Add a dependency between two nodes."""
        if from_node_id in self.nodes and to_node_id in self.nodes:
            self.nodes[to_node_id].add_dependency(from_node_id, dependency_type)
            self.nodes[from_node_id].dependents.add(to_node_id)
    
    def get_ready_nodes(self) -> List[WorkflowNode]:
        """Get nodes that are ready to execute."""
        completed = {nid for nid, node in self.nodes.items() 
                    if node.status == CodeExecutionStatus.SUCCESS}
        
        ready_nodes = []
        for node in self.nodes.values():
            if (node.status == CodeExecutionStatus.PENDING and 
                node.is_ready_to_execute(completed)):
                ready_nodes.append(node)
        
        return ready_nodes
    
    def get_failed_nodes(self) -> List[WorkflowNode]:
        """Get nodes that have failed execution."""
        return [node for node in self.nodes.values() 
                if node.status == CodeExecutionStatus.FAILED]
    
    def get_execution_path(self) -> List[str]:
        """Get the planned execution path using topological sort."""
        if self.execution_order:
            return self.execution_order
        
        # Topological sort to determine execution order
        in_degree = {nid: len(node.dependencies) for nid, node in self.nodes.items()}
        queue = [nid for nid, degree in in_degree.items() if degree == 0]
        execution_path = []
        
        while queue:
            current = queue.pop(0)
            execution_path.append(current)
            
            for dependent_id in self.nodes[current].dependents:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)
        
        self.execution_order = execution_path
        return execution_path
    
    def get_critical_path(self) -> List[str]:
        """Get the critical path (longest path) through the graph."""
        # Simple implementation - can be optimized
        execution_path = self.get_execution_path()
        return execution_path  # For now, return full path
    
    def analyze_failure_impact(self, failed_node_id: str) -> Dict[str, Any]:
        """Analyze the impact of a node failure on the rest of the workflow."""
        failed_node = self.nodes[failed_node_id]
        
        # Find all nodes that depend on the failed node (directly or indirectly)
        affected_nodes = set()
        queue = list(failed_node.dependents)
        
        while queue:
            current = queue.pop(0)
            if current not in affected_nodes:
                affected_nodes.add(current)
                queue.extend(self.nodes[current].dependents)
        
        return {
            "failed_node": failed_node_id,
            "directly_affected": list(failed_node.dependents),
            "total_affected": list(affected_nodes),
            "can_continue": len(affected_nodes) < len(self.nodes) - 1,
            "alternative_paths": self._find_alternative_paths(failed_node_id)
        }
    
    def _find_alternative_paths(self, failed_node_id: str) -> List[List[str]]:
        """Find alternative execution paths that bypass the failed node."""
        # Simplified implementation
        alternative_paths = []
        execution_path = self.get_execution_path()
        
        if failed_node_id in execution_path:
            # Remove failed node and see if we can still reach the end
            remaining_nodes = [nid for nid in execution_path if nid != failed_node_id]
            if remaining_nodes:
                alternative_paths.append(remaining_nodes)
        
        return alternative_paths


class CodeAction(Action):
    """Action that contains executable Python code."""
    
    code: str = Field(description="Python code to execute")
    function_name: str = Field(description="Name of the main function")
    expected_outputs: List[str] = Field(default_factory=list, description="Expected output variables")
    timeout: float = Field(default=30.0, description="Execution timeout in seconds")
    requires_approval: bool = Field(default=False, description="Whether code needs approval before execution")
    safety_level: str = Field(default="safe", description="Safety level: safe, moderate, dangerous")
    
    def extract_function_info(self) -> Dict[str, Any]:
        """Extract information about the function from the code."""
        try:
            tree = ast.parse(self.code)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            if functions:
                main_func = functions[0]  # Assume first function is main
                return {
                    "name": main_func.name,
                    "args": [arg.arg for arg in main_func.args.args],
                    "line_number": main_func.lineno,
                    "docstring": ast.get_docstring(main_func)
                }
        except SyntaxError as e:
            return {"error": f"Syntax error: {e}"}
        
        return {}
    
    def validate_code_safety(self) -> List[str]:
        """Validate code for safety issues."""
        safety_issues = []
        
        # Check for dangerous operations
        dangerous_patterns = [
            "os.system", "subprocess", "eval", "exec", "open", "__import__",
            "file", "input", "raw_input", "compile"
        ]
        
        for pattern in dangerous_patterns:
            if pattern in self.code:
                safety_issues.append(f"Potentially dangerous operation: {pattern}")
        
        return safety_issues


class CodeExecutionResult(Observation):
    """Result of code execution with detailed error information."""
    
    node_id: str = Field(description="ID of the workflow node that was executed")
    status: CodeExecutionStatus = Field(description="Execution status")
    result: Any = Field(default=None, description="Execution result")
    output_variables: Dict[str, Any] = Field(default_factory=dict, description="Variables produced")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    memory_usage: int = Field(default=0, description="Memory usage in bytes")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    error: Optional[CodeError] = Field(default=None, description="Error information if failed")
    line_coverage: Dict[int, bool] = Field(default_factory=dict, description="Line coverage information")
    
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == CodeExecutionStatus.SUCCESS
    
    def get_error_summary(self) -> str:
        """Get a summary of the error if execution failed."""
        if self.error:
            return f"{self.error.error_type}: {self.error.error_message} at {self.error.location}"
        return "No error"
    
    def get_failed_lines(self) -> List[int]:
        """Get line numbers that failed during execution."""
        if self.error:
            return [self.error.location.line_number]
        return []


class CodePlan(Thought):
    """A plan represented as a workflow graph of executable code."""
    
    workflow: WorkflowGraph = Field(description="Workflow dependency graph")
    plan_description: str = Field(description="High-level description of the plan")
    estimated_duration: float = Field(default=0.0, description="Estimated execution time")
    complexity_score: float = Field(default=0.0, description="Plan complexity score")
    
    def add_code_step(self, name: str, code: str, function_name: str, 
                     inputs: List[str] = None, outputs: List[str] = None) -> str:
        """Add a code step to the plan."""
        node = WorkflowNode(
            name=name,
            code=code,
            function_name=function_name,
            inputs=inputs or [],
            outputs=outputs or []
        )
        return self.workflow.add_node(node)
    
    def add_dependency(self, from_step: str, to_step: str, 
                      dependency_type: DependencyType = DependencyType.DATA_DEPENDENCY):
        """Add a dependency between steps."""
        self.workflow.add_dependency(from_step, to_step, dependency_type)
    
    def get_next_executable_steps(self) -> List[WorkflowNode]:
        """Get steps that are ready to execute."""
        return self.workflow.get_ready_nodes()
    
    def estimate_complexity(self) -> float:
        """Estimate the complexity of the plan."""
        num_nodes = len(self.workflow.nodes)
        num_dependencies = sum(len(node.dependencies) for node in self.workflow.nodes.values())
        avg_code_length = sum(len(node.code) for node in self.workflow.nodes.values()) / max(num_nodes, 1)
        
        # Simple complexity formula
        complexity = (num_nodes * 0.3 + num_dependencies * 0.5 + avg_code_length / 100 * 0.2)
        self.complexity_score = complexity
        return complexity


class CodeReflection(Thought):
    """Reflection on code execution failures with targeted analysis."""
    
    failed_node_id: str = Field(description="ID of the failed node")
    error_analysis: Dict[str, Any] = Field(description="Detailed error analysis")
    root_cause: str = Field(description="Identified root cause of failure")
    suggested_fixes: List[str] = Field(description="Suggested fixes for the failure")
    impact_analysis: Dict[str, Any] = Field(description="Analysis of failure impact")
    alternative_approaches: List[str] = Field(description="Alternative approaches to try")
    
    def analyze_failure(self, workflow: WorkflowGraph, failed_node_id: str) -> Dict[str, Any]:
        """Analyze the failure of a specific node."""
        failed_node = workflow.nodes[failed_node_id]
        
        if not failed_node.error:
            return {"error": "No error information available"}
        
        error = failed_node.error
        
        # Analyze error type and context
        analysis = {
            "error_type": error.error_type,
            "error_location": str(error.location),
            "error_message": error.error_message,
            "failed_function": error.location.function_name,
            "failed_line": error.location.line_number,
            "context_lines": error.context_lines,
            "dependencies_met": failed_node.is_ready_to_execute(
                {nid for nid, node in workflow.nodes.items() 
                 if node.status == CodeExecutionStatus.SUCCESS}
            )
        }
        
        # Suggest fixes based on error type
        if "NameError" in error.error_type:
            analysis["suggested_fixes"] = [
                "Check if all required variables are defined",
                "Verify import statements",
                "Check variable naming and scope"
            ]
        elif "TypeError" in error.error_type:
            analysis["suggested_fixes"] = [
                "Check function argument types",
                "Verify data type compatibility",
                "Add type checking and conversion"
            ]
        elif "AttributeError" in error.error_type:
            analysis["suggested_fixes"] = [
                "Check if object has the required attribute",
                "Verify object initialization",
                "Add attribute existence checks"
            ]
        else:
            analysis["suggested_fixes"] = [
                "Review error message and traceback",
                "Check input data validity",
                "Add error handling and validation"
            ]
        
        self.error_analysis = analysis
        return analysis
    
    def generate_fix_code(self, original_code: str, error: CodeError) -> str:
        """Generate fixed code based on error analysis."""
        lines = original_code.split('\n')
        error_line_idx = error.location.line_number - 1
        
        if error_line_idx < len(lines):
            error_line = lines[error_line_idx]
            
            # Simple fix generation based on error type
            if "NameError" in error.error_type:
                # Add variable initialization
                fixed_line = f"    # Fixed: Added variable initialization\n    {error_line}"
            elif "TypeError" in error.error_type:
                # Add type checking
                fixed_line = f"    # Fixed: Added type checking\n    if isinstance(variable, expected_type):\n        {error_line}"
            else:
                # Add try-catch
                fixed_line = f"    try:\n        {error_line}\n    except Exception as e:\n        print(f'Error: {{e}}')"
            
            lines[error_line_idx] = fixed_line
        
        return '\n'.join(lines)


class WorkflowExecutor:
    """Executes workflow graphs with error tracking and recovery."""
    
    def __init__(self):
        self.global_namespace = {}
        self.execution_history = []
        
    def execute_workflow(self, workflow: WorkflowGraph) -> List[CodeExecutionResult]:
        """Execute a complete workflow graph."""
        results = []
        execution_path = workflow.get_execution_path()
        
        for node_id in execution_path:
            node = workflow.nodes[node_id]
            
            # Check if node is ready to execute
            if not node.is_ready_to_execute({nid for nid, n in workflow.nodes.items() 
                                           if n.status == CodeExecutionStatus.SUCCESS}):
                continue
            
            # Execute the node
            result = self.execute_node(node, workflow.global_variables)
            results.append(result)
            
            # Update node status
            node.status = result.status
            node.error = result.error
            node.result = result.result
            node.execution_time = result.execution_time
            
            # Update global variables with outputs
            self.global_namespace.update(result.output_variables)
            
            # If execution failed, analyze impact and stop if necessary
            if result.status == CodeExecutionStatus.FAILED:
                impact = workflow.analyze_failure_impact(node_id)
                if not impact["can_continue"]:
                    break
        
        return results
    
    def execute_node(self, node: WorkflowNode, global_vars: Dict[str, Any] = None) -> CodeExecutionResult:
        """Execute a single workflow node."""
        import time
        import sys
        from io import StringIO
        
        start_time = time.time()
        node.status = CodeExecutionStatus.RUNNING
        
        # Prepare execution environment
        local_namespace = dict(self.global_namespace)
        if global_vars:
            local_namespace.update(global_vars)
        
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Execute the code
            exec(node.code, local_namespace)
            
            # Extract results
            output_variables = {}
            for output_name in node.outputs:
                if output_name in local_namespace:
                    output_variables[output_name] = local_namespace[output_name]
            
            execution_time = time.time() - start_time
            
            return CodeExecutionResult(
                node_id=node.id,
                status=CodeExecutionStatus.SUCCESS,
                result=local_namespace.get(node.function_name, None),
                output_variables=output_variables,
                execution_time=execution_time,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Extract error information
            tb = traceback.extract_tb(e.__traceback__)
            if tb:
                last_frame = tb[-1]
                location = CodeLocation(
                    function_name=node.function_name,
                    line_number=last_frame.lineno,
                    file_name=last_frame.filename
                )
            else:
                location = CodeLocation(
                    function_name=node.function_name,
                    line_number=1
                )
            
            error = CodeError(
                error_type=type(e).__name__,
                error_message=str(e),
                location=location,
                traceback_info=traceback.format_exc(),
                context_lines=node.code.split('\n')[max(0, location.line_number-2):location.line_number+1]
            )
            
            return CodeExecutionResult(
                node_id=node.id,
                status=CodeExecutionStatus.FAILED,
                execution_time=execution_time,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                error=error
            )
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# Integration with existing TapeAgents core
class CodeActTape(BaseModel):
    """Tape that stores workflow graphs instead of linear steps."""
    
    workflow: WorkflowGraph = Field(default_factory=WorkflowGraph)
    execution_results: List[CodeExecutionResult] = Field(default_factory=list)
    reflections: List[CodeReflection] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_code_step(self, action: CodeAction) -> str:
        """Add a code action as a workflow node."""
        func_info = action.extract_function_info()
        
        node = WorkflowNode(
            name=func_info.get("name", "unnamed_function"),
            code=action.code,
            function_name=action.function_name,
            inputs=func_info.get("args", []),
            outputs=action.expected_outputs
        )
        
        return self.workflow.add_node(node)
    
    def execute_next_ready_nodes(self) -> List[CodeExecutionResult]:
        """Execute all nodes that are ready to run."""
        executor = WorkflowExecutor()
        ready_nodes = self.workflow.get_ready_nodes()
        results = []
        
        for node in ready_nodes:
            result = executor.execute_node(node, self.workflow.global_variables)
            results.append(result)
            self.execution_results.append(result)
            
            # Update workflow state
            node.status = result.status
            node.error = result.error
            node.result = result.result
        
        return results
    
    def reflect_on_failures(self) -> List[CodeReflection]:
        """Generate reflections for failed nodes."""
        failed_nodes = self.workflow.get_failed_nodes()
        new_reflections = []
        
        for node in failed_nodes:
            reflection = CodeReflection(
                failed_node_id=node.id,
                error_analysis={},
                root_cause="",
                suggested_fixes=[],
                impact_analysis={},
                alternative_approaches=[]
            )
            
            reflection.analyze_failure(self.workflow, node.id)
            new_reflections.append(reflection)
            self.reflections.append(reflection)
        
        return new_reflections
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of workflow execution."""
        total_nodes = len(self.workflow.nodes)
        completed_nodes = len([n for n in self.workflow.nodes.values() 
                              if n.status == CodeExecutionStatus.SUCCESS])
        failed_nodes = len([n for n in self.workflow.nodes.values() 
                           if n.status == CodeExecutionStatus.FAILED])
        
        return {
            "total_nodes": total_nodes,
            "completed_nodes": completed_nodes,
            "failed_nodes": failed_nodes,
            "success_rate": completed_nodes / max(total_nodes, 1),
            "total_execution_time": sum(r.execution_time for r in self.execution_results),
            "critical_path": self.workflow.get_critical_path(),
            "can_continue": failed_nodes == 0 or len(self.workflow.get_ready_nodes()) > 0
        }