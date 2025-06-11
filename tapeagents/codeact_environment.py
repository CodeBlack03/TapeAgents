"""
CodeAct Environment for executing workflow graphs with precise error tracking.

This environment specializes in:
- Executing Python code from workflow nodes
- Tracking execution dependencies
- Providing detailed error information with line-level precision
- Supporting parallel and sequential execution modes
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any, Dict, List, Optional, Set

from .codeact_core import (
    CodeAction, CodeActTape, CodeError, CodeExecutionResult, 
    CodeExecutionStatus, CodeLocation, WorkflowExecutor, WorkflowGraph, WorkflowNode
)
from .core import Action, Observation, Tape
from .environment import Environment

logger = logging.getLogger(__name__)


class CodeActEnvironment(Environment):
    """Environment for executing CodeAct workflow graphs."""
    
    def __init__(self, 
                 execution_mode: str = "sequential",
                 max_parallel_nodes: int = 4,
                 default_timeout: float = 30.0,
                 enable_sandboxing: bool = True,
                 allowed_imports: Optional[List[str]] = None):
        """
        Initialize CodeAct environment.
        
        Args:
            execution_mode: "sequential", "parallel", or "adaptive"
            max_parallel_nodes: Maximum nodes to execute in parallel
            default_timeout: Default timeout for node execution
            enable_sandboxing: Whether to enable code sandboxing
            allowed_imports: List of allowed import modules
        """
        super().__init__()
        self.execution_mode = execution_mode
        self.max_parallel_nodes = max_parallel_nodes
        self.default_timeout = default_timeout
        self.enable_sandboxing = enable_sandboxing
        self.allowed_imports = allowed_imports or [
            "math", "random", "datetime", "json", "re", "os", "sys",
            "collections", "itertools", "functools", "operator",
            "numpy", "pandas", "matplotlib", "seaborn", "requests"
        ]
        
        self.global_namespace = {}
        self.execution_history = []
        self.active_workflows = {}
        
    def react(self, tape: Tape) -> Tape:
        """React to actions in the tape by executing code."""
        
        new_steps = []
        
        # Process new actions in the tape
        for step in tape.steps[len(tape.steps) - tape.metadata.n_added_steps:]:
            if isinstance(step, CodeAction):
                # Execute individual code action
                result = self._execute_code_action(step)
                new_steps.append(result)
                
            elif hasattr(step, 'workflow') and isinstance(step.workflow, WorkflowGraph):
                # Execute workflow graph
                results = self._execute_workflow(step.workflow)
                new_steps.extend(results)
        
        # Create new tape with results
        if new_steps:
            return tape.append(*new_steps)
        
        return tape
    
    def _execute_code_action(self, action: CodeAction) -> CodeExecutionResult:
        """Execute a single code action."""
        
        start_time = time.time()
        
        try:
            # Validate code safety
            if self.enable_sandboxing:
                safety_issues = self._validate_code_safety(action.code)
                if safety_issues:
                    error = CodeError(
                        error_type="SecurityError",
                        error_message=f"Code safety violations: {safety_issues}",
                        location=CodeLocation(function_name=action.function_name, line_number=1),
                        traceback_info=""
                    )
                    
                    return CodeExecutionResult(
                        node_id="single_action",
                        status=CodeExecutionStatus.FAILED,
                        execution_time=time.time() - start_time,
                        error=error
                    )
            
            # Execute the code
            result = self._execute_code_safely(
                action.code, 
                action.function_name,
                timeout=action.timeout
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error executing code action: {e}")
            
            error = CodeError(
                error_type=type(e).__name__,
                error_message=str(e),
                location=CodeLocation(function_name=action.function_name, line_number=1),
                traceback_info=traceback.format_exc()
            )
            
            return CodeExecutionResult(
                node_id="single_action",
                status=CodeExecutionStatus.FAILED,
                execution_time=time.time() - start_time,
                error=error
            )
    
    def _execute_workflow(self, workflow: WorkflowGraph) -> List[CodeExecutionResult]:
        """Execute a complete workflow graph."""
        
        if self.execution_mode == "sequential":
            return self._execute_workflow_sequential(workflow)
        elif self.execution_mode == "parallel":
            return self._execute_workflow_parallel(workflow)
        elif self.execution_mode == "adaptive":
            return self._execute_workflow_adaptive(workflow)
        else:
            raise ValueError(f"Unknown execution mode: {self.execution_mode}")
    
    def _execute_workflow_sequential(self, workflow: WorkflowGraph) -> List[CodeExecutionResult]:
        """Execute workflow nodes sequentially based on dependencies."""
        
        results = []
        execution_path = workflow.get_execution_path()
        
        for node_id in execution_path:
            node = workflow.nodes[node_id]
            
            # Check if node is ready to execute
            completed_nodes = {nid for nid, n in workflow.nodes.items() 
                             if n.status == CodeExecutionStatus.SUCCESS}
            
            if not node.is_ready_to_execute(completed_nodes):
                continue
            
            # Execute the node
            result = self._execute_workflow_node(node, workflow.global_variables)
            results.append(result)
            
            # Update node status
            node.status = result.status
            node.error = result.error
            node.result = result.result
            node.execution_time = result.execution_time
            
            # Update global variables
            workflow.global_variables.update(result.output_variables)
            
            # Stop on failure if no alternative paths
            if result.status == CodeExecutionStatus.FAILED:
                impact = workflow.analyze_failure_impact(node_id)
                if not impact["can_continue"]:
                    logger.warning(f"Stopping workflow execution due to critical failure in {node.name}")
                    break
        
        return results
    
    def _execute_workflow_parallel(self, workflow: WorkflowGraph) -> List[CodeExecutionResult]:
        """Execute workflow nodes in parallel where possible."""
        
        results = []
        completed_nodes = set()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_nodes) as executor:
            
            while len(completed_nodes) < len(workflow.nodes):
                # Find nodes ready for execution
                ready_nodes = [
                    node for node in workflow.nodes.values()
                    if (node.status == CodeExecutionStatus.PENDING and
                        node.is_ready_to_execute(completed_nodes))
                ]
                
                if not ready_nodes:
                    break  # No more nodes can be executed
                
                # Submit ready nodes for execution
                future_to_node = {}
                for node in ready_nodes:
                    future = executor.submit(
                        self._execute_workflow_node, 
                        node, 
                        workflow.global_variables.copy()
                    )
                    future_to_node[future] = node
                
                # Wait for completion
                for future in concurrent.futures.as_completed(future_to_node):
                    node = future_to_node[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Update node status
                        node.status = result.status
                        node.error = result.error
                        node.result = result.result
                        node.execution_time = result.execution_time
                        
                        if result.status == CodeExecutionStatus.SUCCESS:
                            completed_nodes.add(node.id)
                            # Update global variables (thread-safe update needed)
                            workflow.global_variables.update(result.output_variables)
                        
                    except Exception as e:
                        logger.error(f"Error in parallel execution of {node.name}: {e}")
                        node.status = CodeExecutionStatus.FAILED
        
        return results
    
    def _execute_workflow_adaptive(self, workflow: WorkflowGraph) -> List[CodeExecutionResult]:
        """Execute workflow using adaptive strategy based on dependencies."""
        
        # Analyze workflow structure to decide execution strategy
        total_nodes = len(workflow.nodes)
        avg_dependencies = sum(len(node.dependencies) for node in workflow.nodes.values()) / total_nodes
        
        # Use parallel execution if low dependency coupling
        if avg_dependencies < 2.0 and total_nodes > 3:
            return self._execute_workflow_parallel(workflow)
        else:
            return self._execute_workflow_sequential(workflow)
    
    def _execute_workflow_node(self, node: WorkflowNode, global_vars: Dict[str, Any]) -> CodeExecutionResult:
        """Execute a single workflow node with detailed error tracking."""
        
        start_time = time.time()
        node.status = CodeExecutionStatus.RUNNING
        
        try:
            # Prepare execution environment
            local_namespace = dict(self.global_namespace)
            local_namespace.update(global_vars)
            
            # Execute with monitoring
            result = self._execute_code_safely(
                node.code,
                node.function_name,
                local_namespace,
                timeout=self.default_timeout
            )
            
            # Extract output variables
            output_variables = {}
            for output_name in node.outputs:
                if output_name in result.output_variables:
                    output_variables[output_name] = result.output_variables[output_name]
            
            # Update result with node-specific information
            result.node_id = node.id
            result.output_variables = output_variables
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Create detailed error information
            tb = traceback.extract_tb(e.__traceback__)
            location = CodeLocation(
                function_name=node.function_name,
                line_number=tb[-1].lineno if tb else 1,
                file_name="<workflow_node>"
            )
            
            error = CodeError(
                error_type=type(e).__name__,
                error_message=str(e),
                location=location,
                traceback_info=traceback.format_exc(),
                context_lines=self._get_context_lines(node.code, location.line_number)
            )
            
            return CodeExecutionResult(
                node_id=node.id,
                status=CodeExecutionStatus.FAILED,
                execution_time=execution_time,
                error=error
            )
    
    def _execute_code_safely(self, 
                           code: str, 
                           function_name: str,
                           namespace: Optional[Dict[str, Any]] = None,
                           timeout: float = 30.0) -> CodeExecutionResult:
        """Execute code safely with comprehensive monitoring."""
        
        if namespace is None:
            namespace = dict(self.global_namespace)
        
        start_time = time.time()
        
        # Capture stdout/stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute with timeout
                if timeout > 0:
                    with self._timeout_context(timeout):
                        exec(code, namespace)
                else:
                    exec(code, namespace)
            
            execution_time = time.time() - start_time
            
            # Extract function result if available
            function_result = namespace.get(function_name)
            
            # Extract all new variables as potential outputs
            output_variables = {
                k: v for k, v in namespace.items()
                if k not in self.global_namespace and not k.startswith('_')
            }
            
            return CodeExecutionResult(
                node_id="",  # Will be set by caller
                status=CodeExecutionStatus.SUCCESS,
                result=function_result,
                output_variables=output_variables,
                execution_time=execution_time,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue()
            )
            
        except TimeoutError:
            execution_time = time.time() - start_time
            
            error = CodeError(
                error_type="TimeoutError",
                error_message=f"Code execution timed out after {timeout} seconds",
                location=CodeLocation(function_name=function_name, line_number=1),
                traceback_info=""
            )
            
            return CodeExecutionResult(
                node_id="",
                status=CodeExecutionStatus.FAILED,
                execution_time=execution_time,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                error=error
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Extract detailed error location
            tb = traceback.extract_tb(e.__traceback__)
            if tb:
                # Find the frame that corresponds to our executed code
                for frame in reversed(tb):
                    if frame.filename == '<string>':
                        location = CodeLocation(
                            function_name=function_name,
                            line_number=frame.lineno,
                            file_name="<executed_code>"
                        )
                        break
                else:
                    location = CodeLocation(function_name=function_name, line_number=1)
            else:
                location = CodeLocation(function_name=function_name, line_number=1)
            
            error = CodeError(
                error_type=type(e).__name__,
                error_message=str(e),
                location=location,
                traceback_info=traceback.format_exc(),
                context_lines=self._get_context_lines(code, location.line_number)
            )
            
            return CodeExecutionResult(
                node_id="",
                status=CodeExecutionStatus.FAILED,
                execution_time=execution_time,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                error=error
            )
    
    @contextmanager
    def _timeout_context(self, timeout: float):
        """Context manager for code execution timeout."""
        
        def timeout_handler():
            raise TimeoutError(f"Code execution timed out after {timeout} seconds")
        
        # Simple timeout implementation using threading
        import threading
        timer = threading.Timer(timeout, timeout_handler)
        timer.start()
        
        try:
            yield
        finally:
            timer.cancel()
    
    def _validate_code_safety(self, code: str) -> List[str]:
        """Validate code for safety issues."""
        
        safety_issues = []
        
        # Check for dangerous operations
        dangerous_patterns = [
            ("os.system", "System command execution"),
            ("subprocess", "Subprocess execution"),
            ("eval", "Dynamic code evaluation"),
            ("exec", "Dynamic code execution"),
            ("__import__", "Dynamic imports"),
            ("open(", "File operations"),
            ("file(", "File operations"),
            ("input(", "User input"),
            ("raw_input(", "User input")
        ]
        
        for pattern, description in dangerous_patterns:
            if pattern in code:
                safety_issues.append(f"{description}: {pattern}")
        
        # Check imports
        import ast
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.allowed_imports:
                            safety_issues.append(f"Disallowed import: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module not in self.allowed_imports:
                        safety_issues.append(f"Disallowed import: {node.module}")
        except SyntaxError:
            safety_issues.append("Syntax error in code")
        
        return safety_issues
    
    def _get_context_lines(self, code: str, line_number: int, context: int = 2) -> List[str]:
        """Get context lines around an error location."""
        
        lines = code.split('\n')
        start = max(0, line_number - context - 1)
        end = min(len(lines), line_number + context)
        
        context_lines = []
        for i in range(start, end):
            prefix = ">>> " if i == line_number - 1 else "    "
            context_lines.append(f"{prefix}{i+1:3d}: {lines[i]}")
        
        return context_lines
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all code executions."""
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len([h for h in self.execution_history if h.get("success", False)]),
            "failed_executions": len([h for h in self.execution_history if not h.get("success", True)]),
            "total_execution_time": sum(h.get("execution_time", 0) for h in self.execution_history),
            "average_execution_time": sum(h.get("execution_time", 0) for h in self.execution_history) / max(len(self.execution_history), 1),
            "active_workflows": len(self.active_workflows)
        }
    
    def reset_environment(self):
        """Reset the execution environment."""
        self.global_namespace.clear()
        self.execution_history.clear()
        self.active_workflows.clear()
    
    def actions(self) -> List[type[Action]]:
        """Return list of supported actions."""
        from .codeact_core import CodeAction
        return [CodeAction]
    
    def tools_description(self) -> str:
        """Return description of available tools."""
        return f"""
CodeAct Environment Tools:

1. Code Execution:
   - Execute Python functions with dependency tracking
   - Support for {self.execution_mode} execution mode
   - Automatic error localization and reporting
   - Timeout protection ({self.default_timeout}s default)

2. Workflow Management:
   - Dependency graph execution
   - Parallel execution (up to {self.max_parallel_nodes} nodes)
   - Failure impact analysis
   - Alternative path discovery

3. Safety Features:
   - Code sandboxing: {'Enabled' if self.enable_sandboxing else 'Disabled'}
   - Allowed imports: {', '.join(self.allowed_imports)}
   - Execution monitoring and limits

4. Error Analysis:
   - Line-level error localization
   - Context-aware error reporting
   - Traceback analysis
   - Suggested fixes generation

Use CodeAction to execute individual functions or CodePlan to execute complete workflows.
"""


class AsyncCodeActEnvironment(CodeActEnvironment):
    """Async version of CodeAct environment for concurrent execution."""
    
    async def areact(self, tape: Tape) -> Tape:
        """Async version of react method."""
        
        new_steps = []
        
        # Process new actions in the tape
        for step in tape.steps[len(tape.steps) - tape.metadata.n_added_steps:]:
            if isinstance(step, CodeAction):
                # Execute individual code action
                result = await self._aexecute_code_action(step)
                new_steps.append(result)
                
            elif hasattr(step, 'workflow') and isinstance(step.workflow, WorkflowGraph):
                # Execute workflow graph
                results = await self._aexecute_workflow(step.workflow)
                new_steps.extend(results)
        
        # Create new tape with results
        if new_steps:
            return tape.append(*new_steps)
        
        return tape
    
    async def _aexecute_code_action(self, action: CodeAction) -> CodeExecutionResult:
        """Async execute a single code action."""
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_code_action, action)
    
    async def _aexecute_workflow(self, workflow: WorkflowGraph) -> List[CodeExecutionResult]:
        """Async execute a complete workflow graph."""
        
        if self.execution_mode == "parallel":
            return await self._aexecute_workflow_parallel(workflow)
        else:
            # Run sequential execution in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._execute_workflow_sequential, workflow)
    
    async def _aexecute_workflow_parallel(self, workflow: WorkflowGraph) -> List[CodeExecutionResult]:
        """Async parallel execution of workflow nodes."""
        
        results = []
        completed_nodes = set()
        
        while len(completed_nodes) < len(workflow.nodes):
            # Find nodes ready for execution
            ready_nodes = [
                node for node in workflow.nodes.values()
                if (node.status == CodeExecutionStatus.PENDING and
                    node.is_ready_to_execute(completed_nodes))
            ]
            
            if not ready_nodes:
                break  # No more nodes can be executed
            
            # Execute ready nodes concurrently
            tasks = []
            for node in ready_nodes:
                task = asyncio.create_task(
                    self._aexecute_workflow_node(node, workflow.global_variables.copy())
                )
                tasks.append((task, node))
            
            # Wait for completion
            for task, node in tasks:
                try:
                    result = await task
                    results.append(result)
                    
                    # Update node status
                    node.status = result.status
                    node.error = result.error
                    node.result = result.result
                    node.execution_time = result.execution_time
                    
                    if result.status == CodeExecutionStatus.SUCCESS:
                        completed_nodes.add(node.id)
                        workflow.global_variables.update(result.output_variables)
                    
                except Exception as e:
                    logger.error(f"Error in async execution of {node.name}: {e}")
                    node.status = CodeExecutionStatus.FAILED
        
        return results
    
    async def _aexecute_workflow_node(self, node: WorkflowNode, global_vars: Dict[str, Any]) -> CodeExecutionResult:
        """Async execute a single workflow node."""
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_workflow_node, node, global_vars)