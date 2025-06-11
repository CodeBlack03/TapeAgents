"""
CodeAct Agent Implementation

This module implements agents that use CodeAct-style planning where:
- Plans are generated as executable Python code
- Execution is tracked through workflow dependency graphs
- Errors are precisely localized and reflected upon
- Self-correction targets only failed sub-tasks
"""

from __future__ import annotations

import ast
import json
import logging
from typing import Any, Dict, Generator, List, Optional

from pydantic import Field

from .agent import Agent, Node
from .core import Action, Observation, Prompt, Step, Tape, Thought
from .codeact_core import (
    CodeAction, CodeActTape, CodeError, CodeExecutionResult, CodePlan, 
    CodeReflection, DependencyType, WorkflowExecutor, WorkflowGraph, 
    WorkflowNode, CodeExecutionStatus
)
from .llms import LLMStream

logger = logging.getLogger(__name__)


class CodePlanningNode(Node):
    """Node that generates executable code plans instead of text plans."""
    
    name: str = "code_planning"
    planning_style: str = Field(default="functional", description="Planning style: functional, object_oriented, procedural")
    max_complexity: float = Field(default=10.0, description="Maximum allowed plan complexity")
    safety_level: str = Field(default="safe", description="Code safety level")
    
    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        """Create a prompt for generating executable code plans."""
        
        # Extract task context from tape
        task_description = self._extract_task_description(tape)
        available_tools = self._get_available_tools(agent)
        previous_failures = self._get_previous_failures(tape)
        
        system_prompt = f"""You are a CodeAct planning agent. Your task is to create executable Python code plans.

PLANNING PRINCIPLES:
1. Break complex tasks into small, testable functions
2. Each function should have clear inputs and outputs
3. Use dependency relationships between functions
4. Include error handling and validation
5. Make code modular and reusable

AVAILABLE TOOLS: {available_tools}

TASK: {task_description}

PREVIOUS FAILURES: {previous_failures}

Generate a plan as Python code with the following structure:

```python
# Plan: [Brief description of the overall approach]

def step_1_function_name(input_params):
    \"\"\"
    Description: What this step does
    Inputs: List of input parameters
    Outputs: List of output variables
    Dependencies: List of functions this depends on
    \"\"\"
    # Implementation code here
    return result

def step_2_function_name(input_from_step1):
    \"\"\"
    Description: What this step does
    Inputs: input_from_step1
    Outputs: final_result
    Dependencies: step_1_function_name
    \"\"\"
    # Implementation code here
    return final_result

# Execution plan:
# 1. Execute step_1_function_name with initial inputs
# 2. Execute step_2_function_name with outputs from step 1
```

REQUIREMENTS:
- Each function must be self-contained and testable
- Include clear docstrings with Dependencies information
- Use descriptive function and variable names
- Add input validation and error handling
- Keep functions focused on single responsibilities
- Ensure proper data flow between functions
"""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for step in tape.steps[-5:]:  # Last 5 steps for context
            if isinstance(step, Thought):
                messages.append({"role": "assistant", "content": step.content})
            elif isinstance(step, Observation):
                messages.append({"role": "user", "content": f"Observation: {step.content}"})
        
        return Prompt(messages=messages)
    
    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream) -> Generator[Step, None, None]:
        """Generate a code plan from LLM output."""
        
        plan_code = llm_stream.get_text()
        
        try:
            # Parse the generated code plan
            workflow = self._parse_code_plan(plan_code)
            
            # Create CodePlan step
            code_plan = CodePlan(
                workflow=workflow,
                plan_description=self._extract_plan_description(plan_code),
                complexity_score=workflow.estimate_complexity() if hasattr(workflow, 'estimate_complexity') else 0.0
            )
            
            yield code_plan
            
            # Generate individual CodeAction steps for each function
            for node_id, node in workflow.nodes.items():
                code_action = CodeAction(
                    code=node.code,
                    function_name=node.function_name,
                    expected_outputs=node.outputs,
                    safety_level=self.safety_level
                )
                
                # Validate code safety
                safety_issues = code_action.validate_code_safety()
                if safety_issues and self.safety_level == "strict":
                    yield Thought(content=f"Safety issues detected in {node.name}: {safety_issues}")
                    continue
                
                yield code_action
            
            # Set next node to execution
            from .core import SetWorkflowNode
            ready_nodes = workflow.get_ready_nodes()
            if ready_nodes:
                yield SetWorkflowNode(node_id=ready_nodes[0].id)
        
        except Exception as e:
            logger.error(f"Failed to parse code plan: {e}")
            yield Thought(content=f"Failed to generate valid code plan: {e}")
            yield Thought(content="Falling back to text-based planning...")
    
    def _extract_task_description(self, tape: Tape) -> str:
        """Extract task description from tape."""
        for step in tape.steps:
            if isinstance(step, Observation) and "task" in step.content.lower():
                return step.content
        return "No specific task description found"
    
    def _get_available_tools(self, agent: Any) -> str:
        """Get description of available tools."""
        if hasattr(agent, 'tools_description'):
            return agent.tools_description
        return "Standard Python libraries and functions"
    
    def _get_previous_failures(self, tape: Tape) -> str:
        """Extract information about previous failures."""
        failures = []
        for step in tape.steps:
            if isinstance(step, CodeExecutionResult) and not step.is_success():
                failures.append(f"- {step.get_error_summary()}")
        
        if failures:
            return "Previous failures to avoid:\n" + "\n".join(failures)
        return "No previous failures"
    
    def _parse_code_plan(self, plan_code: str) -> WorkflowGraph:
        """Parse generated code into a workflow graph."""
        workflow = WorkflowGraph()
        
        try:
            # Parse the Python code
            tree = ast.parse(plan_code)
            
            # Extract functions and their metadata
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = self._extract_function_info(node, plan_code)
                    functions.append(func_info)
            
            # Create workflow nodes
            node_map = {}
            for func_info in functions:
                workflow_node = WorkflowNode(
                    name=func_info["name"],
                    code=func_info["code"],
                    function_name=func_info["name"],
                    inputs=func_info["inputs"],
                    outputs=func_info["outputs"]
                )
                node_id = workflow.add_node(workflow_node)
                node_map[func_info["name"]] = node_id
            
            # Add dependencies based on docstring information
            for func_info in functions:
                if func_info["dependencies"]:
                    for dep_name in func_info["dependencies"]:
                        if dep_name in node_map:
                            workflow.add_dependency(
                                node_map[dep_name], 
                                node_map[func_info["name"]], 
                                DependencyType.DATA_DEPENDENCY
                            )
            
            return workflow
            
        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            raise ValueError(f"Generated code has syntax errors: {e}")
    
    def _extract_function_info(self, func_node: ast.FunctionDef, full_code: str) -> Dict[str, Any]:
        """Extract information about a function from its AST node."""
        
        # Get function code
        lines = full_code.split('\n')
        func_start = func_node.lineno - 1
        func_end = func_node.end_lineno if hasattr(func_node, 'end_lineno') else len(lines)
        func_code = '\n'.join(lines[func_start:func_end])
        
        # Parse docstring for metadata
        docstring = ast.get_docstring(func_node) or ""
        
        # Extract inputs from function arguments
        inputs = [arg.arg for arg in func_node.args.args]
        
        # Parse docstring for outputs and dependencies
        outputs = self._parse_docstring_field(docstring, "Outputs")
        dependencies = self._parse_docstring_field(docstring, "Dependencies")
        
        return {
            "name": func_node.name,
            "code": func_code,
            "inputs": inputs,
            "outputs": outputs,
            "dependencies": dependencies,
            "docstring": docstring,
            "line_number": func_node.lineno
        }
    
    def _parse_docstring_field(self, docstring: str, field_name: str) -> List[str]:
        """Parse a specific field from a function docstring."""
        lines = docstring.split('\n')
        in_field = False
        field_values = []
        
        for line in lines:
            line = line.strip()
            if line.startswith(f"{field_name}:"):
                in_field = True
                # Extract value from same line if present
                value = line[len(f"{field_name}:"):].strip()
                if value:
                    field_values.extend([v.strip() for v in value.split(',')])
            elif in_field and line and not line.endswith(':'):
                # Continue reading field values
                field_values.extend([v.strip() for v in line.split(',')])
            elif in_field and (line.endswith(':') or not line):
                # End of field
                in_field = False
        
        return [v for v in field_values if v]
    
    def _extract_plan_description(self, plan_code: str) -> str:
        """Extract plan description from code comments."""
        lines = plan_code.split('\n')
        for line in lines:
            if line.strip().startswith('# Plan:'):
                return line.strip()[7:].strip()
        return "Generated code plan"


class CodeExecutionNode(Node):
    """Node that executes workflow nodes and handles errors."""
    
    name: str = "code_execution"
    timeout: float = Field(default=30.0, description="Execution timeout per node")
    max_retries: int = Field(default=2, description="Maximum retries for failed nodes")
    
    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        """Create prompt for code execution (usually empty as this is rule-based)."""
        return Prompt()  # Empty prompt - execution is rule-based
    
    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream) -> Generator[Step, None, None]:
        """Execute ready workflow nodes."""
        
        # Find the current workflow from the tape
        workflow = self._get_current_workflow(tape)
        if not workflow:
            yield Thought(content="No workflow found for execution")
            return
        
        # Get nodes ready for execution
        ready_nodes = workflow.get_ready_nodes()
        if not ready_nodes:
            yield Thought(content="No nodes ready for execution")
            return
        
        # Execute ready nodes
        executor = WorkflowExecutor()
        
        for node in ready_nodes:
            yield Thought(content=f"Executing node: {node.name}")
            
            try:
                # Execute the node
                result = executor.execute_node(node, workflow.global_variables)
                yield result
                
                # Update workflow state
                node.status = result.status
                node.error = result.error
                node.result = result.result
                node.execution_time = result.execution_time
                
                # Update global variables
                workflow.global_variables.update(result.output_variables)
                
                if result.is_success():
                    yield Thought(content=f"Successfully executed {node.name}")
                else:
                    yield Thought(content=f"Failed to execute {node.name}: {result.get_error_summary()}")
                    
                    # Trigger reflection on failure
                    from .core import SetWorkflowNode
                    yield SetWorkflowNode(node_id="reflection")
                
            except Exception as e:
                logger.error(f"Unexpected error executing node {node.name}: {e}")
                yield Thought(content=f"Unexpected error in {node.name}: {e}")
        
        # Check if more nodes are ready after execution
        remaining_ready = workflow.get_ready_nodes()
        if remaining_ready:
            from .core import SetWorkflowNode
            yield SetWorkflowNode(node_id="code_execution")
        else:
            # Check if workflow is complete
            all_completed = all(
                node.status in [CodeExecutionStatus.SUCCESS, CodeExecutionStatus.SKIPPED]
                for node in workflow.nodes.values()
            )
            
            if all_completed:
                yield Thought(content="Workflow execution completed successfully")
            else:
                failed_nodes = workflow.get_failed_nodes()
                if failed_nodes:
                    yield Thought(content=f"Workflow has failed nodes: {[n.name for n in failed_nodes]}")
                    from .core import SetWorkflowNode
                    yield SetWorkflowNode(node_id="reflection")
    
    def _get_current_workflow(self, tape: Tape) -> Optional[WorkflowGraph]:
        """Extract the current workflow from the tape."""
        for step in reversed(tape.steps):
            if isinstance(step, CodePlan):
                return step.workflow
        return None


class CodeReflectionNode(Node):
    """Node that reflects on code execution failures and generates fixes."""
    
    name: str = "reflection"
    max_reflection_depth: int = Field(default=3, description="Maximum reflection depth")
    
    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        """Create prompt for reflecting on code failures."""
        
        # Get failed nodes and their errors
        workflow = self._get_current_workflow(tape)
        failed_nodes = workflow.get_failed_nodes() if workflow else []
        
        if not failed_nodes:
            return Prompt()  # No failures to reflect on
        
        # Focus on the most recent failure
        failed_node = failed_nodes[-1]
        error = failed_node.error
        
        system_prompt = f"""You are a code debugging and reflection expert. A function execution has failed and you need to:

1. Analyze the specific error
2. Identify the root cause
3. Generate a corrected version of the function
4. Suggest alternative approaches if needed

FAILED FUNCTION: {failed_node.name}
ERROR TYPE: {error.error_type if error else 'Unknown'}
ERROR MESSAGE: {error.error_message if error else 'Unknown'}
ERROR LOCATION: {error.location if error else 'Unknown'}

ORIGINAL CODE:
```python
{failed_node.code}
```

CONTEXT LINES AROUND ERROR:
{error.context_lines if error else []}

ANALYSIS REQUIREMENTS:
1. Focus ONLY on this specific function - don't redesign the entire workflow
2. Identify the exact cause of the failure
3. Provide a corrected version of the function
4. Explain what was wrong and how the fix addresses it
5. Consider edge cases and add appropriate error handling

Generate your response in this format:

## Error Analysis
[Detailed analysis of what went wrong]

## Root Cause
[The fundamental reason for the failure]

## Corrected Code
```python
def {failed_node.function_name}(parameters):
    \"\"\"
    [Updated docstring with any changes]
    \"\"\"
    # [Corrected implementation]
    return result
```

## Explanation
[Explanation of the changes made and why they fix the issue]

## Alternative Approaches
[If applicable, suggest alternative ways to implement this function]
"""

        return Prompt(messages=[{"role": "system", "content": system_prompt}])
    
    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream) -> Generator[Step, None, None]:
        """Generate reflection and corrected code."""
        
        workflow = self._get_current_workflow(tape)
        if not workflow:
            yield Thought(content="No workflow found for reflection")
            return
        
        failed_nodes = workflow.get_failed_nodes()
        if not failed_nodes:
            yield Thought(content="No failed nodes to reflect on")
            return
        
        # Get reflection from LLM
        reflection_text = llm_stream.get_text()
        
        # Parse the reflection
        corrected_code = self._extract_corrected_code(reflection_text)
        error_analysis = self._extract_error_analysis(reflection_text)
        
        # Focus on the most recent failure
        failed_node = failed_nodes[-1]
        
        # Create reflection step
        reflection = CodeReflection(
            failed_node_id=failed_node.id,
            error_analysis={"analysis": error_analysis},
            root_cause=self._extract_root_cause(reflection_text),
            suggested_fixes=[corrected_code] if corrected_code else [],
            impact_analysis=workflow.analyze_failure_impact(failed_node.id),
            alternative_approaches=self._extract_alternatives(reflection_text)
        )
        
        yield reflection
        
        # If we have corrected code, create a new action to replace the failed one
        if corrected_code:
            yield Thought(content=f"Generated corrected code for {failed_node.name}")
            
            # Create new CodeAction with corrected code
            corrected_action = CodeAction(
                code=corrected_code,
                function_name=failed_node.function_name,
                expected_outputs=failed_node.outputs,
                safety_level="safe"
            )
            
            yield corrected_action
            
            # Update the workflow node with corrected code
            failed_node.code = corrected_code
            failed_node.status = CodeExecutionStatus.PENDING
            failed_node.error = None
            
            # Set next node to retry execution
            from .core import SetWorkflowNode
            yield SetWorkflowNode(node_id="code_execution")
        else:
            yield Thought(content="Could not generate corrected code. Manual intervention may be needed.")
    
    def _get_current_workflow(self, tape: Tape) -> Optional[WorkflowGraph]:
        """Extract the current workflow from the tape."""
        for step in reversed(tape.steps):
            if isinstance(step, CodePlan):
                return step.workflow
        return None
    
    def _extract_corrected_code(self, reflection_text: str) -> Optional[str]:
        """Extract corrected code from reflection text."""
        lines = reflection_text.split('\n')
        in_code_block = False
        code_lines = []
        
        for line in lines:
            if line.strip().startswith('```python'):
                in_code_block = True
                continue
            elif line.strip() == '```' and in_code_block:
                break
            elif in_code_block:
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else None
    
    def _extract_error_analysis(self, reflection_text: str) -> str:
        """Extract error analysis from reflection text."""
        lines = reflection_text.split('\n')
        analysis_lines = []
        in_analysis = False
        
        for line in lines:
            if line.strip().startswith('## Error Analysis'):
                in_analysis = True
                continue
            elif line.strip().startswith('##') and in_analysis:
                break
            elif in_analysis:
                analysis_lines.append(line)
        
        return '\n'.join(analysis_lines).strip()
    
    def _extract_root_cause(self, reflection_text: str) -> str:
        """Extract root cause from reflection text."""
        lines = reflection_text.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('## Root Cause'):
                if i + 1 < len(lines):
                    return lines[i + 1].strip()
        return "Root cause not identified"
    
    def _extract_alternatives(self, reflection_text: str) -> List[str]:
        """Extract alternative approaches from reflection text."""
        lines = reflection_text.split('\n')
        alternatives = []
        in_alternatives = False
        
        for line in lines:
            if line.strip().startswith('## Alternative Approaches'):
                in_alternatives = True
                continue
            elif line.strip().startswith('##') and in_alternatives:
                break
            elif in_alternatives and line.strip():
                alternatives.append(line.strip())
        
        return alternatives


class CodeActAgent(Agent):
    """Agent that uses CodeAct-style planning with workflow graphs."""
    
    def __init__(self, **kwargs):
        # Set up CodeAct-specific nodes
        nodes = kwargs.get('nodes', [])
        if not nodes:
            nodes = [
                CodePlanningNode(),
                CodeExecutionNode(), 
                CodeReflectionNode()
            ]
        kwargs['nodes'] = nodes
        
        super().__init__(**kwargs)
    
    def select_node(self, tape: Tape) -> Node:
        """Select node based on workflow state and execution needs."""
        
        # Check if we have a specific node set
        from .core import SetWorkflowNode
        for step in reversed(tape.steps):
            if isinstance(step, SetWorkflowNode):
                node_name = step.node_id
                for node in self.nodes:
                    if node.name == node_name:
                        return node
                break
        
        # Default node selection logic
        workflow = self._get_current_workflow(tape)
        
        if not workflow:
            # No workflow yet, start with planning
            return self.find_node("code_planning")
        
        # Check for failed nodes that need reflection
        failed_nodes = workflow.get_failed_nodes()
        if failed_nodes:
            return self.find_node("reflection")
        
        # Check for ready nodes that need execution
        ready_nodes = workflow.get_ready_nodes()
        if ready_nodes:
            return self.find_node("code_execution")
        
        # Check if workflow is complete
        all_completed = all(
            node.status in [CodeExecutionStatus.SUCCESS, CodeExecutionStatus.SKIPPED]
            for node in workflow.nodes.values()
        )
        
        if all_completed:
            # Workflow complete, might need new planning
            return self.find_node("code_planning")
        
        # Default to planning
        return self.find_node("code_planning")
    
    def _get_current_workflow(self, tape: Tape) -> Optional[WorkflowGraph]:
        """Extract the current workflow from the tape."""
        for step in reversed(tape.steps):
            if isinstance(step, CodePlan):
                return step.workflow
        return None
    
    def should_stop(self, tape: Tape) -> bool:
        """Determine if the agent should stop execution."""
        
        # Check if we have a complete workflow with all nodes successful
        workflow = self._get_current_workflow(tape)
        if workflow:
            all_completed = all(
                node.status == CodeExecutionStatus.SUCCESS
                for node in workflow.nodes.values()
            )
            
            if all_completed:
                return True
            
            # Check if we have too many failures
            failed_nodes = workflow.get_failed_nodes()
            if len(failed_nodes) > 3:  # Too many failures
                return True
        
        # Check for explicit stop conditions
        for step in reversed(tape.steps[-5:]):
            if isinstance(step, Thought) and "stop" in step.content.lower():
                return True
        
        return super().should_stop(tape)


# Factory function for creating CodeAct agents
def create_codeact_agent(llm, **kwargs) -> CodeActAgent:
    """Create a CodeAct agent with default configuration."""
    
    return CodeActAgent.create(
        llm=llm,
        name=kwargs.get('name', 'CodeActAgent'),
        max_iterations=kwargs.get('max_iterations', 50),
        **kwargs
    )