# Planning and Orchestration in TapeAgents

This guide explains how TapeAgents handles planning, task decomposition, and orchestration to perform complex tasks.

## 🏗️ Architecture Overview

TapeAgents uses a multi-layered orchestration approach:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Main Loop Orchestrator                      │
│                  (Agent-Environment Loop)                      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
┌─────────┐    ┌─────────────┐    ┌─────────────┐
│ Agent   │    │ Environment │    │   Nodes     │
│Manager  │◄──►│  Reactor    │    │ (Planning   │
│         │    │             │    │  Units)     │
└─────────┘    └─────────────┘    └─────────────┘
    │                                     │
    ▼                                     ▼
┌─────────┐                      ┌─────────────┐
│Sub-     │                      │ Prompt      │
│Agents   │                      │ Generation  │
│         │                      │ & Step Gen  │
└─────────┘                      └─────────────┘
```

## 🎯 Core Planning Components

### 1. **Agent-Level Orchestration**

The `Agent` class serves as the main orchestrator:

```python
class Agent(BaseModel, Generic[TapeType]):
    """
    Main orchestrator that coordinates planning and execution.
    
    Key responsibilities:
    - Select appropriate nodes for each planning step
    - Delegate to subagents for specialized tasks
    - Manage the overall execution flow
    - Coordinate between different planning components
    """
    
    def select_node(self, tape: TapeType) -> Node:
        """
        Core planning decision: which node should handle the current state?
        
        Selection logic:
        1. If next_node is explicitly set → use that node
        2. If no nodes have run yet → use first node  
        3. Otherwise → use next node in sequence
        """
        view = self.compute_view(tape).top
        if view.next_node:
            return self.find_node(view.next_node)
        if not view.last_node:
            return self.nodes[0]
        # Select next node in sequence
        for i, node in enumerate(self.nodes):
            if node.name == view.last_node and i + 1 < len(self.nodes):
                return self.nodes[i + 1]
```

### 2. **Node-Based Task Decomposition**

Each `Node` represents an atomic planning unit:

```python
class Node(BaseModel):
    """
    Atomic unit of agent behavior - handles specific planning aspects.
    
    Each node can:
    - Analyze the current tape state
    - Generate appropriate prompts for the LLM
    - Process LLM output into concrete steps
    - Decide what should happen next
    """
    
    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        """Create context-specific prompt for this planning step."""
        
    def generate_steps(self, agent: Any, tape: Tape, llm_stream: LLMStream):
        """Convert LLM output into executable steps."""
```

### 3. **Main Loop Orchestration**

The orchestrator manages the agent-environment interaction:

```python
def main_loop(agent: Agent, start_tape: TapeType, environment: Environment):
    """
    Main planning and execution loop:
    
    1. Agent analyzes current state and plans next actions
    2. Environment reacts to agent actions  
    3. Agent observes results and replans
    4. Repeat until task completion
    """
    while not finished:
        # PLANNING PHASE: Agent decides what to do
        for event in agent.run(tape):
            # Agent uses nodes to plan and generate steps
            yield MainLoopEvent(agent_event=event)
        
        # EXECUTION PHASE: Environment executes actions
        tape = environment.react(agent_tape)
        
        # OBSERVATION PHASE: Agent observes results
        for observation in new_observations:
            yield MainLoopEvent(observation=observation)
```

## 🧠 Planning Strategies

### 1. **Sequential Node Planning**

Simple linear planning through predefined nodes:

```python
class SequentialPlanningAgent(Agent):
    """
    Plans by moving through nodes in sequence.
    Each node handles a specific aspect of the task.
    """
    nodes = [
        AnalyzeTaskNode(),      # Understand the task
        DecomposeTaskNode(),    # Break into subtasks  
        ExecuteTaskNode(),      # Execute subtasks
        ValidateResultNode()    # Verify completion
    ]
```

### 2. **Conditional Node Planning**

Dynamic planning based on tape state:

```python
class ConditionalPlanningNode(Node):
    def generate_steps(self, agent, tape, llm_stream):
        # Analyze current state
        current_state = self.analyze_tape_state(tape)
        
        if current_state.needs_more_info:
            yield SetNextNode(next_node="information_gathering")
        elif current_state.ready_to_execute:
            yield SetNextNode(next_node="task_execution")
        elif current_state.needs_validation:
            yield SetNextNode(next_node="result_validation")
        else:
            yield SetNextNode(next_node="error_handling")
```

### 3. **Hierarchical Planning with Subagents**

Complex task decomposition using specialized subagents:

```python
class HierarchicalPlanningAgent(Agent):
    """
    Decomposes complex tasks by delegating to specialized subagents.
    """
    subagents = [
        PlannerAgent(),      # High-level planning
        ResearchAgent(),     # Information gathering
        AnalysisAgent(),     # Data analysis
        ExecutionAgent(),    # Task execution
        ReviewAgent()        # Quality assurance
    ]
    
    def delegate(self, tape: TapeType) -> Agent:
        """Choose which subagent should handle current task state."""
        views = self.compute_view(tape)
        # Logic to select appropriate subagent based on current needs
        return self.find_subagent(selected_agent_name)
```

## 📋 Task Decomposition Examples

### 1. **GAIA Agent Planning**

The GAIA agent demonstrates sophisticated planning:

```python
class Plan(Thought):
    """High-level plan for answering complex questions."""
    plan: list[str] = Field(description="List of steps to follow")

class FactsSurvey(Thought):
    """Decompose information needs."""
    given_facts: list[str] = Field(description="Facts already provided")
    facts_to_lookup: list[str] = Field(description="Facts to find online")
    facts_to_derive: list[str] = Field(description="Facts to compute")
    facts_to_guess: list[str] = Field(description="Facts to infer")

# Planning process:
# 1. Analyze question → Generate Plan
# 2. Survey information needs → Generate FactsSurvey  
# 3. Execute information gathering → Use appropriate tools
# 4. Synthesize answer → Combine all information
```

### 2. **Data Science Team Planning**

Multi-agent planning for data science tasks:

```python
def make_world():
    # Specialized agents for different aspects
    coder = TeamAgent.create(
        name="SoftwareEngineer",
        system_prompt="Write code to solve data science problems"
    )
    
    code_executor = TeamAgent.create(
        name="CodeExecutor", 
        execute_code=True
    )
    
    analyst = TeamAgent.create(
        name="AssetReviewer",
        system_prompt="Review and provide feedback on generated assets"
    )
    
    # Team manager coordinates planning
    team = TeamAgent.create_team_manager(
        name="Manager",
        subagents=[coder, code_executor, analyst],
        max_calls=15
    )
    
    # Planning flow:
    # 1. Manager analyzes task
    # 2. Delegates to SoftwareEngineer for code generation
    # 3. Delegates to CodeExecutor for execution
    # 4. Delegates to AssetReviewer for quality check
    # 5. Iterates until task completion
```

### 3. **Web Agent Planning**

Planning for web interaction tasks:

```python
class WebPlanningNode(Node):
    def make_prompt(self, agent, tape):
        """Create prompt that includes web-specific planning context."""
        current_page = self.get_current_page_state(tape)
        available_actions = self.get_available_web_actions(current_page)
        
        return Prompt(messages=[
            {"role": "system", "content": f"""
            You are planning web interactions. Current page: {current_page.url}
            Available actions: {available_actions}
            
            Plan your next steps to accomplish the goal.
            Consider: navigation, form filling, data extraction, etc.
            """},
            *tape_to_messages(tape)
        ])
    
    def generate_steps(self, agent, tape, llm_stream):
        """Generate web-specific action steps."""
        plan_text = llm_stream.get_text()
        
        # Parse plan into executable web actions
        if "click" in plan_text.lower():
            yield ClickElementAction(element_id=self.extract_element_id(plan_text))
        elif "type" in plan_text.lower():
            yield TypeTextAction(text=self.extract_text(plan_text))
        elif "navigate" in plan_text.lower():
            yield OpenUrlAction(url=self.extract_url(plan_text))
        
        # Set next planning step
        yield SetNextNode(next_node="web_execution")
```

## 🔄 Planning Patterns

### 1. **Plan-Execute-Observe-Replan (PEOR)**

```python
class PEORPlanningAgent(Agent):
    """
    Implements Plan-Execute-Observe-Replan pattern.
    """
    nodes = [
        PlanningNode(),      # Generate initial plan
        ExecutionNode(),     # Execute planned actions
        ObservationNode(),   # Observe results
        ReplanningNode()     # Adjust plan based on observations
    ]
    
    def should_replan(self, tape):
        """Determine if replanning is needed based on observations."""
        last_observation = self.get_last_observation(tape)
        return (
            last_observation.indicates_failure() or
            last_observation.suggests_better_approach() or
            last_observation.reveals_new_information()
        )
```

### 2. **Hierarchical Task Network (HTN) Style**

```python
class HTNPlanningNode(Node):
    """
    Hierarchical task decomposition.
    """
    
    def decompose_task(self, task, tape):
        """Break complex task into simpler subtasks."""
        if self.is_primitive_task(task):
            return [task]  # Can't decompose further
        
        # Choose decomposition method based on task type
        if task.type == "data_analysis":
            return [
                Task("load_data", task.data_source),
                Task("clean_data", task.cleaning_requirements), 
                Task("analyze_data", task.analysis_type),
                Task("visualize_results", task.output_format)
            ]
        elif task.type == "web_research":
            return [
                Task("formulate_search_query", task.topic),
                Task("search_web", task.search_terms),
                Task("extract_information", task.extraction_criteria),
                Task("synthesize_findings", task.output_requirements)
            ]
        
        return self.default_decomposition(task)
    
    def generate_steps(self, agent, tape, llm_stream):
        """Generate steps using HTN decomposition."""
        current_task = self.get_current_task(tape)
        subtasks = self.decompose_task(current_task, tape)
        
        for subtask in subtasks:
            if self.is_primitive_task(subtask):
                yield self.create_action_step(subtask)
            else:
                # Further decomposition needed
                yield SetNextNode(next_node="task_decomposition")
                yield Thought(content=f"Need to decompose: {subtask}")
```

### 3. **Goal-Oriented Action Planning (GOAP)**

```python
class GOAPPlanningNode(Node):
    """
    Goal-oriented planning that works backwards from desired state.
    """
    
    def plan_actions(self, current_state, goal_state):
        """Plan sequence of actions to reach goal from current state."""
        if current_state.satisfies(goal_state):
            return []  # Already at goal
        
        # Find actions that can achieve goal conditions
        possible_actions = self.get_actions_for_goal(goal_state)
        
        for action in possible_actions:
            # Check if we can satisfy action's preconditions
            preconditions = action.get_preconditions()
            subplan = self.plan_actions(current_state, preconditions)
            
            if subplan is not None:
                return subplan + [action]
        
        return None  # No plan found
    
    def generate_steps(self, agent, tape, llm_stream):
        """Generate steps using GOAP planning."""
        current_state = self.extract_world_state(tape)
        goal_state = self.extract_goal_state(tape)
        
        action_plan = self.plan_actions(current_state, goal_state)
        
        if action_plan:
            # Execute first action in plan
            next_action = action_plan[0]
            yield self.create_action_step(next_action)
            
            # Store remaining plan for future steps
            yield Thought(content=f"Remaining plan: {action_plan[1:]}")
        else:
            yield Thought(content="No plan found to reach goal")
            yield SetNextNode(next_node="error_handling")
```

## 🛠️ Implementation Patterns

### 1. **Prompt-Based Planning**

```python
class PromptBasedPlanningNode(Node):
    """Use LLM for flexible planning through prompts."""
    
    def make_prompt(self, agent, tape):
        task_context = self.extract_task_context(tape)
        available_tools = self.get_available_tools(agent)
        
        return Prompt(messages=[
            {"role": "system", "content": f"""
            You are a task planning assistant. Given the current situation,
            create a step-by-step plan to accomplish the goal.
            
            Available tools: {available_tools}
            Current context: {task_context}
            
            Format your plan as:
            1. [Action] Description
            2. [Action] Description
            ...
            
            Consider dependencies, prerequisites, and potential failure points.
            """},
            *tape_to_messages(tape)
        ])
    
    def generate_steps(self, agent, tape, llm_stream):
        """Parse LLM-generated plan into executable steps."""
        plan_text = llm_stream.get_text()
        
        # Extract structured plan from LLM output
        plan_steps = self.parse_plan_text(plan_text)
        
        # Convert plan into thought and action steps
        yield Thought(content=f"Generated plan: {plan_text}")
        
        for i, step in enumerate(plan_steps):
            if i == 0:
                # Execute first step immediately
                yield self.create_action_from_plan_step(step)
            else:
                # Store remaining steps for later
                yield Thought(content=f"Planned step {i+1}: {step}")
        
        yield SetNextNode(next_node="plan_execution")
```

### 2. **State-Machine Planning**

```python
class StateMachinePlanningAgent(Agent):
    """
    Planning using explicit state machine.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_machine = {
            "initial": {
                "analyze_task": "planning",
                "error": "error_handling"
            },
            "planning": {
                "plan_ready": "execution", 
                "need_more_info": "information_gathering",
                "error": "error_handling"
            },
            "execution": {
                "action_success": "validation",
                "action_failure": "replanning", 
                "need_more_actions": "execution"
            },
            "validation": {
                "task_complete": "final",
                "task_incomplete": "replanning"
            },
            "replanning": {
                "new_plan_ready": "execution",
                "give_up": "final"
            }
        }
    
    def select_node(self, tape):
        """Select node based on current state machine state."""
        current_state = self.get_current_state(tape)
        transition_event = self.get_last_event(tape)
        
        if transition_event in self.state_machine[current_state]:
            next_state = self.state_machine[current_state][transition_event]
            return self.find_node(f"{next_state}_node")
        
        return self.find_node("error_handling_node")
```

### 3. **Memory-Augmented Planning**

```python
class MemoryAugmentedPlanningNode(Node):
    """
    Planning that leverages memory of past successful plans.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plan_memory = PlanMemory()
    
    def make_prompt(self, agent, tape):
        current_task = self.extract_task(tape)
        
        # Retrieve similar past plans
        similar_plans = self.plan_memory.retrieve_similar_plans(current_task)
        
        context = f"""
        Current task: {current_task}
        
        Similar past successful plans:
        {self.format_similar_plans(similar_plans)}
        
        Create a plan for the current task, learning from past successes.
        """
        
        return Prompt(messages=[
            {"role": "system", "content": context},
            *tape_to_messages(tape)
        ])
    
    def generate_steps(self, agent, tape, llm_stream):
        """Generate steps and store successful plans in memory."""
        plan = self.parse_plan(llm_stream.get_text())
        
        # Store plan for future reference
        task = self.extract_task(tape)
        self.plan_memory.store_plan(task, plan)
        
        yield Thought(content=f"Generated plan: {plan}")
        yield self.create_first_action(plan)
        yield SetNextNode(next_node="plan_execution")
```

## 🎯 Best Practices

### 1. **Modular Planning**
- Break planning into specialized nodes
- Each node handles one aspect of planning
- Use clear interfaces between nodes

### 2. **Context-Aware Planning**
- Always consider current tape state
- Use observations to inform planning decisions
- Adapt plans based on environment feedback

### 3. **Robust Error Handling**
- Plan for failure scenarios
- Include replanning capabilities
- Graceful degradation when plans fail

### 4. **Efficient Resource Usage**
- Cache planning results when appropriate
- Reuse successful plan patterns
- Minimize redundant planning steps

### 5. **Clear Plan Representation**
- Use structured plan formats
- Make plans human-readable
- Include rationale for planning decisions

The TapeAgents orchestration system provides a flexible foundation for implementing sophisticated planning and task decomposition strategies, from simple sequential execution to complex hierarchical planning with multiple specialized agents.