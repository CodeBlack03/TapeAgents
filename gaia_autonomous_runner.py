#!/usr/bin/env python3
"""
GAIA Benchmark Runner with Autonomous Learning + CodeAct (Simplified)

This script runs the actual GAIA benchmark from Hugging Face using the existing
TapeAgents infrastructure enhanced with autonomous learning concepts.

Usage:
    python gaia_autonomous_runner.py --sample-percent 0.1
    python gaia_autonomous_runner.py --max-tasks 10 --level 1
    python gaia_autonomous_runner.py --tasks 5 --all-levels
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from pydantic import Field

# TapeAgents imports
from tapeagents.agent import Agent, Node
from tapeagents.core import Prompt, SetNextNode
from tapeagents.dialog_tape import AssistantStep, DialogTape, UserStep
from tapeagents.environment import ToolCollectionEnvironment
from tapeagents.tools.web_search import WebSearch
from tapeagents.tools.code_executor import CodeExecutor
from tapeagents.llms import LiteLLM
from tapeagents.orchestrator import main_loop
from tapeagents.prompting import tape_to_messages
from tapeagents.io import save_json_tape

# GAIA benchmark imports
from examples.gaia_agent.eval import load_dataset, task_to_observations, calculate_accuracy
from examples.gaia_agent.steps import GaiaTape, GaiaMetadata, GaiaAnswer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutonomousLearningNode(Node):
    """Enhanced node with autonomous learning capabilities"""
    
    name: str = "autonomous_learning"
    learning_memory: List = Field(default_factory=list)
    success_patterns: List = Field(default_factory=list)
    failure_patterns: List = Field(default_factory=list)
    task_count: int = 0
    
    def make_prompt(self, agent: Agent, tape: GaiaTape) -> Prompt:
        """Create enhanced prompt with learning context"""
        
        # Base messages from tape
        messages = tape_to_messages(tape)
        
        # Add autonomous learning context
        learning_context = self._create_learning_context()
        
        # Enhanced system prompt with learning
        system_prompt = f"""You are an advanced AI agent with autonomous learning capabilities and CodeAct framework integration.

AUTONOMOUS LEARNING CONTEXT:
{learning_context}

CODEACT FRAMEWORK CAPABILITIES:
- Create executable Python code for complex tasks
- Use workflow dependency graphs for task decomposition
- Implement precise error localization and targeted reflection
- Execute code in a safe sandbox environment

ENHANCED REASONING APPROACH:
1. ANALYZE: Break down the question into sub-components
2. PLAN: Create a step-by-step execution plan with code
3. EXECUTE: Write and run Python code to solve each step
4. VERIFY: Check results and refine if needed
5. LEARN: Update your approach based on success/failure patterns

For this GAIA task, use your enhanced capabilities to provide the most accurate answer possible.
Focus on executable solutions and learn from each interaction."""

        # Insert system prompt at the beginning
        enhanced_messages = [{"role": "system", "content": system_prompt}] + messages
        
        return Prompt(messages=enhanced_messages)
    
    def _create_learning_context(self) -> str:
        """Create learning context from previous experiences"""
        context_parts = []
        
        context_parts.append(f"Tasks completed: {self.task_count}")
        
        if self.success_patterns:
            context_parts.append(f"Successful patterns learned: {len(self.success_patterns)}")
            context_parts.append("Recent successful approaches:")
            for pattern in self.success_patterns[-3:]:  # Last 3 patterns
                context_parts.append(f"- {pattern}")
        
        if self.failure_patterns:
            context_parts.append(f"Failure patterns to avoid: {len(self.failure_patterns)}")
            context_parts.append("Common failure modes:")
            for pattern in self.failure_patterns[-2:]:  # Last 2 patterns
                context_parts.append(f"- {pattern}")
        
        if not context_parts:
            context_parts.append("Starting fresh - no prior learning data")
        
        return "\n".join(context_parts)
    
    def generate_steps(self, agent: Agent, tape: GaiaTape, llm_stream):
        """Generate steps with autonomous learning integration"""
        
        # Get the response
        response_text = llm_stream.get_text()
        
        # Extract answer from response
        answer = self._extract_answer(response_text)
        
        # Create assistant step with enhanced response
        enhanced_response = f"""AUTONOMOUS LEARNING + CODEACT RESPONSE:

{response_text}

EXTRACTED ANSWER: {answer}

LEARNING UPDATE: This response will be analyzed for pattern learning and future improvement."""

        yield AssistantStep(content=enhanced_response)
        
        # Create answer step
        if answer:
            yield GaiaAnswer(answer=answer)
        
        # Update learning based on this interaction
        self._update_learning(tape, response_text, answer)
        
        yield SetNextNode(next_node="autonomous_learning")
    
    def _extract_answer(self, response: str) -> str:
        """Extract the final answer from the response"""
        # Look for common answer patterns
        import re
        
        # Look for "Answer:" or "Final answer:" patterns
        answer_patterns = [
            r"(?:final\s+)?answer\s*:\s*(.+?)(?:\n|$)",
            r"(?:the\s+)?answer\s+is\s*:?\s*(.+?)(?:\n|$)",
            r"result\s*:\s*(.+?)(?:\n|$)",
            r"solution\s*:\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # If no pattern found, try to extract from the last line
        lines = response.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if len(last_line) < 200:  # Reasonable answer length
                return last_line
        
        # Fallback: return first sentence
        sentences = response.split('.')
        if sentences:
            return sentences[0].strip()
        
        return response.strip()[:100]  # Fallback to first 100 chars
    
    def _update_learning(self, tape: GaiaTape, response: str, answer: str):
        """Update learning patterns based on the interaction"""
        self.task_count += 1
        
        # Analyze the task type
        question = ""
        for step in tape.steps:
            if hasattr(step, 'content') and 'Question' in str(step.content):
                question = str(step.content)
                break
        
        # Determine task characteristics
        task_type = self._classify_task(question)
        approach_used = self._analyze_approach(response)
        
        # Store learning data
        learning_entry = {
            "task_count": self.task_count,
            "task_type": task_type,
            "approach": approach_used,
            "answer_length": len(answer),
            "response_length": len(response),
            "timestamp": time.time()
        }
        
        self.learning_memory.append(learning_entry)
        
        # Update patterns (simplified learning)
        if "code" in response.lower() or "python" in response.lower():
            self.success_patterns.append(f"Used code-based approach for {task_type}")
        
        if len(answer) > 0 and len(answer) < 500:
            self.success_patterns.append(f"Provided concise answer for {task_type}")
        
        # Keep only recent patterns
        self.success_patterns = self.success_patterns[-10:]
        self.failure_patterns = self.failure_patterns[-5:]
    
    def _classify_task(self, question: str) -> str:
        """Classify the type of GAIA task"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['calculate', 'compute', 'math', '+', '-', '*', '/']):
            return "mathematical"
        elif any(word in question_lower for word in ['capital', 'country', 'geography']):
            return "geographical"
        elif any(word in question_lower for word in ['search', 'find', 'lookup']):
            return "research"
        elif any(word in question_lower for word in ['analyze', 'compare', 'evaluate']):
            return "analytical"
        else:
            return "general"
    
    def _analyze_approach(self, response: str) -> str:
        """Analyze the approach used in the response"""
        response_lower = response.lower()
        
        approaches = []
        if "python" in response_lower or "code" in response_lower:
            approaches.append("code-based")
        if "search" in response_lower or "research" in response_lower:
            approaches.append("research-based")
        if "calculate" in response_lower or "math" in response_lower:
            approaches.append("mathematical")
        if "analyze" in response_lower or "break down" in response_lower:
            approaches.append("analytical")
        
        return ", ".join(approaches) if approaches else "general"

class GAIAAutonomousRunner:
    """GAIA benchmark runner with autonomous learning"""
    
    def __init__(self, 
                 azure_deployment: str = "gpt-4o-mini",
                 results_dir: str = "gaia_autonomous_results"):
        
        self.azure_deployment = azure_deployment
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Verify Azure setup
        self._verify_azure_setup()
        
        # Create enhanced LLM
        self.llm = self._create_llm()
        
        # Create autonomous learning agent
        self.agent = self._create_agent()
        
        # Create enhanced environment
        self.environment = self._create_environment()
        
        # Statistics
        self.stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_time": 0,
            "level_stats": {1: {"total": 0, "successful": 0}, 
                           2: {"total": 0, "successful": 0}, 
                           3: {"total": 0, "successful": 0}}
        }
    
    def _verify_azure_setup(self):
        """Verify Azure OpenAI setup"""
        required_vars = ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing Azure OpenAI environment variables: {', '.join(missing_vars)}")
        
        logger.info("✓ Azure OpenAI environment verified")
    
    def _create_llm(self) -> LiteLLM:
        """Create Azure OpenAI LLM"""
        return LiteLLM(
            model_name=f"azure/{self.azure_deployment}",
            use_cache=True,
            stream=False,
            parameters={
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 0.9
            }
        )
    
    def _create_agent(self) -> Agent:
        """Create agent with autonomous learning node"""
        autonomous_node = AutonomousLearningNode()
        return Agent[GaiaTape].create(self.llm, nodes=[autonomous_node])
    
    def _create_environment(self) -> ToolCollectionEnvironment:
        """Create enhanced environment with GAIA tools"""
        tools = [
            WebSearch(),
            CodeExecutor(exp_path=str(self.results_dir))
        ]
        return ToolCollectionEnvironment(tools=tools)
    
    def _load_gaia_tasks(self, 
                        levels: List[int] = [1, 2, 3],
                        sample_percent: Optional[float] = None,
                        max_tasks: Optional[int] = None) -> Dict[int, List]:
        """Load GAIA tasks"""
        logger.info("Loading GAIA dataset from Hugging Face...")
        
        try:
            all_tasks = load_dataset("validation")
        except Exception as e:
            logger.error(f"Failed to load GAIA dataset: {e}")
            logger.info("Make sure you have access: huggingface-cli login")
            raise
        
        # Filter and sample tasks
        final_tasks = {}
        for level in levels:
            level_tasks = all_tasks.get(level, [])
            if not level_tasks:
                continue
            
            # Apply max tasks limit
            if max_tasks:
                level_tasks = level_tasks[:max_tasks]
            
            # Apply percentage sampling
            if sample_percent:
                n_samples = max(1, int(len(level_tasks) * sample_percent))
                random.seed(42)
                level_tasks = random.sample(level_tasks, n_samples)
            
            final_tasks[level] = level_tasks
            logger.info(f"Level {level}: Selected {len(level_tasks)} tasks")
        
        return final_tasks
    
    def _solve_task(self, task: Dict, level: int, task_num: int) -> GaiaTape:
        """Solve a single GAIA task"""
        logger.info(f"Solving Level {level} Task {task_num}: {task['Question'][:80]}...")
        
        start_time = time.time()
        
        try:
            # Create initial tape with task
            start_steps = task_to_observations(task)
            tape = GaiaTape(steps=start_steps)
            tape.metadata = GaiaMetadata(task=task, level=level)
            
            # Run the agent
            max_loops = 10  # Limit loops for efficiency
            final_tape = None
            
            for event in main_loop(self.agent, tape, self.environment, max_loops=max_loops):
                if event.agent_tape:
                    final_tape = event.agent_tape
                    break
            
            if not final_tape:
                final_tape = tape
            
            # Extract result
            result = ""
            for step in reversed(final_tape.steps):
                if isinstance(step, GaiaAnswer):
                    result = step.answer
                    break
                elif hasattr(step, 'content') and step.kind == "assistant":
                    # Try to extract answer from assistant response
                    result = self._extract_answer_from_content(step.content)
                    break
            
            # Update metadata
            final_tape.metadata.result = result
            final_tape.metadata.level = level
            
            # Validate answer
            expected = task.get("Final answer", "")
            is_correct = self._validate_answer(result, expected)
            
            # Update statistics
            elapsed_time = time.time() - start_time
            self._update_stats(level, is_correct, elapsed_time)
            
            # Log result
            status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
            logger.info(f"{status}: '{result}' (expected: '{expected}') [{elapsed_time:.1f}s]")
            
            # Save tape
            tape_file = self.results_dir / f"level_{level}_task_{task_num:03d}.json"
            save_json_tape(final_tape, str(tape_file))
            
            return final_tape
            
        except Exception as e:
            logger.error(f"Error solving task: {e}")
            # Create error tape
            error_tape = GaiaTape(steps=task_to_observations(task))
            error_tape.metadata = GaiaMetadata(task=task, level=level, error=str(e), result="")
            self._update_stats(level, False, time.time() - start_time)
            return error_tape
    
    def _extract_answer_from_content(self, content: str) -> str:
        """Extract answer from assistant content"""
        import re
        
        # Look for extracted answer
        if "EXTRACTED ANSWER:" in content:
            lines = content.split('\n')
            for line in lines:
                if "EXTRACTED ANSWER:" in line:
                    return line.split("EXTRACTED ANSWER:")[-1].strip()
        
        # Fallback to pattern matching
        answer_patterns = [
            r"(?:final\s+)?answer\s*:\s*(.+?)(?:\n|$)",
            r"(?:the\s+)?answer\s+is\s*:?\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _validate_answer(self, predicted: str, expected: str) -> bool:
        """Validate answer"""
        if not predicted or not expected or expected == "?":
            return False
        
        predicted_clean = predicted.strip().lower()
        expected_clean = expected.strip().lower()
        
        # Exact match
        if predicted_clean == expected_clean:
            return True
        
        # Partial match
        if len(expected_clean) > 5 and expected_clean in predicted_clean:
            return True
        
        # Numerical tolerance
        try:
            pred_num = float(predicted_clean)
            exp_num = float(expected_clean)
            return abs(pred_num - exp_num) < 0.01
        except ValueError:
            pass
        
        return False
    
    def _update_stats(self, level: int, is_correct: bool, elapsed_time: float):
        """Update statistics"""
        self.stats["total_tasks"] += 1
        self.stats["level_stats"][level]["total"] += 1
        self.stats["total_time"] += elapsed_time
        
        if is_correct:
            self.stats["successful_tasks"] += 1
            self.stats["level_stats"][level]["successful"] += 1
        else:
            self.stats["failed_tasks"] += 1
    
    def run_benchmark(self, 
                     levels: List[int] = [1, 2, 3],
                     sample_percent: Optional[float] = None,
                     max_tasks: Optional[int] = None) -> Dict:
        """Run the benchmark"""
        
        logger.info("Starting GAIA Benchmark with Autonomous Learning")
        logger.info("=" * 60)
        
        # Load tasks
        tasks_by_level = self._load_gaia_tasks(levels, sample_percent, max_tasks)
        
        # Process tasks
        all_tapes = []
        
        for level in sorted(levels):
            level_tasks = tasks_by_level.get(level, [])
            if not level_tasks:
                continue
            
            logger.info(f"\nProcessing Level {level} ({len(level_tasks)} tasks)")
            logger.info("-" * 40)
            
            for task_num, task in enumerate(tqdm(level_tasks, desc=f"Level {level}")):
                tape = self._solve_task(task, level, task_num)
                all_tapes.append(tape)
        
        # Generate results
        results = self._generate_results()
        
        # Save results
        self._save_results(results, all_tapes)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _generate_results(self) -> Dict:
        """Generate results summary"""
        overall_accuracy = (self.stats["successful_tasks"] / self.stats["total_tasks"] * 100 
                          if self.stats["total_tasks"] > 0 else 0)
        
        avg_time = (self.stats["total_time"] / self.stats["total_tasks"] 
                   if self.stats["total_tasks"] > 0 else 0)
        
        level_analysis = {}
        for level in [1, 2, 3]:
            level_stats = self.stats["level_stats"][level]
            if level_stats["total"] > 0:
                level_analysis[level] = {
                    "total_tasks": level_stats["total"],
                    "successful_tasks": level_stats["successful"],
                    "accuracy": level_stats["successful"] / level_stats["total"] * 100
                }
        
        return {
            "overall_performance": {
                "total_tasks": self.stats["total_tasks"],
                "successful_tasks": self.stats["successful_tasks"],
                "overall_accuracy": overall_accuracy,
                "average_time_per_task": avg_time,
                "total_time": self.stats["total_time"]
            },
            "level_analysis": level_analysis,
            "system_info": {
                "azure_deployment": self.azure_deployment,
                "autonomous_learning": True,
                "enhanced_prompting": True
            }
        }
    
    def _save_results(self, results: Dict, all_tapes: List[GaiaTape]):
        """Save results"""
        # JSON results
        results_file = self.results_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # CSV details
        csv_data = []
        for tape in all_tapes:
            csv_data.append({
                "level": tape.metadata.level,
                "question": tape.metadata.task.get("Question", "")[:100],
                "expected": tape.metadata.task.get("Final answer", ""),
                "predicted": tape.metadata.result,
                "correct": self._validate_answer(tape.metadata.result, tape.metadata.task.get("Final answer", "")),
                "error": tape.metadata.error or ""
            })
        
        df = pd.DataFrame(csv_data)
        csv_file = self.results_dir / "detailed_results.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {self.results_dir}")
    
    def _print_summary(self, results: Dict):
        """Print summary"""
        print("\n" + "=" * 60)
        print("GAIA BENCHMARK - AUTONOMOUS LEARNING RESULTS")
        print("=" * 60)
        
        overall = results["overall_performance"]
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Tasks: {overall['total_tasks']}")
        print(f"  Successful: {overall['successful_tasks']}")
        print(f"  Accuracy: {overall['overall_accuracy']:.2f}%")
        print(f"  Average Time: {overall['average_time_per_task']:.2f}s")
        
        print(f"\nLEVEL BREAKDOWN:")
        for level, analysis in results["level_analysis"].items():
            print(f"  Level {level}: {analysis['accuracy']:.1f}% "
                  f"({analysis['successful_tasks']}/{analysis['total_tasks']})")
        
        print(f"\nFEATURES USED:")
        print(f"  ✓ Autonomous Learning with pattern recognition")
        print(f"  ✓ Enhanced prompting with learning context")
        print(f"  ✓ CodeAct-inspired reasoning approach")
        print(f"  ✓ Azure OpenAI integration")
        
        print("=" * 60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="GAIA Benchmark with Autonomous Learning")
    
    parser.add_argument("--levels", type=str, default="1,2,3",
                       help="Levels to test (default: 1,2,3)")
    parser.add_argument("--sample-percent", type=float, default=None,
                       help="Percentage of tasks to sample")
    parser.add_argument("--max-tasks", type=int, default=None,
                       help="Maximum tasks per level")
    parser.add_argument("--tasks", type=int, default=None,
                       help="Alias for --max-tasks")
    parser.add_argument("--level", type=int, default=None,
                       help="Single level to test")
    parser.add_argument("--all-levels", action="store_true",
                       help="Test all levels")
    parser.add_argument("--azure-deployment", type=str, default="gpt-4o-mini",
                       help="Azure deployment name")
    parser.add_argument("--results-dir", type=str, default="gaia_autonomous_results",
                       help="Results directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse levels
    if args.level:
        levels = [args.level]
    elif args.all_levels:
        levels = [1, 2, 3]
    else:
        levels = [int(x.strip()) for x in args.levels.split(",")]
    
    # Handle tasks parameter
    max_tasks = args.max_tasks or args.tasks
    
    try:
        runner = GAIAAutonomousRunner(
            azure_deployment=args.azure_deployment,
            results_dir=args.results_dir
        )
        
        results = runner.run_benchmark(
            levels=levels,
            sample_percent=args.sample_percent,
            max_tasks=max_tasks
        )
        
        logger.info("Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()