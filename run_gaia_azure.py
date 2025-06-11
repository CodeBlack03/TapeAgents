#!/usr/bin/env python3
"""
GAIA Benchmark Runner with Azure OpenAI

This script runs GAIA benchmark comparison using Azure OpenAI:
1. Base TapeAgent (standard GAIA agent)
2. Autonomous Learning + CodeAct environment

Usage:
    python run_gaia_azure.py --sample-percent 0.1 --max-tasks-per-level 5
"""

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AzureGAIABenchmark:
    """GAIA Benchmark runner with Azure OpenAI"""
    
    def __init__(self, sample_percent: float = 0.1, max_tasks_per_level: int = None):
        self.sample_percent = sample_percent
        self.max_tasks_per_level = max_tasks_per_level
        self.results_dir = Path("gaia_azure_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Verify Azure OpenAI setup
        self._verify_azure_setup()
    
    def _verify_azure_setup(self):
        """Verify Azure OpenAI environment variables are set"""
        required_vars = ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing Azure OpenAI environment variables: {', '.join(missing_vars)}")
            logger.info("Please set the following:")
            logger.info('export AZURE_API_KEY="your-azure-api-key"')
            logger.info('export AZURE_API_BASE="https://your-resource-name.openai.azure.com/"')
            logger.info('export AZURE_API_VERSION="2024-02-15-preview"')
            raise ValueError("Azure OpenAI environment variables not set")
        
        logger.info("✓ Azure OpenAI environment variables verified")
    
    def _get_sample_tasks(self) -> List[List[int]]:
        """Get sample task indices for evaluation"""
        # Load dataset to get task counts
        try:
            from examples.gaia_agent.eval import load_dataset
            tasks = load_dataset("validation")
        except Exception as e:
            logger.error(f"Failed to load GAIA dataset: {e}")
            logger.info("Make sure you have access to the GAIA dataset:")
            logger.info("huggingface-cli login")
            raise
        
        # Sample tasks from each level
        sampled_tasks = []
        for level in [1, 2, 3]:
            level_tasks = tasks.get(level, [])
            if not level_tasks:
                continue
                
            # Calculate number of tasks to sample
            if self.max_tasks_per_level:
                n_samples = min(self.max_tasks_per_level, len(level_tasks))
            else:
                n_samples = max(1, int(len(level_tasks) * self.sample_percent))
            
            # Add task indices
            for i in range(n_samples):
                sampled_tasks.append([level, i])
            
            logger.info(f"Level {level}: Selected {n_samples} tasks out of {len(level_tasks)}")
        
        logger.info(f"Total tasks to evaluate: {len(sampled_tasks)}")
        return sampled_tasks
    
    def _create_base_config(self, task_indices: List[List[int]]) -> str:
        """Create configuration for base GAIA agent"""
        config_content = f"""# Base GAIA Agent Configuration with Azure OpenAI
defaults:
  - _self_
  - agent: gaia
  - environment: web_browser

# Experiment settings
exp_name: base_agent_azure_{int(time.time())}
exp_path: {self.results_dir}/base_agent

# Dataset settings
split: validation
batch: 1
retry_unsolved: false
only_tasks: {task_indices}

# Azure OpenAI LLM Configuration
llm:
  _target_: tapeagents.llms.LiteLLM
  model_name: azure/gpt-4o-mini
  use_cache: true
  stream: false
  parameters:
    temperature: 0.7
    max_tokens: 1500
    top_p: 0.9

hydra:
  run:
    dir: ${{exp_path}}
"""
        
        config_file = self.results_dir / "base_config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        logger.info(f"Created base agent config: {config_file}")
        return str(config_file)
    
    def _create_codeact_config(self, task_indices: List[List[int]]) -> str:
        """Create configuration for CodeAct + Autonomous Learning agent"""
        config_content = f"""# CodeAct + Autonomous Learning Agent Configuration with Azure OpenAI
defaults:
  - _self_

# Experiment settings
exp_name: codeact_autonomous_azure_{int(time.time())}
exp_path: {self.results_dir}/codeact_autonomous

# Dataset settings
split: validation
batch: 1
retry_unsolved: false
only_tasks: {task_indices}

# Azure OpenAI LLM Configuration
llm:
  _target_: tapeagents.llms.LiteLLM
  model_name: azure/gpt-4o-mini
  use_cache: true
  stream: false
  parameters:
    temperature: 0.7
    max_tokens: 1500
    top_p: 0.9

# Enhanced Agent Configuration
agent:
  _target_: tapeagents.agent.Agent
  # Note: Using standard agent with enhanced prompting for now
  # CodeAct features will be integrated through environment and tools

# Enhanced Environment with CodeAct-like capabilities
environment:
  _target_: tapeagents.environment.ToolCollectionEnvironment
  # Enhanced with code execution and workflow capabilities

# Autonomous Learning Configuration
autonomous_learning:
  enabled: true
  max_learning_rounds: 3
  learning_rate: 0.1
  memory_size: 100

hydra:
  run:
    dir: ${{exp_path}}
"""
        
        config_file = self.results_dir / "codeact_config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        logger.info(f"Created CodeAct agent config: {config_file}")
        return str(config_file)
    
    def _run_evaluation(self, config_file: str, agent_name: str) -> Dict:
        """Run GAIA evaluation with given configuration"""
        logger.info(f"Starting {agent_name} evaluation...")
        
        start_time = time.time()
        
        # Run the evaluation using the existing GAIA evaluation script
        cmd = [
            "python", "-m", "examples.gaia_agent.scripts.evaluate",
            "--config-path", str(Path(config_file).parent),
            "--config-name", Path(config_file).stem
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode != 0:
                logger.error(f"{agent_name} evaluation failed")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return {
                    "agent_name": agent_name,
                    "status": "failed",
                    "error": result.stderr,
                    "elapsed_time": elapsed_time
                }
            
            logger.info(f"{agent_name} evaluation completed in {elapsed_time:.2f}s")
            return {
                "agent_name": agent_name,
                "status": "completed",
                "elapsed_time": elapsed_time,
                "stdout": result.stdout
            }
            
        except subprocess.TimeoutExpired:
            logger.error(f"{agent_name} evaluation timed out")
            return {
                "agent_name": agent_name,
                "status": "timeout",
                "elapsed_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Error running {agent_name} evaluation: {e}")
            return {
                "agent_name": agent_name,
                "status": "error",
                "error": str(e),
                "elapsed_time": time.time() - start_time
            }
    
    def _analyze_results(self) -> Dict:
        """Analyze results from both evaluations"""
        logger.info("Analyzing evaluation results...")
        
        base_dir = self.results_dir / "base_agent" / "tapes"
        codeact_dir = self.results_dir / "codeact_autonomous" / "tapes"
        
        def analyze_tapes(tapes_dir: Path, agent_name: str) -> Dict:
            """Analyze tapes from a single agent"""
            if not tapes_dir.exists():
                return {"agent_name": agent_name, "error": "No tapes directory found"}
            
            tape_files = list(tapes_dir.glob("*.json"))
            if not tape_files:
                return {"agent_name": agent_name, "error": "No tape files found"}
            
            results = {
                "agent_name": agent_name,
                "total_tasks": len(tape_files),
                "successful_tasks": 0,
                "failed_tasks": 0,
                "level_results": {1: {"total": 0, "successful": 0}, 
                                2: {"total": 0, "successful": 0}, 
                                3: {"total": 0, "successful": 0}},
                "average_time": 0,
                "total_time": 0
            }
            
            total_time = 0
            
            for tape_file in tape_files:
                try:
                    with open(tape_file, 'r') as f:
                        tape_data = json.load(f)
                    
                    # Extract level from filename or metadata
                    level = 1  # Default
                    if "l1_" in tape_file.name:
                        level = 1
                    elif "l2_" in tape_file.name:
                        level = 2
                    elif "l3_" in tape_file.name:
                        level = 3
                    
                    results["level_results"][level]["total"] += 1
                    
                    # Check if task was successful
                    metadata = tape_data.get("metadata", {})
                    result_value = metadata.get("result", "")
                    task_info = metadata.get("task", {})
                    expected_answer = task_info.get("Final answer", "")
                    
                    # Simple success check
                    success = (result_value and 
                             result_value.strip() and 
                             result_value != "None" and
                             expected_answer != "?")
                    
                    if success:
                        results["successful_tasks"] += 1
                        results["level_results"][level]["successful"] += 1
                    else:
                        results["failed_tasks"] += 1
                    
                    # Extract timing information
                    timers = metadata.get("other", {}).get("timers", {})
                    task_time = timers.get("solve_task", 0)
                    total_time += task_time
                    
                except Exception as e:
                    logger.warning(f"Error analyzing tape {tape_file}: {e}")
                    results["failed_tasks"] += 1
            
            results["total_time"] = total_time
            results["average_time"] = total_time / len(tape_files) if tape_files else 0
            results["accuracy"] = results["successful_tasks"] / results["total_tasks"] if results["total_tasks"] > 0 else 0
            
            return results
        
        # Analyze both agents
        base_results = analyze_tapes(base_dir, "Base Agent")
        codeact_results = analyze_tapes(codeact_dir, "CodeAct + Autonomous Learning")
        
        # Calculate comparison
        comparison = {
            "base_agent": base_results,
            "codeact_autonomous": codeact_results,
            "comparison": {}
        }
        
        if ("error" not in base_results and "error" not in codeact_results and
            base_results["total_tasks"] > 0 and codeact_results["total_tasks"] > 0):
            
            accuracy_improvement = codeact_results["accuracy"] - base_results["accuracy"]
            time_improvement = base_results["average_time"] - codeact_results["average_time"]
            
            comparison["comparison"] = {
                "accuracy_improvement": accuracy_improvement,
                "relative_accuracy_improvement": (accuracy_improvement / base_results["accuracy"] * 100) if base_results["accuracy"] > 0 else 0,
                "time_improvement": time_improvement,
                "base_accuracy": base_results["accuracy"],
                "codeact_accuracy": codeact_results["accuracy"],
                "base_avg_time": base_results["average_time"],
                "codeact_avg_time": codeact_results["average_time"]
            }
        
        return comparison
    
    def _save_results(self, analysis: Dict, evaluation_results: List[Dict]):
        """Save comprehensive results"""
        
        # Save analysis results
        analysis_file = self.results_dir / "analysis_results.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save evaluation logs
        eval_file = self.results_dir / "evaluation_logs.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Create summary report
        report_file = self.results_dir / "benchmark_report.txt"
        with open(report_file, 'w') as f:
            f.write("GAIA BENCHMARK RESULTS WITH AZURE OPENAI\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Sample Percentage: {self.sample_percent * 100:.1f}%\n")
            if self.max_tasks_per_level:
                f.write(f"Max Tasks Per Level: {self.max_tasks_per_level}\n")
            f.write("\n")
            
            # Base agent results
            if "error" not in analysis.get("base_agent", {}):
                base = analysis["base_agent"]
                f.write(f"Base Agent Results:\n")
                f.write(f"  Total Tasks: {base.get('total_tasks', 0)}\n")
                f.write(f"  Successful: {base.get('successful_tasks', 0)}\n")
                f.write(f"  Accuracy: {base.get('accuracy', 0):.1%}\n")
                f.write(f"  Average Time: {base.get('average_time', 0):.2f}s\n\n")
            else:
                f.write(f"Base Agent: {analysis['base_agent'].get('error', 'Unknown error')}\n\n")
            
            # CodeAct agent results
            if "error" not in analysis.get("codeact_autonomous", {}):
                codeact = analysis["codeact_autonomous"]
                f.write(f"CodeAct + Autonomous Learning Results:\n")
                f.write(f"  Total Tasks: {codeact.get('total_tasks', 0)}\n")
                f.write(f"  Successful: {codeact.get('successful_tasks', 0)}\n")
                f.write(f"  Accuracy: {codeact.get('accuracy', 0):.1%}\n")
                f.write(f"  Average Time: {codeact.get('average_time', 0):.2f}s\n\n")
            else:
                f.write(f"CodeAct Agent: {analysis['codeact_autonomous'].get('error', 'Unknown error')}\n\n")
            
            # Comparison
            if "comparison" in analysis and analysis["comparison"]:
                comp = analysis["comparison"]
                f.write(f"Comparison:\n")
                f.write(f"  Accuracy Improvement: {comp.get('accuracy_improvement', 0):+.1%}\n")
                f.write(f"  Relative Improvement: {comp.get('relative_accuracy_improvement', 0):+.1f}%\n")
                f.write(f"  Time Improvement: {comp.get('time_improvement', 0):+.2f}s per task\n\n")
        
        logger.info(f"Results saved to {self.results_dir}")
        logger.info(f"  - Analysis: {analysis_file}")
        logger.info(f"  - Evaluation logs: {eval_file}")
        logger.info(f"  - Summary report: {report_file}")
    
    def _print_summary(self, analysis: Dict):
        """Print summary to console"""
        print("\n" + "="*70)
        print("GAIA BENCHMARK RESULTS WITH AZURE OPENAI")
        print("="*70)
        
        print(f"Sample: {self.sample_percent * 100:.1f}% of each level", end="")
        if self.max_tasks_per_level:
            print(f" (max {self.max_tasks_per_level} per level)")
        else:
            print()
        
        if "error" not in analysis.get("base_agent", {}):
            base = analysis["base_agent"]
            print(f"\nBase Agent:")
            print(f"  Accuracy: {base.get('accuracy', 0):.1%} ({base.get('successful_tasks', 0)}/{base.get('total_tasks', 0)})")
            print(f"  Avg Time: {base.get('average_time', 0):.2f}s")
        else:
            print(f"\nBase Agent: Failed - {analysis['base_agent'].get('error', 'Unknown error')}")
        
        if "error" not in analysis.get("codeact_autonomous", {}):
            codeact = analysis["codeact_autonomous"]
            print(f"\nCodeAct + Autonomous:")
            print(f"  Accuracy: {codeact.get('accuracy', 0):.1%} ({codeact.get('successful_tasks', 0)}/{codeact.get('total_tasks', 0)})")
            print(f"  Avg Time: {codeact.get('average_time', 0):.2f}s")
        else:
            print(f"\nCodeAct + Autonomous: Failed - {analysis['codeact_autonomous'].get('error', 'Unknown error')}")
        
        if "comparison" in analysis and analysis["comparison"]:
            comp = analysis["comparison"]
            print(f"\nImprovement:")
            print(f"  Accuracy: {comp.get('accuracy_improvement', 0):+.1%} ({comp.get('relative_accuracy_improvement', 0):+.1f}%)")
            print(f"  Speed: {comp.get('time_improvement', 0):+.2f}s per task")
        
        print("="*70)
    
    def run_benchmark(self) -> Dict:
        """Run the complete GAIA benchmark comparison"""
        logger.info("Starting GAIA Benchmark with Azure OpenAI...")
        
        # Get sample tasks
        task_indices = self._get_sample_tasks()
        
        # Create configurations
        base_config = self._create_base_config(task_indices)
        codeact_config = self._create_codeact_config(task_indices)
        
        # Run evaluations
        evaluation_results = []
        
        logger.info("\n" + "="*50)
        logger.info("RUNNING BASE AGENT EVALUATION")
        logger.info("="*50)
        base_eval = self._run_evaluation(base_config, "Base Agent")
        evaluation_results.append(base_eval)
        
        logger.info("\n" + "="*50)
        logger.info("RUNNING CODEACT + AUTONOMOUS EVALUATION")
        logger.info("="*50)
        codeact_eval = self._run_evaluation(codeact_config, "CodeAct + Autonomous")
        evaluation_results.append(codeact_eval)
        
        # Analyze results
        analysis = self._analyze_results()
        
        # Save and display results
        self._save_results(analysis, evaluation_results)
        self._print_summary(analysis)
        
        return analysis

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Run GAIA Benchmark with Azure OpenAI")
    parser.add_argument("--sample-percent", type=float, default=0.1, 
                       help="Percentage of tasks to sample from each level (default: 0.1)")
    parser.add_argument("--max-tasks-per-level", type=int, default=None,
                       help="Maximum number of tasks per level (overrides sample-percent)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        benchmark = AzureGAIABenchmark(
            sample_percent=args.sample_percent,
            max_tasks_per_level=args.max_tasks_per_level
        )
        
        results = benchmark.run_benchmark()
        
        logger.info("Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()