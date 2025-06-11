#!/usr/bin/env python3
"""
GAIA Benchmark Comparison Script

This script compares the performance of different TapeAgent configurations on the GAIA benchmark:
1. Base TapeAgent (standard GAIA agent)
2. Autonomous Learning + CodeAct environment

The script runs on 10% samples of each GAIA level and provides side-by-side comparison.
"""

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from tapeagents.agent import Agent
from tapeagents.autonomous_learning import EnvironmentLearner
from tapeagents.codeact_agent import CodeActAgent
from tapeagents.codeact_environment import CodeActEnvironment
from tapeagents.environment import ToolCollectionEnvironment
from tapeagents.io import save_json_tape
from tapeagents.llms import LiteLLM
from tapeagents.orchestrator import get_agent_and_env_from_config

from examples.gaia_agent.eval import calculate_accuracy, load_dataset, solve_task
from examples.gaia_agent.steps import GaiaTape

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Results for a single benchmark run"""
    agent_type: str
    level: int
    total_tasks: int
    completed_tasks: int
    accuracy: float
    avg_time_per_task: float
    successful_tasks: int
    failed_tasks: int
    error_rate: float
    tapes: List[GaiaTape]

@dataclass
class ComparisonResults:
    """Complete comparison results"""
    base_agent_results: List[BenchmarkResult]
    autonomous_codeact_results: List[BenchmarkResult]
    summary_stats: Dict
    
class GAIABenchmarkComparison:
    """Main class for running GAIA benchmark comparison"""
    
    def __init__(self, sample_percentage: float = 0.1, max_loops: int = 30):
        self.sample_percentage = sample_percentage
        self.max_loops = max_loops
        self.results_dir = Path("gaia_comparison_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Set random seed for reproducible sampling
        random.seed(42)
        
    def sample_tasks(self, tasks: Dict[int, List], percentage: float = 0.1) -> Dict[int, List]:
        """Sample a percentage of tasks from each level"""
        sampled_tasks = {}
        for level, level_tasks in tasks.items():
            n_samples = max(1, int(len(level_tasks) * percentage))
            sampled_tasks[level] = random.sample(level_tasks, n_samples)
            logger.info(f"Level {level}: Sampled {n_samples} tasks out of {len(level_tasks)}")
        return sampled_tasks
    
    def create_base_agent_config(self) -> DictConfig:
        """Create configuration for base GAIA agent"""
        config = {
            "llm": {
                "_target_": "tapeagents.llms.LiteLLM",
                "model_name": "azure/gpt-4o-mini",  # Azure OpenAI model
                "use_cache": True,
                "stream": False,
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            },
            "agent": {
                "_target_": "tapeagents.agent.Agent"
            },
            "environment": {
                "_target_": "tapeagents.environment.ToolCollectionEnvironment"
            }
        }
        return OmegaConf.create(config)
    
    def create_autonomous_codeact_config(self) -> DictConfig:
        """Create configuration for autonomous learning + CodeAct agent"""
        config = {
            "llm": {
                "_target_": "tapeagents.llms.LiteLLM",
                "model_name": "azure/gpt-4o-mini",  # Azure OpenAI model
                "use_cache": True,
                "stream": False,
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            },
            "agent": {
                "_target_": "tapeagents.codeact_agent.CodeActAgent",
                "enable_autonomous_learning": True,
                "enable_workflow_graphs": True,
                "enable_error_localization": True,
                "enable_targeted_reflection": True
            },
            "environment": {
                "_target_": "tapeagents.codeact_environment.CodeActEnvironment",
                "enable_parallel_execution": True,
                "safety_checks": True,
                "sandbox_mode": True
            },
            "autonomous_learning": {
                "environment_learner": {
                    "max_learning_rounds": 3,
                    "tasks_per_round": 2,
                    "optimization_rounds_per_cycle": 2
                }
            }
        }
        return OmegaConf.create(config)
    
    def setup_agent_and_env(self, config: DictConfig, agent_type: str) -> Tuple[Agent, ToolCollectionEnvironment]:
        """Setup agent and environment from configuration"""
        logger.info(f"Setting up {agent_type} agent and environment...")
        
        llm = instantiate(config.llm)
        
        if agent_type == "base":
            # Use standard GAIA agent setup
            with initialize_config_dir(config_dir=str(Path(__file__).parent / "conf")):
                gaia_cfg = compose(config_name="gaia_agent")
                agent, env = get_agent_and_env_from_config(gaia_cfg)
        else:
            # Create CodeAct agent with autonomous learning
            agent = instantiate(config.agent, llm=llm)
            env = instantiate(config.environment)
            
            # Setup autonomous learning if enabled
            if hasattr(config, 'autonomous_learning'):
                learner = EnvironmentLearner(
                    agent=agent,
                    environment=env,
                    **config.autonomous_learning.environment_learner
                )
                # Pre-warm the agent with some learning
                logger.info("Pre-warming agent with autonomous learning...")
                learner.learn_from_environment(num_episodes=2)
        
        return agent, env
    
    def run_single_benchmark(self, 
                           agent_type: str, 
                           tasks: Dict[int, List], 
                           config: DictConfig) -> List[BenchmarkResult]:
        """Run benchmark for a single agent type"""
        logger.info(f"Running benchmark for {agent_type} agent...")
        
        results = []
        agent, env = self.setup_agent_and_env(config, agent_type)
        
        for level, level_tasks in tasks.items():
            logger.info(f"Processing Level {level} with {len(level_tasks)} tasks...")
            
            level_tapes = []
            level_times = []
            completed = 0
            failed = 0
            
            for task_num, task in enumerate(level_tasks):
                try:
                    start_time = time.time()
                    
                    # Create results directory for this run
                    run_dir = self.results_dir / f"{agent_type}_level_{level}"
                    run_dir.mkdir(exist_ok=True)
                    
                    # Solve the task
                    tape = solve_task(
                        task=task,
                        agent=agent,
                        env=env,
                        level=level,
                        task_num=task_num,
                        tapes_dir=str(run_dir),
                        max_loops=self.max_loops
                    )
                    
                    elapsed_time = time.time() - start_time
                    level_times.append(elapsed_time)
                    level_tapes.append(tape)
                    completed += 1
                    
                    # Save individual tape
                    save_json_tape(tape, str(run_dir), f"task_{task_num:03d}")
                    
                    logger.info(f"Task {task_num + 1}/{len(level_tasks)} completed in {elapsed_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Failed to solve task {task_num}: {e}")
                    failed += 1
                    level_times.append(0)  # Record 0 time for failed tasks
            
            # Calculate level statistics
            accuracy, successful = calculate_accuracy(level_tapes)
            avg_time = sum(level_times) / len(level_times) if level_times else 0
            error_rate = failed / len(level_tasks) if level_tasks else 0
            
            result = BenchmarkResult(
                agent_type=agent_type,
                level=level,
                total_tasks=len(level_tasks),
                completed_tasks=completed,
                accuracy=accuracy,
                avg_time_per_task=avg_time,
                successful_tasks=successful,
                failed_tasks=failed,
                error_rate=error_rate,
                tapes=level_tapes
            )
            
            results.append(result)
            logger.info(f"Level {level} completed: {accuracy:.2f}% accuracy, {avg_time:.2f}s avg time")
        
        env.close()
        return results
    
    def run_comparison(self) -> ComparisonResults:
        """Run the complete comparison between base and autonomous+CodeAct agents"""
        logger.info("Starting GAIA benchmark comparison...")
        
        # Load and sample dataset
        logger.info("Loading GAIA dataset...")
        all_tasks = load_dataset("validation")
        sampled_tasks = self.sample_tasks(all_tasks, self.sample_percentage)
        
        # Log sampling statistics
        for level, tasks in sampled_tasks.items():
            logger.info(f"Level {level}: {len(tasks)} tasks sampled")
        
        # Run base agent benchmark
        logger.info("=" * 60)
        logger.info("RUNNING BASE AGENT BENCHMARK")
        logger.info("=" * 60)
        base_config = self.create_base_agent_config()
        base_results = self.run_single_benchmark("base", sampled_tasks, base_config)
        
        # Run autonomous + CodeAct agent benchmark
        logger.info("=" * 60)
        logger.info("RUNNING AUTONOMOUS + CODEACT AGENT BENCHMARK")
        logger.info("=" * 60)
        autonomous_config = self.create_autonomous_codeact_config()
        autonomous_results = self.run_single_benchmark("autonomous_codeact", sampled_tasks, autonomous_config)
        
        # Calculate summary statistics
        summary_stats = self.calculate_summary_stats(base_results, autonomous_results)
        
        return ComparisonResults(
            base_agent_results=base_results,
            autonomous_codeact_results=autonomous_results,
            summary_stats=summary_stats
        )
    
    def calculate_summary_stats(self, 
                              base_results: List[BenchmarkResult], 
                              autonomous_results: List[BenchmarkResult]) -> Dict:
        """Calculate summary statistics for comparison"""
        
        def aggregate_results(results: List[BenchmarkResult]) -> Dict:
            total_tasks = sum(r.total_tasks for r in results)
            total_successful = sum(r.successful_tasks for r in results)
            total_time = sum(r.avg_time_per_task * r.total_tasks for r in results)
            avg_accuracy = sum(r.accuracy for r in results) / len(results) if results else 0
            
            return {
                "total_tasks": total_tasks,
                "total_successful": total_successful,
                "overall_accuracy": (total_successful / total_tasks * 100) if total_tasks > 0 else 0,
                "average_accuracy": avg_accuracy,
                "average_time_per_task": total_time / total_tasks if total_tasks > 0 else 0,
                "total_failed": sum(r.failed_tasks for r in results),
                "average_error_rate": sum(r.error_rate for r in results) / len(results) if results else 0
            }
        
        base_stats = aggregate_results(base_results)
        autonomous_stats = aggregate_results(autonomous_results)
        
        # Calculate improvements
        accuracy_improvement = autonomous_stats["overall_accuracy"] - base_stats["overall_accuracy"]
        time_improvement = base_stats["average_time_per_task"] - autonomous_stats["average_time_per_task"]
        error_rate_improvement = base_stats["average_error_rate"] - autonomous_stats["average_error_rate"]
        
        return {
            "base_agent": base_stats,
            "autonomous_codeact": autonomous_stats,
            "improvements": {
                "accuracy_improvement_percentage": accuracy_improvement,
                "time_improvement_seconds": time_improvement,
                "error_rate_improvement": error_rate_improvement,
                "relative_accuracy_improvement": (accuracy_improvement / base_stats["overall_accuracy"] * 100) if base_stats["overall_accuracy"] > 0 else 0
            }
        }
    
    def generate_report(self, results: ComparisonResults) -> None:
        """Generate detailed comparison report"""
        logger.info("Generating comparison report...")
        
        # Create detailed results table
        report_data = []
        
        # Add base agent results
        for result in results.base_agent_results:
            report_data.append({
                "Agent Type": "Base Agent",
                "Level": result.level,
                "Total Tasks": result.total_tasks,
                "Successful": result.successful_tasks,
                "Accuracy (%)": f"{result.accuracy:.2f}",
                "Avg Time (s)": f"{result.avg_time_per_task:.2f}",
                "Error Rate (%)": f"{result.error_rate * 100:.2f}"
            })
        
        # Add autonomous + CodeAct results
        for result in results.autonomous_codeact_results:
            report_data.append({
                "Agent Type": "Autonomous + CodeAct",
                "Level": result.level,
                "Total Tasks": result.total_tasks,
                "Successful": result.successful_tasks,
                "Accuracy (%)": f"{result.accuracy:.2f}",
                "Avg Time (s)": f"{result.avg_time_per_task:.2f}",
                "Error Rate (%)": f"{result.error_rate * 100:.2f}"
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(report_data)
        report_file = self.results_dir / "detailed_comparison.csv"
        df.to_csv(report_file, index=False)
        
        # Generate summary report
        summary_file = self.results_dir / "summary_report.txt"
        with open(summary_file, 'w') as f:
            f.write("GAIA BENCHMARK COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Sample Size: {self.sample_percentage * 100:.1f}% of each level\n")
            f.write(f"Max Loops per Task: {self.max_loops}\n\n")
            
            f.write("OVERALL RESULTS:\n")
            f.write("-" * 20 + "\n")
            
            base_stats = results.summary_stats["base_agent"]
            autonomous_stats = results.summary_stats["autonomous_codeact"]
            improvements = results.summary_stats["improvements"]
            
            f.write(f"Base Agent:\n")
            f.write(f"  - Overall Accuracy: {base_stats['overall_accuracy']:.2f}%\n")
            f.write(f"  - Average Time per Task: {base_stats['average_time_per_task']:.2f}s\n")
            f.write(f"  - Error Rate: {base_stats['average_error_rate'] * 100:.2f}%\n\n")
            
            f.write(f"Autonomous + CodeAct Agent:\n")
            f.write(f"  - Overall Accuracy: {autonomous_stats['overall_accuracy']:.2f}%\n")
            f.write(f"  - Average Time per Task: {autonomous_stats['average_time_per_task']:.2f}s\n")
            f.write(f"  - Error Rate: {autonomous_stats['average_error_rate'] * 100:.2f}%\n\n")
            
            f.write("IMPROVEMENTS:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Accuracy Improvement: {improvements['accuracy_improvement_percentage']:+.2f} percentage points\n")
            f.write(f"Relative Accuracy Improvement: {improvements['relative_accuracy_improvement']:+.2f}%\n")
            f.write(f"Time Improvement: {improvements['time_improvement_seconds']:+.2f} seconds per task\n")
            f.write(f"Error Rate Improvement: {improvements['error_rate_improvement']:+.2f} percentage points\n\n")
            
            f.write("LEVEL-BY-LEVEL COMPARISON:\n")
            f.write("-" * 30 + "\n")
            
            for level in [1, 2, 3]:
                base_result = next((r for r in results.base_agent_results if r.level == level), None)
                autonomous_result = next((r for r in results.autonomous_codeact_results if r.level == level), None)
                
                if base_result and autonomous_result:
                    f.write(f"Level {level}:\n")
                    f.write(f"  Base Agent: {base_result.accuracy:.2f}% accuracy, {base_result.avg_time_per_task:.2f}s avg time\n")
                    f.write(f"  Autonomous + CodeAct: {autonomous_result.accuracy:.2f}% accuracy, {autonomous_result.avg_time_per_task:.2f}s avg time\n")
                    f.write(f"  Improvement: {autonomous_result.accuracy - base_result.accuracy:+.2f} percentage points\n\n")
        
        # Save detailed results as JSON
        results_json = {
            "base_agent_results": [
                {
                    "agent_type": r.agent_type,
                    "level": r.level,
                    "total_tasks": r.total_tasks,
                    "completed_tasks": r.completed_tasks,
                    "accuracy": r.accuracy,
                    "avg_time_per_task": r.avg_time_per_task,
                    "successful_tasks": r.successful_tasks,
                    "failed_tasks": r.failed_tasks,
                    "error_rate": r.error_rate
                } for r in results.base_agent_results
            ],
            "autonomous_codeact_results": [
                {
                    "agent_type": r.agent_type,
                    "level": r.level,
                    "total_tasks": r.total_tasks,
                    "completed_tasks": r.completed_tasks,
                    "accuracy": r.accuracy,
                    "avg_time_per_task": r.avg_time_per_task,
                    "successful_tasks": r.successful_tasks,
                    "failed_tasks": r.failed_tasks,
                    "error_rate": r.error_rate
                } for r in results.autonomous_codeact_results
            ],
            "summary_stats": results.summary_stats
        }
        
        json_file = self.results_dir / "results.json"
        with open(json_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"Reports saved to {self.results_dir}")
        logger.info(f"- Detailed comparison: {report_file}")
        logger.info(f"- Summary report: {summary_file}")
        logger.info(f"- JSON results: {json_file}")
    
    def print_summary(self, results: ComparisonResults) -> None:
        """Print summary to console"""
        print("\n" + "=" * 80)
        print("GAIA BENCHMARK COMPARISON SUMMARY")
        print("=" * 80)
        
        base_stats = results.summary_stats["base_agent"]
        autonomous_stats = results.summary_stats["autonomous_codeact"]
        improvements = results.summary_stats["improvements"]
        
        print(f"\nSample Size: {self.sample_percentage * 100:.1f}% of each level")
        print(f"Max Loops per Task: {self.max_loops}")
        
        print(f"\nBASE AGENT RESULTS:")
        print(f"  Overall Accuracy: {base_stats['overall_accuracy']:.2f}%")
        print(f"  Avg Time per Task: {base_stats['average_time_per_task']:.2f}s")
        print(f"  Error Rate: {base_stats['average_error_rate'] * 100:.2f}%")
        
        print(f"\nAUTONOMOUS + CODEACT AGENT RESULTS:")
        print(f"  Overall Accuracy: {autonomous_stats['overall_accuracy']:.2f}%")
        print(f"  Avg Time per Task: {autonomous_stats['average_time_per_task']:.2f}s")
        print(f"  Error Rate: {autonomous_stats['average_error_rate'] * 100:.2f}%")
        
        print(f"\nIMPROVEMENTS:")
        print(f"  Accuracy: {improvements['accuracy_improvement_percentage']:+.2f} percentage points")
        print(f"  Relative Accuracy: {improvements['relative_accuracy_improvement']:+.2f}%")
        print(f"  Time per Task: {improvements['time_improvement_seconds']:+.2f} seconds")
        print(f"  Error Rate: {improvements['error_rate_improvement']:+.2f} percentage points")
        
        print(f"\nLEVEL-BY-LEVEL COMPARISON:")
        for level in [1, 2, 3]:
            base_result = next((r for r in results.base_agent_results if r.level == level), None)
            autonomous_result = next((r for r in results.autonomous_codeact_results if r.level == level), None)
            
            if base_result and autonomous_result:
                improvement = autonomous_result.accuracy - base_result.accuracy
                print(f"  Level {level}: {base_result.accuracy:.1f}% → {autonomous_result.accuracy:.1f}% ({improvement:+.1f}pp)")
        
        print("=" * 80)

def main():
    """Main function to run the benchmark comparison"""
    
    # Check if required Azure OpenAI environment variables are set
    required_azure_vars = ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"]
    missing_vars = [var for var in required_azure_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Please set the following Azure OpenAI environment variables: {', '.join(missing_vars)}")
        logger.info("Example:")
        logger.info('export AZURE_API_KEY="your-azure-api-key"')
        logger.info('export AZURE_API_BASE="https://your-resource-name.openai.azure.com/"')
        logger.info('export AZURE_API_VERSION="2024-02-15-preview"')
        return
    
    # Initialize benchmark
    benchmark = GAIABenchmarkComparison(
        sample_percentage=0.1,  # 10% of each level
        max_loops=30
    )
    
    try:
        # Run comparison
        results = benchmark.run_comparison()
        
        # Generate reports
        benchmark.generate_report(results)
        
        # Print summary
        benchmark.print_summary(results)
        
        logger.info("Benchmark comparison completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmark comparison failed: {e}")
        raise

if __name__ == "__main__":
    main()