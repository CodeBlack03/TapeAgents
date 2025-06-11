#!/usr/bin/env python3
"""
Simplified GAIA Benchmark Comparison Script

This script runs a comparison between base TapeAgent and Autonomous Learning + CodeAct
on 10% samples of GAIA benchmark using the existing evaluation infrastructure.
"""

import json
import logging
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from examples.gaia_agent.eval import calculate_accuracy, load_dataset
from tapeagents.io import load_tapes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleGAIAComparison:
    """Simplified GAIA benchmark comparison using existing infrastructure"""
    
    def __init__(self, sample_percentage: float = 0.1):
        self.sample_percentage = sample_percentage
        self.results_dir = Path("gaia_comparison_simple")
        self.results_dir.mkdir(exist_ok=True)
        
        # Set random seed for reproducible sampling
        random.seed(42)
    
    def sample_tasks_to_config(self, tasks: Dict[int, List], percentage: float = 0.1) -> List[List[int]]:
        """Sample tasks and return in format suitable for GAIA config"""
        sampled_task_indices = []
        
        for level, level_tasks in tasks.items():
            n_samples = max(1, int(len(level_tasks) * percentage))
            sampled_indices = random.sample(range(len(level_tasks)), n_samples)
            
            for idx in sampled_indices:
                sampled_task_indices.append([level, idx])
            
            logger.info(f"Level {level}: Sampled {n_samples} tasks out of {len(level_tasks)}")
        
        return sampled_task_indices
    
    def create_base_config(self, task_indices: List[List[int]]) -> str:
        """Create config file for base GAIA agent"""
        config_content = f"""defaults:
  - _self_
  - llm: azure_gpt4o_mini
  - agent: gaia
  - environment: web_browser

exp_name: base_agent_comparison
exp_path: {self.results_dir}/base_agent_results

split: validation
batch: 1
retry_unsolved: false

only_tasks: {task_indices}

hydra:
  run:
    dir: ${{exp_path}}
"""
        
        config_file = self.results_dir / "base_config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        return str(config_file)
    
    def create_autonomous_codeact_config(self, task_indices: List[List[int]]) -> str:
        """Create config file for autonomous learning + CodeAct agent"""
        config_content = f"""defaults:
  - _self_
  - llm: azure_gpt4o_mini

exp_name: autonomous_codeact_comparison
exp_path: {self.results_dir}/autonomous_codeact_results

split: validation
batch: 1
retry_unsolved: false

only_tasks: {task_indices}

# Custom agent configuration for CodeAct + Autonomous Learning
agent:
  _target_: tapeagents.codeact_agent.CodeActAgent
  enable_autonomous_learning: true
  enable_workflow_graphs: true
  enable_error_localization: true
  enable_targeted_reflection: true

# Custom environment configuration for CodeAct
environment:
  _target_: tapeagents.codeact_environment.CodeActEnvironment
  enable_parallel_execution: false  # Disable for simpler comparison
  safety_checks: true
  sandbox_mode: true

# LLM configuration - Azure OpenAI
llm:
  _target_: tapeagents.llms.LiteLLM
  model_name: azure/gpt-4o-mini
  use_cache: true
  stream: false
  parameters:
    temperature: 0.7
    max_tokens: 1000

hydra:
  run:
    dir: ${{exp_path}}
"""
        
        config_file = self.results_dir / "autonomous_codeact_config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        return str(config_file)
    
    def run_evaluation(self, config_file: str, agent_type: str) -> str:
        """Run GAIA evaluation using the provided config"""
        logger.info(f"Running {agent_type} evaluation...")
        
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
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Evaluation failed for {agent_type}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"Evaluation failed for {agent_type}")
            
            logger.info(f"{agent_type} evaluation completed successfully")
            return result.stdout
            
        except subprocess.TimeoutExpired:
            logger.error(f"Evaluation timed out for {agent_type}")
            raise
        except Exception as e:
            logger.error(f"Error running evaluation for {agent_type}: {e}")
            raise
    
    def analyze_results(self, base_results_dir: str, autonomous_results_dir: str) -> Dict:
        """Analyze and compare results from both evaluations"""
        logger.info("Analyzing results...")
        
        def load_and_analyze_tapes(results_dir: str, agent_type: str) -> Dict:
            """Load tapes and calculate statistics"""
            tapes_dir = Path(results_dir) / "tapes"
            if not tapes_dir.exists():
                logger.warning(f"Tapes directory not found: {tapes_dir}")
                return {"error": "No tapes found"}
            
            try:
                # Load all tapes
                tapes = load_tapes(tape_type=None, tapes_dir=str(tapes_dir), file_extension=".json")
                
                if not tapes:
                    logger.warning(f"No tapes loaded from {tapes_dir}")
                    return {"error": "No tapes loaded"}
                
                # Calculate accuracy
                accuracy, successful_count = calculate_accuracy(tapes)
                
                # Calculate statistics by level
                level_stats = {}
                for tape in tapes:
                    level = getattr(tape.metadata, 'level', 'unknown')
                    if level not in level_stats:
                        level_stats[level] = {"total": 0, "successful": 0, "tapes": []}
                    
                    level_stats[level]["total"] += 1
                    level_stats[level]["tapes"].append(tape)
                    
                    # Check if tape was successful
                    if hasattr(tape.metadata, 'result') and tape.metadata.result:
                        predicted = str(tape.metadata.result)
                        golden = tape.metadata.task.get("Final answer", "")
                        if golden != "?" and predicted.strip():
                            # Simple success check - could be improved with proper scorer
                            level_stats[level]["successful"] += 1
                
                # Calculate level accuracies
                for level in level_stats:
                    if level_stats[level]["total"] > 0:
                        level_accuracy, level_successful = calculate_accuracy(level_stats[level]["tapes"])
                        level_stats[level]["accuracy"] = level_accuracy
                        level_stats[level]["successful"] = level_successful
                
                return {
                    "agent_type": agent_type,
                    "total_tasks": len(tapes),
                    "successful_tasks": successful_count,
                    "overall_accuracy": accuracy,
                    "level_stats": level_stats,
                    "tapes": tapes
                }
                
            except Exception as e:
                logger.error(f"Error analyzing tapes for {agent_type}: {e}")
                return {"error": str(e)}
        
        # Analyze both sets of results
        base_analysis = load_and_analyze_tapes(base_results_dir, "Base Agent")
        autonomous_analysis = load_and_analyze_tapes(autonomous_results_dir, "Autonomous + CodeAct")
        
        # Compare results
        comparison = {
            "base_agent": base_analysis,
            "autonomous_codeact": autonomous_analysis,
            "comparison": {}
        }
        
        if "error" not in base_analysis and "error" not in autonomous_analysis:
            # Calculate improvements
            accuracy_improvement = autonomous_analysis["overall_accuracy"] - base_analysis["overall_accuracy"]
            relative_improvement = (accuracy_improvement / base_analysis["overall_accuracy"] * 100) if base_analysis["overall_accuracy"] > 0 else 0
            
            comparison["comparison"] = {
                "accuracy_improvement": accuracy_improvement,
                "relative_improvement_percent": relative_improvement,
                "base_accuracy": base_analysis["overall_accuracy"],
                "autonomous_accuracy": autonomous_analysis["overall_accuracy"]
            }
            
            # Level-by-level comparison
            level_comparison = {}
            for level in [1, 2, 3]:
                if (level in base_analysis.get("level_stats", {}) and 
                    level in autonomous_analysis.get("level_stats", {})):
                    
                    base_level_acc = base_analysis["level_stats"][level].get("accuracy", 0)
                    auto_level_acc = autonomous_analysis["level_stats"][level].get("accuracy", 0)
                    
                    level_comparison[f"level_{level}"] = {
                        "base_accuracy": base_level_acc,
                        "autonomous_accuracy": auto_level_acc,
                        "improvement": auto_level_acc - base_level_acc
                    }
            
            comparison["level_comparison"] = level_comparison
        
        return comparison
    
    def generate_report(self, analysis: Dict) -> None:
        """Generate comparison report"""
        logger.info("Generating comparison report...")
        
        # Save detailed JSON results
        json_file = self.results_dir / "comparison_results.json"
        with open(json_file, 'w') as f:
            # Remove tapes from JSON to avoid serialization issues
            analysis_copy = analysis.copy()
            if "base_agent" in analysis_copy and "tapes" in analysis_copy["base_agent"]:
                del analysis_copy["base_agent"]["tapes"]
            if "autonomous_codeact" in analysis_copy and "tapes" in analysis_copy["autonomous_codeact"]:
                del analysis_copy["autonomous_codeact"]["tapes"]
            
            json.dump(analysis_copy, f, indent=2)
        
        # Generate text report
        report_file = self.results_dir / "comparison_report.txt"
        with open(report_file, 'w') as f:
            f.write("GAIA BENCHMARK COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Sample Size: {self.sample_percentage * 100:.1f}% of each level\n\n")
            
            if "error" in analysis.get("base_agent", {}):
                f.write(f"Base Agent Error: {analysis['base_agent']['error']}\n")
            else:
                base = analysis["base_agent"]
                f.write(f"Base Agent Results:\n")
                f.write(f"  Total Tasks: {base.get('total_tasks', 0)}\n")
                f.write(f"  Successful: {base.get('successful_tasks', 0)}\n")
                f.write(f"  Overall Accuracy: {base.get('overall_accuracy', 0):.2f}%\n\n")
            
            if "error" in analysis.get("autonomous_codeact", {}):
                f.write(f"Autonomous + CodeAct Agent Error: {analysis['autonomous_codeact']['error']}\n")
            else:
                auto = analysis["autonomous_codeact"]
                f.write(f"Autonomous + CodeAct Agent Results:\n")
                f.write(f"  Total Tasks: {auto.get('total_tasks', 0)}\n")
                f.write(f"  Successful: {auto.get('successful_tasks', 0)}\n")
                f.write(f"  Overall Accuracy: {auto.get('overall_accuracy', 0):.2f}%\n\n")
            
            if "comparison" in analysis and analysis["comparison"]:
                comp = analysis["comparison"]
                f.write(f"Comparison:\n")
                f.write(f"  Accuracy Improvement: {comp.get('accuracy_improvement', 0):+.2f} percentage points\n")
                f.write(f"  Relative Improvement: {comp.get('relative_improvement_percent', 0):+.2f}%\n\n")
            
            if "level_comparison" in analysis:
                f.write("Level-by-Level Comparison:\n")
                for level_key, level_data in analysis["level_comparison"].items():
                    level_num = level_key.split("_")[1]
                    f.write(f"  Level {level_num}:\n")
                    f.write(f"    Base: {level_data['base_accuracy']:.2f}%\n")
                    f.write(f"    Autonomous: {level_data['autonomous_accuracy']:.2f}%\n")
                    f.write(f"    Improvement: {level_data['improvement']:+.2f}pp\n\n")
        
        logger.info(f"Reports saved:")
        logger.info(f"  - JSON: {json_file}")
        logger.info(f"  - Text: {report_file}")
    
    def print_summary(self, analysis: Dict) -> None:
        """Print summary to console"""
        print("\n" + "=" * 60)
        print("GAIA BENCHMARK COMPARISON SUMMARY")
        print("=" * 60)
        
        print(f"Sample Size: {self.sample_percentage * 100:.1f}% of each level\n")
        
        if "error" not in analysis.get("base_agent", {}):
            base = analysis["base_agent"]
            print(f"Base Agent: {base.get('overall_accuracy', 0):.2f}% accuracy ({base.get('successful_tasks', 0)}/{base.get('total_tasks', 0)} tasks)")
        else:
            print(f"Base Agent: Error - {analysis['base_agent']['error']}")
        
        if "error" not in analysis.get("autonomous_codeact", {}):
            auto = analysis["autonomous_codeact"]
            print(f"Autonomous + CodeAct: {auto.get('overall_accuracy', 0):.2f}% accuracy ({auto.get('successful_tasks', 0)}/{auto.get('total_tasks', 0)} tasks)")
        else:
            print(f"Autonomous + CodeAct: Error - {analysis['autonomous_codeact']['error']}")
        
        if "comparison" in analysis and analysis["comparison"]:
            comp = analysis["comparison"]
            print(f"\nImprovement: {comp.get('accuracy_improvement', 0):+.2f} percentage points ({comp.get('relative_improvement_percent', 0):+.2f}%)")
        
        if "level_comparison" in analysis:
            print(f"\nLevel-by-Level:")
            for level_key, level_data in analysis["level_comparison"].items():
                level_num = level_key.split("_")[1]
                print(f"  Level {level_num}: {level_data['base_accuracy']:.1f}% → {level_data['autonomous_accuracy']:.1f}% ({level_data['improvement']:+.1f}pp)")
        
        print("=" * 60)
    
    def run_comparison(self) -> Dict:
        """Run the complete comparison"""
        logger.info("Starting GAIA benchmark comparison...")
        
        # Load and sample dataset
        logger.info("Loading and sampling GAIA dataset...")
        all_tasks = load_dataset("validation")
        sampled_task_indices = self.sample_tasks_to_config(all_tasks, self.sample_percentage)
        
        logger.info(f"Total sampled tasks: {len(sampled_task_indices)}")
        
        # Create config files
        base_config = self.create_base_config(sampled_task_indices)
        autonomous_config = self.create_autonomous_codeact_config(sampled_task_indices)
        
        # Run evaluations
        try:
            logger.info("Running base agent evaluation...")
            self.run_evaluation(base_config, "base_agent")
            
            logger.info("Running autonomous + CodeAct evaluation...")
            self.run_evaluation(autonomous_config, "autonomous_codeact")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Continue with analysis of any partial results
        
        # Analyze results
        base_results_dir = str(self.results_dir / "base_agent_results")
        autonomous_results_dir = str(self.results_dir / "autonomous_codeact_results")
        
        analysis = self.analyze_results(base_results_dir, autonomous_results_dir)
        
        # Generate reports
        self.generate_report(analysis)
        self.print_summary(analysis)
        
        return analysis

def main():
    """Main function"""
    
    # Check Azure OpenAI environment variables
    required_azure_vars = ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"]
    missing_vars = [var for var in required_azure_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Please set the following Azure OpenAI environment variables: {', '.join(missing_vars)}")
        logger.info("Example:")
        logger.info('export AZURE_API_KEY="your-azure-api-key"')
        logger.info('export AZURE_API_BASE="https://your-resource-name.openai.azure.com/"')
        logger.info('export AZURE_API_VERSION="2024-02-15-preview"')
        return
    
    # Run comparison
    comparison = SimpleGAIAComparison(sample_percentage=0.1)
    
    try:
        results = comparison.run_comparison()
        logger.info("Comparison completed successfully!")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise

if __name__ == "__main__":
    main()