#!/usr/bin/env python3
"""
GAIA Benchmark Demo Script

This script demonstrates how to run the GAIA benchmark with Autonomous Learning + CodeAct
using Azure OpenAI. It provides a simple interface to test the system.

Usage:
    python run_gaia_demo.py
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_azure_setup():
    """Check if Azure OpenAI is configured"""
    required_vars = ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing Azure OpenAI environment variables: {', '.join(missing_vars)}")
        logger.info("Please set the following:")
        logger.info('export AZURE_API_KEY="your-azure-api-key"')
        logger.info('export AZURE_API_BASE="https://your-resource-name.openai.azure.com/"')
        logger.info('export AZURE_API_VERSION="2024-02-15-preview"')
        return False
    
    logger.info("✓ Azure OpenAI environment variables are set")
    return True

def check_gaia_access():
    """Check if GAIA dataset is accessible"""
    try:
        from examples.gaia_agent.eval import load_dataset
        tasks = load_dataset("validation")
        total_tasks = sum(len(level_tasks) for level_tasks in tasks.values())
        logger.info(f"✓ GAIA dataset accessible ({total_tasks} validation tasks)")
        return True
    except Exception as e:
        logger.error(f"✗ GAIA dataset access failed: {e}")
        logger.info("Please run: huggingface-cli login")
        return False

def run_demo_test():
    """Run a small demo test"""
    logger.info("Running GAIA Autonomous Learning Demo...")
    
    # Run with 2 tasks per level for quick demo
    cmd = [
        "python", "gaia_autonomous_runner.py",
        "--tasks", "2",
        "--verbose"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info("✓ Demo completed successfully!")
            logger.info("Check gaia_autonomous_results/ for detailed results")
            
            # Show summary from stdout
            lines = result.stdout.split('\n')
            summary_started = False
            for line in lines:
                if "GAIA BENCHMARK - AUTONOMOUS LEARNING RESULTS" in line:
                    summary_started = True
                if summary_started:
                    print(line)
            
            return True
        else:
            logger.error("✗ Demo failed")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("✗ Demo timed out (10 minutes)")
        return False
    except Exception as e:
        logger.error(f"✗ Demo failed: {e}")
        return False

def show_usage_options():
    """Show different usage options"""
    print("\n" + "="*60)
    print("GAIA AUTONOMOUS LEARNING - USAGE OPTIONS")
    print("="*60)
    
    print("""
QUICK TESTS:
  python gaia_autonomous_runner.py --tasks 2        # 2 tasks per level (fastest)
  python gaia_autonomous_runner.py --level 1 --tasks 5  # Level 1 only, 5 tasks

MEDIUM TESTS:
  python gaia_autonomous_runner.py --tasks 5        # 5 tasks per level
  python gaia_autonomous_runner.py --sample-percent 0.05  # 5% of all tasks

COMPREHENSIVE TESTS:
  python gaia_autonomous_runner.py --sample-percent 0.1   # 10% of all tasks
  python gaia_autonomous_runner.py --all-levels --tasks 10  # 10 tasks per level

SPECIFIC CONFIGURATIONS:
  python gaia_autonomous_runner.py --level 3 --tasks 3    # Focus on hardest level
  python gaia_autonomous_runner.py --azure-deployment gpt-4o  # Use different model
  python gaia_autonomous_runner.py --results-dir my_results   # Custom output dir

ADVANCED OPTIONS:
  python run_gaia_autonomous_codeact.py --sample-percent 0.1  # Full implementation
  python run_gaia_autonomous_codeact.py --learning-rounds 5   # More learning rounds
""")

def main():
    """Main demo function"""
    print("="*70)
    print("GAIA BENCHMARK WITH AUTONOMOUS LEARNING + CODEACT")
    print("="*70)
    
    print("\nThis demo shows how to run GAIA benchmark with:")
    print("✓ Autonomous Learning with pattern recognition")
    print("✓ CodeAct-inspired reasoning and execution")
    print("✓ Azure OpenAI integration")
    print("✓ Enhanced prompting and error handling")
    
    # Check prerequisites
    print("\n" + "="*50)
    print("CHECKING PREREQUISITES")
    print("="*50)
    
    azure_ok = check_azure_setup()
    gaia_ok = check_gaia_access()
    
    if not azure_ok:
        print("\n❌ Azure OpenAI not configured. Please set environment variables.")
        return
    
    if not gaia_ok:
        print("\n❌ GAIA dataset not accessible. Please run 'huggingface-cli login'.")
        return
    
    print("\n✅ All prerequisites met!")
    
    # Ask user what to do
    print("\n" + "="*50)
    print("DEMO OPTIONS")
    print("="*50)
    
    print("\n1. Run quick demo (2 tasks per level, ~2 minutes)")
    print("2. Show usage options")
    print("3. Exit")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n" + "="*50)
            print("RUNNING QUICK DEMO")
            print("="*50)
            
            success = run_demo_test()
            
            if success:
                print("\n🎉 Demo completed successfully!")
                print("\nNext steps:")
                print("1. Check gaia_autonomous_results/ for detailed results")
                print("2. Run larger tests with --tasks 5 or --sample-percent 0.1")
                print("3. Compare with baseline GAIA agent results")
            else:
                print("\n❌ Demo failed. Check the error messages above.")
        
        elif choice == "2":
            show_usage_options()
        
        elif choice == "3":
            print("\nExiting demo.")
        
        else:
            print("\nInvalid choice. Exiting.")
    
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo error: {e}")

if __name__ == "__main__":
    main()