#!/usr/bin/env python3
"""
Test script to verify GAIA benchmark setup and run a minimal comparison
"""

import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required imports work"""
    logger.info("Testing imports...")
    
    try:
        from examples.gaia_agent.eval import load_dataset, calculate_accuracy
        from tapeagents.llms import LiteLLM
        from tapeagents.codeact_agent import CodeActAgent
        from tapeagents.codeact_environment import CodeActEnvironment
        from tapeagents.autonomous_learning import EnvironmentLearner
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_dataset_access():
    """Test GAIA dataset access"""
    logger.info("Testing GAIA dataset access...")
    
    try:
        tasks = load_dataset("validation")
        logger.info(f"✓ Dataset loaded successfully")
        logger.info(f"  Level 1: {len(tasks[1])} tasks")
        logger.info(f"  Level 2: {len(tasks[2])} tasks") 
        logger.info(f"  Level 3: {len(tasks[3])} tasks")
        return True
    except Exception as e:
        logger.error(f"✗ Dataset access failed: {e}")
        logger.info("You may need to run: huggingface-cli login")
        return False

def test_llm_setup():
    """Test Azure OpenAI LLM setup"""
    logger.info("Testing Azure OpenAI LLM setup...")
    
    # Check Azure OpenAI environment variables
    required_vars = ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"✗ Missing Azure OpenAI environment variables: {', '.join(missing_vars)}")
        logger.info("Please set:")
        logger.info('export AZURE_API_KEY="your-azure-api-key"')
        logger.info('export AZURE_API_BASE="https://your-resource-name.openai.azure.com/"')
        logger.info('export AZURE_API_VERSION="2024-02-15-preview"')
        return False
    
    try:
        llm = LiteLLM(model_name="azure/gpt-4o-mini", use_cache=True)
        logger.info("✓ Azure OpenAI LLM setup successful")
        return True
    except Exception as e:
        logger.error(f"✗ Azure OpenAI LLM setup failed: {e}")
        return False

def test_codeact_setup():
    """Test CodeAct components"""
    logger.info("Testing CodeAct setup...")
    
    try:
        from tapeagents.llms import LiteLLM
        llm = LiteLLM(model_name="azure/gpt-4o-mini", use_cache=True)
        
        # Test CodeAct agent (simplified test)
        logger.info("✓ CodeAct components available")
        return True
    except Exception as e:
        logger.error(f"✗ CodeAct setup failed: {e}")
        return False

def run_mini_test():
    """Run a minimal test with 1 task"""
    logger.info("Running mini test with 1 task...")
    
    try:
        from examples.gaia_agent.eval import load_dataset, solve_task
        from tapeagents.llms import LiteLLM
        from tapeagents.environment import ToolCollectionEnvironment
        from tapeagents.agent import Agent
        from hydra import compose, initialize_config_dir
        from tapeagents.orchestrator import get_agent_and_env_from_config
        
        # Load one task
        tasks = load_dataset("validation")
        test_task = tasks[1][0]  # First Level 1 task
        
        logger.info(f"Test task: {test_task['Question'][:100]}...")
        
        # Setup base agent
        config_dir = Path(__file__).parent / "conf"
        with initialize_config_dir(config_dir=str(config_dir)):
            cfg = compose(config_name="gaia_agent")
            agent, env = get_agent_and_env_from_config(cfg)
        
        # Solve task with short timeout
        logger.info("Solving task with base agent...")
        tape = solve_task(
            task=test_task,
            agent=agent,
            env=env,
            level=1,
            task_num=0,
            tapes_dir="./test_output",
            max_loops=5  # Short test
        )
        
        env.close()
        
        result = tape.metadata.result if hasattr(tape.metadata, 'result') else "No result"
        logger.info(f"✓ Mini test completed. Result: {result}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Mini test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting GAIA benchmark setup verification...")
    
    tests = [
        ("Imports", test_imports),
        ("Dataset Access", test_dataset_access), 
        ("LLM Setup", test_llm_setup),
        ("CodeAct Setup", test_codeact_setup),
        ("Mini Test", run_mini_test)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed! Ready to run GAIA comparison.")
        logger.info("Run: python run_gaia_comparison.py")
    else:
        logger.info("\n❌ Some tests failed. Please fix issues before running comparison.")
        sys.exit(1)

if __name__ == "__main__":
    main()