#!/usr/bin/env python3
"""
Test script to verify Azure OpenAI setup with TapeAgents.

This script helps you verify that your Azure OpenAI credentials and 
deployment are correctly configured for use with TapeAgents.

Usage:
    python test_azure_setup.py
"""

import os
import sys
from tapeagents.llms import LiteLLM
from tapeagents.core import Prompt


def check_environment_variables():
    """Check if required environment variables are set."""
    print("Checking environment variables...")
    
    required_vars = ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION"]
    missing_vars = []
    
    for var in required_vars:
        value = os.environ.get(var)
        if not value:
            missing_vars.append(var)
            print(f"❌ {var}: Not set")
        else:
            # Mask the API key for security
            if var == "AZURE_API_KEY":
                masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"✅ {var}: {masked_value}")
            else:
                print(f"✅ {var}: {value}")
    
    if missing_vars:
        print(f"\n❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease set these variables:")
        for var in missing_vars:
            if var == "AZURE_API_KEY":
                print(f'export {var}="your-azure-api-key"')
            elif var == "AZURE_API_BASE":
                print(f'export {var}="https://your-resource-name.openai.azure.com/"')
            elif var == "AZURE_API_VERSION":
                print(f'export {var}="2024-02-15-preview"')
        return False
    
    print("✅ All environment variables are set")
    return True


def test_litellm_import():
    """Test if LiteLLM can be imported and used."""
    print("\nTesting LiteLLM import...")
    
    try:
        from tapeagents.llms import LiteLLM
        print("✅ LiteLLM imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import LiteLLM: {e}")
        print("Make sure TapeAgents is installed with litellm dependency")
        return False


def test_azure_connection(deployment_name="gpt-35-turbo"):
    """Test connection to Azure OpenAI."""
    print(f"\nTesting Azure OpenAI connection with deployment: {deployment_name}")
    
    try:
        # Create LiteLLM instance
        llm = LiteLLM(
            model_name=f"azure/{deployment_name}",
            parameters={"temperature": 0.1, "max_tokens": 50}
        )
        print(f"✅ LiteLLM instance created for azure/{deployment_name}")
        
        # Test token counting (doesn't require API call)
        test_text = "Hello, world!"
        try:
            token_count = llm.count_tokens(test_text)
            print(f"✅ Token counting works: '{test_text}' = {token_count} tokens")
        except Exception as e:
            print(f"⚠️  Token counting failed: {e}")
        
        # Test actual API call
        prompt = Prompt(messages=[
            {"role": "user", "content": "Say 'Hello from Azure OpenAI' and nothing else."}
        ])
        
        print("Making test API call...")
        llm_stream = llm.generate(prompt)
        response = llm_stream.get_text()
        
        print(f"✅ API call successful!")
        print(f"Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Azure OpenAI connection failed: {e}")
        
        # Provide specific error guidance
        error_str = str(e).lower()
        if "authentication" in error_str or "unauthorized" in error_str:
            print("💡 This looks like an authentication error. Check your AZURE_API_KEY.")
        elif "not found" in error_str or "404" in error_str:
            print("💡 This looks like a deployment or endpoint error.")
            print(f"   - Check that '{deployment_name}' is your correct deployment name")
            print("   - Verify your AZURE_API_BASE URL is correct")
        elif "quota" in error_str or "rate" in error_str:
            print("💡 This looks like a quota or rate limiting issue.")
            print("   - Check your Azure OpenAI quota in the Azure portal")
        elif "version" in error_str:
            print("💡 This looks like an API version issue.")
            print("   - Try updating AZURE_API_VERSION to a newer version")
        
        return False


def main():
    """Main test function."""
    print("Azure OpenAI Setup Test for TapeAgents")
    print("=" * 40)
    
    # Check environment variables
    if not check_environment_variables():
        sys.exit(1)
    
    # Test LiteLLM import
    if not test_litellm_import():
        sys.exit(1)
    
    # Get deployment name from user
    deployment_name = input("\nEnter your Azure OpenAI deployment name (e.g., gpt-35-turbo, gpt-4o): ").strip()
    
    if not deployment_name:
        print("❌ No deployment name provided")
        sys.exit(1)
    
    # Test Azure connection
    if test_azure_connection(deployment_name):
        print("\n🎉 Azure OpenAI setup is working correctly!")
        print("\nYou can now use Azure OpenAI with TapeAgents:")
        print(f"""
from tapeagents.llms import LiteLLM

llm = LiteLLM(model_name="azure/{deployment_name}")
# Your TapeAgent code here...
""")
    else:
        print("\n❌ Azure OpenAI setup has issues. Please fix the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()