#!/usr/bin/env python3
"""
Azure OpenAI Example with TapeAgents

This example demonstrates how to use Azure OpenAI with TapeAgents using the LiteLLM integration.
It shows both basic LLM usage and full TapeAgent workflows.

Prerequisites:
1. Azure OpenAI resource with a deployed model
2. Environment variables set for Azure OpenAI credentials
3. TapeAgents installed with litellm dependency

Usage:
    python azure_openai_example.py
"""

import os
from tapeagents.agent import Agent, Node
from tapeagents.core import Prompt, SetNextNode
from tapeagents.dialog_tape import AssistantStep, DialogTape, UserStep
from tapeagents.llms import LiteLLM, LLMStream
from tapeagents.prompting import tape_to_messages


def setup_azure_credentials():
    """
    Set up Azure OpenAI credentials.
    Replace these with your actual Azure OpenAI values.
    """
    # Required environment variables for Azure OpenAI
    os.environ["AZURE_API_KEY"] = "your-azure-api-key-here"
    os.environ["AZURE_API_BASE"] = "https://your-resource-name.openai.azure.com/"
    os.environ["AZURE_API_VERSION"] = "2024-02-15-preview"
    
    print("Azure OpenAI credentials configured")
    print(f"API Base: {os.environ.get('AZURE_API_BASE', 'Not set')}")
    print(f"API Version: {os.environ.get('AZURE_API_VERSION', 'Not set')}")


def basic_azure_llm_example():
    """
    Basic example of using Azure OpenAI with LiteLLM directly.
    """
    print("\n=== Basic Azure OpenAI LLM Example ===")
    
    # Create LiteLLM instance for Azure OpenAI
    azure_llm = LiteLLM(
        model_name="azure/your_deployment_name",  # Replace with your deployment name
        parameters={
            "temperature": 0.7,
            "max_tokens": 500
        },
        use_cache=False,
        stream=False
    )
    
    # Create a simple prompt
    prompt = Prompt(messages=[
        {"role": "user", "content": "Explain what Azure OpenAI is in one paragraph."}
    ])
    
    try:
        # Generate response
        llm_stream = azure_llm.generate(prompt)
        response_text = llm_stream.get_text()
        print(f"Response: {response_text}")
        
        # Get token information
        input_tokens = azure_llm.count_tokens(prompt.messages)
        output_tokens = azure_llm.count_tokens(response_text)
        print(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your Azure credentials and deployment name are correct")


def streaming_azure_llm_example():
    """
    Example of streaming responses from Azure OpenAI.
    """
    print("\n=== Streaming Azure OpenAI Example ===")
    
    # Create streaming LiteLLM instance
    azure_llm_streaming = LiteLLM(
        model_name="azure/your_deployment_name",  # Replace with your deployment name
        parameters={"temperature": 0.8},
        stream=True
    )
    
    prompt = Prompt(messages=[
        {"role": "user", "content": "Write a short story about AI and humans working together."}
    ])
    
    try:
        llm_stream = azure_llm_streaming.generate(prompt)
        print("Streaming response:")
        
        # Stream the response chunk by chunk
        for event in llm_stream:
            if event.chunk:
                print(event.chunk, end="", flush=True)
            elif event.output:
                print("\n\nStreaming complete.")
                break
                
    except Exception as e:
        print(f"Streaming error: {e}")


class AzureDialogNode(Node):
    """
    A simple dialog node for Azure OpenAI TapeAgent.
    """
    name: str = "dialog"

    def make_prompt(self, agent: Agent, tape: DialogTape) -> Prompt:
        """Convert tape to messages for the LLM."""
        return Prompt(messages=tape_to_messages(tape))

    def generate_steps(self, agent: Agent, tape: DialogTape, llm_stream: LLMStream):
        """Generate response steps from LLM output."""
        yield AssistantStep(content=llm_stream.get_text())
        yield SetNextNode(next_node="dialog")  # Continue with same node


def azure_tapeagent_example():
    """
    Complete TapeAgent example using Azure OpenAI.
    """
    print("\n=== Azure OpenAI TapeAgent Example ===")
    
    # Create Azure OpenAI LLM
    azure_llm = LiteLLM(
        model_name="azure/your_deployment_name",  # Replace with your deployment name
        parameters={
            "temperature": 0.6,
            "max_tokens": 300
        },
        use_cache=True,  # Enable caching for efficiency
        observe_llm_calls=True  # Enable observability
    )
    
    # Create TapeAgent with Azure LLM
    agent = Agent[DialogTape].create(azure_llm, nodes=[AzureDialogNode()])
    
    # Create initial tape with user message
    start_tape = DialogTape(steps=[
        UserStep(content="What are the benefits of using Azure OpenAI for enterprise applications?")
    ])
    
    try:
        # Run the agent
        print("Running TapeAgent with Azure OpenAI...")
        final_tape = agent.run(start_tape).get_final_tape()
        
        # Display results
        print("\nTape execution completed!")
        print(f"Number of steps: {len(final_tape.steps)}")
        
        # Print conversation
        for i, step in enumerate(final_tape.steps):
            if hasattr(step, 'content'):
                role = step.kind
                content = step.content
                print(f"\n{i+1}. {role.upper()}: {content}")
        
        # Show metadata
        print(f"\nTape metadata:")
        print(f"- ID: {final_tape.metadata.id}")
        print(f"- Author: {final_tape.metadata.author}")
        print(f"- Steps added: {final_tape.metadata.n_added_steps}")
        
    except Exception as e:
        print(f"TapeAgent error: {e}")
        print("Check your Azure OpenAI configuration")


def multi_turn_conversation_example():
    """
    Example of a multi-turn conversation with Azure OpenAI TapeAgent.
    """
    print("\n=== Multi-turn Conversation Example ===")
    
    azure_llm = LiteLLM(
        model_name="azure/your_deployment_name",
        parameters={"temperature": 0.7}
    )
    
    agent = Agent[DialogTape].create(azure_llm, nodes=[AzureDialogNode()])
    
    # Start with initial conversation
    tape = DialogTape(steps=[
        UserStep(content="Hello! Can you help me understand machine learning?")
    ])
    
    try:
        # First turn
        tape = agent.run(tape).get_final_tape()
        print("Turn 1 completed")
        
        # Add follow-up question
        tape.steps.append(UserStep(content="What's the difference between supervised and unsupervised learning?"))
        
        # Second turn
        tape = agent.run(tape).get_final_tape()
        print("Turn 2 completed")
        
        # Add another question
        tape.steps.append(UserStep(content="Can you give me a practical example of each?"))
        
        # Third turn
        tape = agent.run(tape).get_final_tape()
        print("Turn 3 completed")
        
        # Display full conversation
        print("\n=== Full Conversation ===")
        for i, step in enumerate(tape.steps):
            if hasattr(step, 'content'):
                role = "USER" if step.kind == "user" else "ASSISTANT"
                print(f"\n{role}: {step.content}")
                
    except Exception as e:
        print(f"Multi-turn conversation error: {e}")


def main():
    """
    Main function to run all examples.
    """
    print("Azure OpenAI with TapeAgents Examples")
    print("=" * 40)
    
    # Setup credentials
    setup_azure_credentials()
    
    # Run examples
    basic_azure_llm_example()
    streaming_azure_llm_example()
    azure_tapeagent_example()
    multi_turn_conversation_example()
    
    print("\n" + "=" * 40)
    print("Examples completed!")
    print("\nTo use these examples with your Azure OpenAI:")
    print("1. Replace 'your_deployment_name' with your actual deployment name")
    print("2. Set the correct AZURE_API_KEY, AZURE_API_BASE, and AZURE_API_VERSION")
    print("3. Ensure your deployment has sufficient quota")


if __name__ == "__main__":
    main()