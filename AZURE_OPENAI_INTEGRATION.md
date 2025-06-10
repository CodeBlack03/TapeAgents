# Azure OpenAI Integration with TapeAgents

This document summarizes the Azure OpenAI integration that has been added to TapeAgents, providing comprehensive examples and documentation for using Azure OpenAI with the LiteLLM integration.

## What's Been Added

### 1. Enhanced intro.ipynb Notebook

The main tutorial notebook (`intro.ipynb`) now includes a comprehensive Azure OpenAI section with:

- **Basic Azure OpenAI Setup**: How to configure environment variables and create LiteLLM instances
- **Simple Completion Examples**: Direct usage equivalent to `litellm.completion()`
- **Streaming Examples**: How to use streaming responses with Azure OpenAI
- **TapeAgent Integration**: Complete example of using Azure OpenAI with TapeAgents
- **Configuration Documentation**: Detailed explanation of all parameters and options
- **Direct litellm.completion() Usage**: Alternative approach for users who prefer the direct API

### 2. Standalone Example Script

**File**: `examples/azure_openai_example.py`

A comprehensive standalone Python script demonstrating:
- Basic Azure OpenAI LLM usage
- Streaming responses
- Complete TapeAgent workflows
- Multi-turn conversations
- Error handling and best practices
- Token counting and cost tracking

### 3. Configuration Files

**Files**: 
- `conf/llm/azure_gpt4o.yaml`
- `conf/llm/azure_gpt35_turbo.yaml`

YAML configuration files showing how to set up Azure OpenAI models with different parameters, following the TapeAgents configuration pattern.

### 4. Comprehensive Documentation

**File**: `docs/azure_openai_guide.md`

Complete guide covering:
- Prerequisites and setup
- Environment variable configuration
- Model naming conventions
- Parameter explanations
- Streaming usage
- Error handling and troubleshooting
- Best practices
- Common issues and solutions

### 5. Setup Test Script

**File**: `test_azure_setup.py`

Interactive test script that helps users verify their Azure OpenAI setup:
- Checks environment variables
- Tests LiteLLM import
- Validates Azure OpenAI connection
- Provides specific error guidance
- Confirms working setup

## Key Features Demonstrated

### Environment Variables Setup
```bash
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://your-resource-name.openai.azure.com/"
export AZURE_API_VERSION="2024-02-15-preview"
```

### Basic LiteLLM Usage
```python
from tapeagents.llms import LiteLLM

llm = LiteLLM(
    model_name="azure/your_deployment_name",
    parameters={"temperature": 0.7}
)
```

### TapeAgent Integration
```python
from tapeagents.agent import Agent
from tapeagents.dialog_tape import DialogTape

agent = Agent[DialogTape].create(llm, nodes=[YourNode()])
```

### Streaming Support
```python
llm = LiteLLM(model_name="azure/deployment", stream=True)
for event in llm.generate(prompt):
    if event.chunk:
        print(event.chunk, end="")
```

## Usage Patterns Covered

1. **Direct API Calls**: Using LiteLLM for simple completions
2. **TapeAgent Workflows**: Full agent-based conversations
3. **Streaming Responses**: Real-time response generation
4. **Multi-turn Conversations**: Maintaining conversation context
5. **Configuration Management**: Using YAML files for model setup
6. **Error Handling**: Robust error handling and recovery
7. **Caching**: Response caching for efficiency
8. **Observability**: LLM call logging and monitoring

## Comparison with Direct litellm.completion()

The documentation clearly explains the difference between:

### Direct litellm.completion() (Basic)
```python
import litellm
response = litellm.completion(
    model="azure/deployment_name",
    messages=[{"content": "Hello", "role": "user"}]
)
```

### TapeAgents LiteLLM (Full Integration)
```python
from tapeagents.llms import LiteLLM
llm = LiteLLM(model_name="azure/deployment_name")
response = llm.generate(prompt)
```

The TapeAgents approach provides:
- Automatic caching
- Response logging
- Token counting
- Cost tracking
- Integration with TapeAgent workflows
- Error handling with retries
- Observability features

## Files Modified/Created

### Modified
- `intro.ipynb` - Added comprehensive Azure OpenAI sections

### Created
- `examples/azure_openai_example.py` - Standalone example script
- `conf/llm/azure_gpt4o.yaml` - GPT-4o configuration
- `conf/llm/azure_gpt35_turbo.yaml` - GPT-3.5 Turbo configuration  
- `docs/azure_openai_guide.md` - Complete documentation guide
- `test_azure_setup.py` - Setup verification script
- `AZURE_OPENAI_INTEGRATION.md` - This summary document

## Getting Started

1. **Set up Azure OpenAI credentials** (see environment variables above)
2. **Run the test script**: `python test_azure_setup.py`
3. **Try the examples**: Run `python examples/azure_openai_example.py`
4. **Explore the notebook**: Open `intro.ipynb` and run the Azure OpenAI sections
5. **Read the guide**: Check `docs/azure_openai_guide.md` for detailed information

## Benefits for Users

- **Easy Migration**: Simple transition from OpenAI to Azure OpenAI
- **Full Feature Support**: All TapeAgents features work with Azure OpenAI
- **Comprehensive Examples**: Multiple usage patterns demonstrated
- **Production Ready**: Error handling, caching, and observability included
- **Well Documented**: Clear explanations and troubleshooting guides
- **Flexible Configuration**: Support for YAML configs and direct instantiation

This integration makes Azure OpenAI a first-class citizen in the TapeAgents ecosystem, providing users with enterprise-grade AI capabilities while maintaining the full power and flexibility of the TapeAgents framework.