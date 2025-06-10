# Azure OpenAI with TapeAgents

This guide explains how to use Azure OpenAI with TapeAgents through the LiteLLM integration.

## Prerequisites

1. **Azure OpenAI Resource**: You need an Azure OpenAI resource with at least one deployed model
2. **API Credentials**: Access to your Azure OpenAI API key, endpoint, and API version
3. **TapeAgents Installation**: TapeAgents with the `litellm` dependency

## Quick Start

### 1. Set Environment Variables

```bash
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://your-resource-name.openai.azure.com/"
export AZURE_API_VERSION="2024-02-15-preview"
```

### 2. Basic Usage

```python
from tapeagents.llms import LiteLLM
from tapeagents.core import Prompt

# Create Azure OpenAI LLM instance
llm = LiteLLM(
    model_name="azure/your_deployment_name",  # Replace with your deployment
    parameters={"temperature": 0.7}
)

# Make a completion call
prompt = Prompt(messages=[
    {"role": "user", "content": "Hello, how are you?"}
])

response = llm.generate(prompt)
print(response.get_text())
```

### 3. TapeAgent with Azure OpenAI

```python
from tapeagents.agent import Agent, Node
from tapeagents.dialog_tape import DialogTape, UserStep, AssistantStep
from tapeagents.core import SetNextNode
from tapeagents.prompting import tape_to_messages

class SimpleNode(Node):
    name: str = "main"
    
    def make_prompt(self, agent, tape):
        return Prompt(messages=tape_to_messages(tape))
    
    def generate_steps(self, agent, tape, llm_stream):
        yield AssistantStep(content=llm_stream.get_text())
        yield SetNextNode(next_node="main")

# Create agent
agent = Agent[DialogTape].create(llm, nodes=[SimpleNode()])

# Run agent
tape = DialogTape(steps=[UserStep(content="Tell me about AI")])
final_tape = agent.run(tape).get_final_tape()
```

## Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_API_KEY` | Your Azure OpenAI API key | `abc123...` |
| `AZURE_API_BASE` | Your Azure OpenAI endpoint | `https://myresource.openai.azure.com/` |
| `AZURE_API_VERSION` | API version to use | `2024-02-15-preview` |

### Model Names

Azure OpenAI model names in TapeAgents must follow the format: `azure/<deployment_name>`

Common examples:
- `azure/gpt-4o` - GPT-4o deployment
- `azure/gpt-4o-mini` - GPT-4o Mini deployment  
- `azure/gpt-35-turbo` - GPT-3.5 Turbo deployment
- `azure/gpt-4` - GPT-4 deployment

**Important**: Use your actual deployment name from Azure OpenAI Studio, not the base model name.

### LiteLLM Parameters

```python
llm = LiteLLM(
    model_name="azure/your_deployment_name",
    parameters={
        "temperature": 0.7,        # Randomness (0.0 to 2.0)
        "max_tokens": 1000,        # Maximum response length
        "top_p": 0.95,            # Nucleus sampling
        "frequency_penalty": 0.0,  # Reduce repetition
        "presence_penalty": 0.0,   # Encourage new topics
    },
    use_cache=False,              # Enable response caching
    stream=False,                 # Enable streaming responses
    context_size=128000,          # Maximum context window
    observe_llm_calls=True        # Enable observability
)
```

## Configuration Files

You can use YAML configuration files for different Azure OpenAI setups:

### conf/llm/azure_gpt4o.yaml
```yaml
_target_: tapeagents.llms.LiteLLM
model_name: azure/gpt-4o
use_cache: false
context_size: 128000
parameters:
  temperature: 0.7
  max_tokens: 2000
```

### Usage with Config
```python
from hydra import compose, initialize
from hydra.utils import instantiate

with initialize(config_path="../conf"):
    cfg = compose(config_name="llm/azure_gpt4o")
    llm = instantiate(cfg)
```

## Streaming Responses

```python
# Enable streaming
llm = LiteLLM(
    model_name="azure/your_deployment_name",
    stream=True
)

# Stream response
llm_stream = llm.generate(prompt)
for event in llm_stream:
    if event.chunk:
        print(event.chunk, end="", flush=True)
    elif event.output:
        break
```

## Error Handling

Common issues and solutions:

### Authentication Errors
```python
try:
    response = llm.generate(prompt)
except Exception as e:
    if "authentication" in str(e).lower():
        print("Check your AZURE_API_KEY")
    elif "not found" in str(e).lower():
        print("Check your deployment name and AZURE_API_BASE")
```

### Rate Limiting
TapeAgents automatically handles rate limiting with exponential backoff:

```python
llm = LiteLLM(
    model_name="azure/your_deployment_name",
    # Rate limiting is handled automatically
)
```

### Quota Exceeded
Monitor your Azure OpenAI quota in the Azure portal. TapeAgents will raise an exception when quota is exceeded.

## Best Practices

1. **Use Caching**: Enable `use_cache=True` for development to avoid redundant API calls
2. **Monitor Costs**: Use `llm.get_token_costs()` to track token usage
3. **Set Context Limits**: Configure `context_size` based on your model's limits
4. **Handle Errors**: Always wrap LLM calls in try-catch blocks
5. **Use Configuration Files**: Store model configurations in YAML files for easy management

## Examples

See the following files for complete examples:
- `examples/azure_openai_example.py` - Comprehensive Azure OpenAI examples
- `intro.ipynb` - Jupyter notebook with Azure OpenAI sections
- `conf/llm/azure_*.yaml` - Configuration file examples

## Troubleshooting

### Common Issues

1. **"Model not found"**: Check your deployment name matches exactly
2. **"Invalid API key"**: Verify your `AZURE_API_KEY` is correct
3. **"Endpoint not found"**: Ensure `AZURE_API_BASE` includes the full URL with protocol
4. **"API version not supported"**: Update `AZURE_API_VERSION` to a supported version

### Debug Mode

Enable debug logging to see detailed API calls:

```python
import logging
logging.getLogger("LiteLLM").setLevel(logging.DEBUG)
```

### Verify Configuration

```python
# Test your configuration
print(f"API Base: {os.environ.get('AZURE_API_BASE')}")
print(f"API Version: {os.environ.get('AZURE_API_VERSION')}")
print(f"Model: {llm.model_name}")

# Test token counting
tokens = llm.count_tokens("Hello world")
print(f"Token count test: {tokens}")
```

## Support

For Azure OpenAI specific issues:
- Check the [Azure OpenAI documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/)
- Review your Azure OpenAI resource in the Azure portal
- Verify your deployment status in Azure OpenAI Studio

For TapeAgents issues:
- Check the main TapeAgents documentation
- Review the examples in the `examples/` directory
- Open an issue on the TapeAgents GitHub repository