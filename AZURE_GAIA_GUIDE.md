# GAIA Benchmark with Azure OpenAI - Complete Guide

This guide shows how to run GAIA benchmark comparisons using Azure OpenAI instead of standard OpenAI.

## Quick Start

### 1. Setup Azure OpenAI Environment

Set your Azure OpenAI credentials:

```bash
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://your-resource-name.openai.azure.com/"
export AZURE_API_VERSION="2024-02-15-preview"
```

### 2. Install Dependencies

```bash
pip install pdf2image python-Levenshtein huggingface_hub omegaconf hydra-core pandas
pip install -e .  # Install TapeAgents
```

### 3. Setup GAIA Dataset Access

```bash
huggingface-cli login
```

### 4. Run the Benchmark

```bash
# Run with 10% sample of each level
python run_gaia_azure.py --sample-percent 0.1

# Or run with max 5 tasks per level
python run_gaia_azure.py --max-tasks-per-level 5

# Verbose output
python run_gaia_azure.py --sample-percent 0.1 --verbose
```

## Azure OpenAI Configuration

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_API_KEY` | Your Azure OpenAI API key | `"abc123..."` |
| `AZURE_API_BASE` | Your Azure OpenAI endpoint | `"https://myresource.openai.azure.com/"` |
| `AZURE_API_VERSION` | API version to use | `"2024-02-15-preview"` |

### Model Configuration

The benchmark uses `azure/gpt-4o-mini` by default. You can modify this in the configuration files:

```yaml
llm:
  _target_: tapeagents.llms.LiteLLM
  model_name: azure/gpt-4o-mini  # Your Azure deployment name
  use_cache: true
  stream: false
  parameters:
    temperature: 0.7
    max_tokens: 1500
    top_p: 0.9
```

## Benchmark Components

### 1. Base TapeAgent
- Uses standard GAIA agent configuration
- Linear execution flow
- Text-based planning
- Basic error handling

### 2. CodeAct + Autonomous Learning Agent
- Enhanced workflow capabilities
- Executable code planning
- Improved error localization
- Autonomous learning features

## Command Line Options

```bash
python run_gaia_azure.py [OPTIONS]

Options:
  --sample-percent FLOAT    Percentage of tasks to sample (default: 0.1)
  --max-tasks-per-level INT Maximum tasks per level (overrides sample-percent)
  --verbose, -v            Enable verbose logging
  --help                   Show help message
```

## Example Usage

### Small Scale Test (Recommended First Run)
```bash
# Test with just 2 tasks per level
python run_gaia_azure.py --max-tasks-per-level 2 --verbose
```

### Medium Scale Evaluation
```bash
# 10% of each level
python run_gaia_azure.py --sample-percent 0.1
```

### Custom Evaluation
```bash
# 5 tasks per level with verbose output
python run_gaia_azure.py --max-tasks-per-level 5 --verbose
```

## Output and Results

### Directory Structure
```
gaia_azure_results/
├── base_agent/
│   ├── tapes/           # Individual task results
│   └── logs/            # Execution logs
├── codeact_autonomous/
│   ├── tapes/           # Individual task results
│   └── logs/            # Execution logs
├── analysis_results.json    # Detailed analysis
├── evaluation_logs.json     # Evaluation process logs
└── benchmark_report.txt     # Human-readable summary
```

### Sample Output
```
GAIA BENCHMARK RESULTS WITH AZURE OPENAI
======================================================================
Sample: 10.0% of each level

Base Agent:
  Accuracy: 75.0% (6/8)
  Avg Time: 45.2s

CodeAct + Autonomous:
  Accuracy: 87.5% (7/8)
  Avg Time: 38.7s

Improvement:
  Accuracy: +12.5% (+16.7%)
  Speed: +6.5s per task
======================================================================
```

## Troubleshooting

### Common Issues

1. **Azure API Key Issues**
   ```
   Error: AuthenticationError: Incorrect API key
   ```
   - Verify your `AZURE_API_KEY` is correct
   - Check that the key has proper permissions

2. **Endpoint Configuration**
   ```
   Error: Invalid URL or endpoint
   ```
   - Ensure `AZURE_API_BASE` includes the full URL with protocol
   - Verify the endpoint is accessible

3. **Model Deployment**
   ```
   Error: Model not found
   ```
   - Check that `gpt-4o-mini` is deployed in your Azure OpenAI resource
   - Update the model name to match your deployment

4. **GAIA Dataset Access**
   ```
   Error: Access to dataset gaia-benchmark/GAIA is restricted
   ```
   - Run `huggingface-cli login` and authenticate
   - Ensure you have access to the GAIA dataset

### Performance Optimization

1. **Reduce Sample Size**: Start with `--max-tasks-per-level 2` for testing
2. **Enable Caching**: LLM responses are cached by default to avoid repeated calls
3. **Monitor Costs**: Azure OpenAI charges per token, monitor usage in Azure portal

## Advanced Configuration

### Custom Model Configuration

Create a custom LLM config file:

```yaml
# conf/llm/custom_azure.yaml
_target_: tapeagents.llms.LiteLLM
model_name: azure/your-custom-deployment
use_cache: true
stream: false
parameters:
  temperature: 0.5
  max_tokens: 2000
  top_p: 0.95
```

### Environment Customization

Modify the agent configurations in the generated config files:

```yaml
# Enhanced environment settings
environment:
  _target_: tapeagents.environment.ToolCollectionEnvironment
  tools:
    - web_search
    - code_executor
    - file_reader
  safety_checks: true
  timeout: 300
```

## Integration with Existing Workflows

### Using with Hydra Configs

The benchmark generates Hydra-compatible configurations that can be used with existing TapeAgents workflows:

```bash
# Use generated config directly
python -m examples.gaia_agent.scripts.evaluate \
  --config-path gaia_azure_results \
  --config-name base_config
```

### Batch Processing

For large-scale evaluations, consider running in batches:

```bash
# Level 1 only
python run_gaia_azure.py --max-tasks-per-level 10 --level 1

# Level 2 only  
python run_gaia_azure.py --max-tasks-per-level 10 --level 2

# Level 3 only
python run_gaia_azure.py --max-tasks-per-level 10 --level 3
```

## Monitoring and Logging

### Azure OpenAI Monitoring

Monitor your Azure OpenAI usage:
1. Go to Azure Portal
2. Navigate to your OpenAI resource
3. Check "Metrics" for token usage and costs

### Local Logging

The benchmark provides detailed logging:
- Set `--verbose` for detailed output
- Check `evaluation_logs.json` for execution details
- Review individual tape files for task-level analysis

## Cost Estimation

Approximate costs for different sample sizes (using GPT-4o Mini pricing):

| Sample Size | Estimated Tasks | Approx. Tokens | Est. Cost (USD) |
|-------------|----------------|-----------------|-----------------|
| 2 per level | 6 tasks | ~50K tokens | $0.01 |
| 5 per level | 15 tasks | ~125K tokens | $0.03 |
| 10% sample | ~30 tasks | ~250K tokens | $0.06 |

*Costs are estimates and may vary based on task complexity and model responses.*

## Next Steps

1. **Start Small**: Begin with `--max-tasks-per-level 2`
2. **Verify Setup**: Check that both agents run successfully
3. **Scale Up**: Gradually increase sample size
4. **Analyze Results**: Review the generated reports and identify improvements
5. **Customize**: Modify configurations for your specific use case

## Support

For issues:
1. Check the troubleshooting section above
2. Verify Azure OpenAI setup and permissions
3. Test with minimal sample size first
4. Review the generated logs for detailed error information

---

This guide provides everything needed to run GAIA benchmark comparisons with Azure OpenAI, demonstrating the advantages of the CodeAct + Autonomous Learning framework.