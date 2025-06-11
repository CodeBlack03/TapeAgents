# Gaia Agent

The Gaia Agent is designed to answer knowledge-grounded questions using web search, calculations, and reasoning. The agent utilizes LLM models through APIs to accomplish these tasks. It is equipped with the following tools:
- [BrowserGym](https://github.com/ServiceNow/BrowserGym) as the main web browser tool
- Web search tool
- Code executor
- Media reader

We demonstrate how it solves tasks from the [GAIA Benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard).

## Structure

The agent is defined using this [config](../../conf/gaia_agent.yaml), which implements the following workflow:

- Expose the set of all available actions and thoughts to the model in each prompt
- Render the whole tape into the prompt, trimming only when the tape does not fit into the context window
- Append a short textual guidance prompt that briefly instructs the LLM on what to do next
- Append hints about formatting to the end of the prompt

The agent is free to choose which thoughts and actions to use to satisfy the current guidance recommendations without additional constraints of a specific node or subagent.

Additionally, the Gaia Agent implements an initial planning step, which produces a "plan" in the form of a sequence of free-form descriptions of the actions that should be taken to solve the task.

## Results

Results are also available on the [Hugging Face Leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard) under the names "TapeAgent ...".

| Model | Validation Accuracy | Test Accuracy |
| --- | --- | --- |
| Sonnet 3.7 maj@3 | 55.8 | |
| Sonnet 3.7 | 53.9 | |
| GPT-4o | 37.0 | 33.2 |
| GPT-4o mini maj@3 | 32.3 | |
| GPT-4o mini | 27.3 | 21.9 |

## Tape Example 
<img width="867" alt="gaia_perfect_demo_tape" src="https://github.com/user-attachments/assets/a81c22d8-9cf5-42c4-a390-933108753966">

## Quickstart

Perform all the following steps from the top folder of the repo.
First, install the dependencies for file converters:

```bash
pip install 'tapeagents[converters]'
```

Then, ensure you have `FFmpeg` version 7.1.x or newer installed (more details [here](https://github.com/kkroening/ffmpeg-python?tab=readme-ov-file#installing-ffmpeg)).

If you want to convert PDF files to images to preserve tables and complex formatting, please install the prerequisites of the pdf2image library as described [in their documentation](https://pypi.org/project/pdf2image/).

Then you can run the agent using the following commands:

- `uv run -m examples.gaia_agent.scripts.studio` - Interactive GUI that allows you to set the task for the agent and observe how it solves it step by step.
- `uv run -m examples.gaia_agent.scripts.evaluate` - Script to run evaluation on the GAIA validation set.
- `uv run -m examples.gaia_agent.scripts.tape_browser` - Gradio UI for exploring the tapes and metrics produced during evaluation.

If you see the error `remote: Access to dataset gaia-benchmark/GAIA is restricted. You must have access to it and be authenticated to access it. Please log in.`, you need to log in to your Hugging Face account first:

```bash
huggingface-cli login
```

## Running GAIA Agent with Azure OpenAI (GPT-4o)

To run the GAIA agent on the validation set using Azure OpenAI's GPT-4o model, follow these steps:

### 1. Set up Azure OpenAI Environment Variables

First, configure your Azure OpenAI credentials:

```bash
export AZURE_API_KEY="your-azure-openai-api-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2024-02-01"
```

### 2. Configure the Azure GPT-4o Model

Edit the Azure GPT-4o configuration file at `conf/llm/azure_gpt4o.yaml` to match your deployment:

```yaml
_target_: tapeagents.llms.LiteLLM
model_name: azure/your-gpt-4o-deployment-name  # Replace with your actual deployment name
use_cache: false
stream: false
context_size: 128000
observe_llm_calls: true
parameters:
  temperature: 0.7
  max_tokens: 2000
  top_p: 0.95
  frequency_penalty: 0.0
  presence_penalty: 0.0
```

### 3. Create a Custom Configuration

Create a new configuration file `conf/gaia_azure_gpt4o.yaml`:

```yaml
defaults:
  - _self_
  - llm: azure_gpt4o
  - agent: gaia
  - environment: web_browser

exp_name: azure_gpt4o_validation_run
exp_path: outputs/gaia/runs/${exp_name}

split: validation
batch: 4  # Adjust based on your rate limits
retry_unsolved: true

only_tasks: []  # Leave empty to run all tasks, or specify specific tasks like:
# - [1, 0]  # Level 1, Task 0
# - [1, 1]  # Level 1, Task 1

hydra:
  run:
    dir: ${exp_path}
```

### 4. Run the Evaluation

Execute the evaluation on the GAIA validation set:

```bash
# Run all validation tasks
uv run -m examples.gaia_agent.scripts.evaluate --config-name=gaia_azure_gpt4o

# Run specific tasks only (Level 1, first 5 tasks)
uv run -m examples.gaia_agent.scripts.evaluate --config-name=gaia_azure_gpt4o \
  only_tasks="[[1,0],[1,1],[1,2],[1,3],[1,4]]"

# Run with different batch size (for rate limiting)
uv run -m examples.gaia_agent.scripts.evaluate --config-name=gaia_azure_gpt4o \
  batch=1

# Run only Level 1 tasks
uv run -m examples.gaia_agent.scripts.evaluate --config-name=gaia_azure_gpt4o \
  only_tasks="[[1,i] for i in range(30)]"
```

### 5. Monitor Results

The results will be saved to `outputs/gaia/runs/azure_gpt4o_validation_run/`:

- **Tapes**: Individual task execution traces in `tapes/`
- **Logs**: Execution logs in `logs/`
- **Images**: Screenshots and attachments in `attachments/images/`

### 6. View Results with Tape Browser

Launch the interactive tape browser to explore results:

```bash
uv run -m examples.gaia_agent.scripts.tape_browser \
  --exp-path outputs/gaia/runs/azure_gpt4o_validation_run
```

### 7. Collect and Analyze Results

Generate summary statistics:

```bash
uv run -m examples.gaia_agent.scripts.collect_results \
  outputs/gaia/runs/azure_gpt4o_validation_run
```

### Example Commands

```bash
# Quick test with first 3 tasks from Level 1
uv run -m examples.gaia_agent.scripts.evaluate --config-name=gaia_azure_gpt4o \
  only_tasks="[[1,0],[1,1],[1,2]]" batch=1

# Full validation set evaluation (all levels)
uv run -m examples.gaia_agent.scripts.evaluate --config-name=gaia_azure_gpt4o \
  batch=4

# Level 1 only (easier tasks)
uv run -m examples.gaia_agent.scripts.evaluate --config-name=gaia_azure_gpt4o \
  only_tasks="[[1,i] for i in range(165)]"
```

### Troubleshooting

- **Rate Limits**: Reduce `batch` size to 1 if you hit Azure OpenAI rate limits
- **Authentication**: Ensure your Azure credentials are correctly set
- **Deployment Name**: Verify your GPT-4o deployment name in the config file
- **Permissions**: Make sure you have access to the GAIA dataset on Hugging Face
