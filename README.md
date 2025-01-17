# Port of TinyAgent

This is a port of TinyAgent which allows for evaluation and completely local execution.

## Evaluation

- We modify the code to use a single planning step with LLMCompiler. Replanning requires the actual results of running tools on MacOS, which makes evaluation challenging. A single planning step can evaluate the planner LLM without relying on tool execution results.

- We also add a convenient evaluation mode flag which makes sure that tools are not actually run (since we don't need tool execution results for evaluation and also since many tools rely on being run in MacOS). This flag also makes sure that placeholder arguments produced by the planner are not replaced (since they are necessary for creating the DAG for evaluation).

- TinyAgent is also coupled with OpenAI's Whisper, which we remove since it isn't relevant for evaluation.

- We provide code to evaluate TinyAgent on the TinyAgent dataset

- We also provide paraphrases of the tool names, tool argument names and tool descriptions of all the tools in the TinyAgent dataset. These were used to probe the generalization capability of TinyAgent models when they were given the same tools with different but equivalent names, arguments and descriptions.

- The parsing of the LLM planner's output had some assumptions which don't always hold true. For example it was assumed that the LLM would always produce a list of actions starting with index 1. When this was not the case the program would crash. We modified the parsing to make it more robust.

To run the evaluation fill the config at `src/configs/eval/config.json` and then run `src/eval/eval.py`

## Completely local execution

- The orignal TinyAgent lets you run local LLMs but still requires the OpenAI embedding API

- We replace OpenAI embeddings with embeddings from `nomic-ai/nomic-embed-text-v1.5`. It has similar performance on the MTEB benchmark as the `text-embedding-3-small` used by the original TinyAgent

- The embeddings of in-context examples produced by `nomic-ai/nomic-embed-text-v1.5` are provided
-  A script is provided which can use any other embedding model to generate the embeddings


Check out the original project at [https://github.com/SqueezeAILab/TinyAgent](https://github.com/SqueezeAILab/TinyAgent)

TinyAgent aims to enable complex reasoning and function calling capabilities in Small Language Models (SLMs) that can be deployed securely and privately at the edge. Traditional Large Language Models (LLMs) like GPT-4 and Gemini-1.5, while powerful, are often too large and resource-intensive for edge deployment, posing challenges in terms of privacy, connectivity, and latency. TinyAgent addresses these challenges by training specialized SLMs with high-quality, curated data, and focusing on function calling with [LLMCompiler](https://github.com/SqueezeAILab/LLMCompiler). As a driving application, TinyAgent can interact with various MacOS applications, assisting users with day-to-day tasks such as composing emails, managing contacts, scheduling calendar events, and organizing Zoom meetings.

## Demo

<a href="https://youtu.be/0GvaGL9IDpQ" target="_blank" rel="noopener noreferrer">
  <img src="https://github.com/SqueezeAILab/TinyAgent/assets/65496977/014542fe-e4a1-4113-92a5-873fe3a01715" alt="TinyAgent Demo" width="700">
</a>

## What can TinyAgent do?

TinyAgent is equipped with 16 different functions that can interact with different applications on Mac, which includes:

### 📧 Mail

- **Compose New Email**
  - Start a new email with options for adding recipients and attachments.
  - _Example Query:_ “Email Sid and Nick about the meeting with attachment project.pdf.”
- **Reply to Emails**
  - Respond to received emails, optionally adding new recipients and attachments.
  - _Example Query:_ “Reply to Alice's email with the updated budget document attached.”
- **Forward Emails**
  - Forward an existing email to other contacts, including additional attachments if necessary.
  - _Example Query:_ “Forward the project briefing to the marketing team.”

### 📇 Contacts

- Retrieve phone numbers and email addresses from the contacts database.
- _Example Query:_ “Get John’s phone number” or “Find Alice’s email address.”

### 📨 SMS

- Send text messages to contacts directly from TinyAgent.
- _Example Query:_ “Send an SMS to Canberk saying ‘I’m running late.’”

### 📅 Calendar

- Create new calendar events with specified titles, dates, and times.
- _Example Query:_ “Create an event called 'Meeting with Sid' on Friday at 3 PM.”

### 🗺️ Maps

- Find directions or open map locations for points of interest via Apple Maps.
- _Example Query:_ “Show me directions to the nearest Starbucks.”

### 🗒️ Notes

- Create, open, and append content to notes stored in various folders.
- _Example Query:_ “Create a note called 'Meeting Notes' in the Meetings folder.”

### 🗂️ File Management

- **File Reading**
  - Open and read files directly through TinyAgent.
  - _Example Query:_ “Open the LLM Compiler.pdf.”
- **PDF Summarization**
  - Generate summaries of PDF documents, enhancing content digestion and review efficiency.
  - _Example Query:_ “Summarize the document LLM Compiler.pdf and save the summary in my Documents folder.”

### ⏰ Reminders

- Set reminders for various activities or tasks, ensuring nothing is forgotten.
- _Example Query:_ “Remind me to call Sid at 3 PM about the budget approval.”

### 🎥 Zoom Meetings

- Schedule and organize Zoom meetings, including setting names and times.
- _Example Query:_ “Create a Zoom meeting called 'Team Standup' at 10 AM next Monday.”

### 💬 Custom Instructions

- Write and configure specific instructions for your TinyAgent.
- _Example Query:_ “Always cc team members Nick and Sid in all emails.”

> You can choose to enable/disable certain apps by going to the Preferences window.

### 🤖 Sub-Agents

Depending on the task simplicity, TinyAgent orchestrates the execution of different more specialized or smaller LMs. TinyAgent currently can operate LMs that can summarize a PDF, write emails, or take notes.

> See the [Customization](#customization) section to see how to add your own sub-agentx.

### 🛠️ ToolRAG

When faced with challenging tasks, SLM agents require appropriate tools and in-context examples to guide them. If the model sees irrelevant examples, it can hallucinate. Likewise, if the model sees the descriptions of the tools that it doesn’t need, it usually gets confused, and these tools take up unnecessary prompt space. To tackle this, TinyAgent uses ToolRAG to retrieve the best tools and examples suited for a given query. This process has minimal latency and increases the accuracy of TinyAgent substantially. Please take a look at our [blog post](https://bair.berkeley.edu/blog/2024/05/29/tiny-agent) and our [ToolRAG model](https://huggingface.co/squeeze-ai-lab/TinyAgent-ToolRAG) for more details.

> You need to first install our [ToolRAG model](https://huggingface.co/squeeze-ai-lab/TinyAgent-ToolRAG) from Hugging Face and enable it from the TinyAgent settings to use it.  

## Providers

You can use with your OpenAI key, Azure deployments, or even your own local models!

### Azure Deployments

You need to provide your deployment name and the endpoints for the main agent/sub-agents/embedding model as well as the context length of the agent models in the 'Preferences' window.

### Local Models

You can plug-and-play every part of TinyAgent with your local models! TinyAgent can use an OpenAI-compatible server to run models locally. There are several options you can take:

- **[LMStudio](https://lmstudio.ai/) :** For models already on Huggingface, LMStudio provides an easy-to-use to interface to get started with locally served models.

- **[llama.cpp server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md):** However, if you want more control over your models, we recommend using the official llama.cpp server to get started. Please read through the tagged documentation to get started with it.

> All TinyAgent needs is the port numbers that you are serving your model at and its context length.

## Fine-tuned TinyAgents

We also provide our own fine-tuned open source models, TinyAgent-1.1B and TinyAgent-7B! We curated a [dataset](https://huggingface.co/datasets/squeeze-ai-lab/TinyAgent-dataset) of 40000 real-life use cases for TinyAgent and fine-tuned two small open-source language models on this dataset with LoRA. After fine-tuning and using ToolRAG, both TinyAgent-1.1B and TinyAgent-7B exceed the performance of GPT-4-turbo. Check out our for the specifics of dataset generation, evaluation, and fine-tuning.

| Model                                                                                                                                                       | Success Rate |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| GPT-3.5-turbo                                                                                                                                               | 65.04%       |
| GPT-4-turbo                                                                                                                                                 | 79.08%       |
| [TinyLLama-1.1B-32K-Instruct](https://huggingface.co/Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct)                                                            | 12.71%       |
| [WizardLM-2-7b](https://huggingface.co/MaziyarPanahi/WizardLM-2-7B-GGUF)                                                                                    | 41.25%       |
| TinyAgent-1.1B + ToolRAG / [[hf](https://huggingface.co/squeeze-ai-lab/TinyAgent-1.1B)] [[gguf](https://huggingface.co/squeeze-ai-lab/TinyAgent-1.1B-GGUF)] | **80.06%**   |
| TinyAgent-7B + ToolRAG / [[hf](https://huggingface.co/squeeze-ai-lab/TinyAgent-7B)] [[gguf](https://huggingface.co/squeeze-ai-lab/TinyAgent-7B-GGUF)]       | **84.95%**   |

## Customization

You can customize your TinyAgent by going to `~/Library/Application Support/TinyAgent/tinyagent-llmcompiler` directory and changing the code yourself.

### Using TinyAgent programmatically

You can use TinyAgent programmatically by just passing in a config file.

```python
from src.tiny_agent.tiny_agent import TinyAgent
from src.tiny_agent.config import get_tiny_agent_config

config_path = "..."
tiny_agent_config = get_tiny_agent_config(config_path=config_path)
tiny_agent = TinyAgent(tiny_agent_config)

await TinyAgent.arun(query="Create a meeting with Sid and Lutfi for tomorrow 2pm to discuss the meeting notes.")
```

### Adding your own tools

1. Navigate to `src/tiny_agent/models.py` and add your tool;s name to `TinyAgentToolName(Enum)`

```python
class TinyAgentToolName(Enum):
...
CUSTOM_TOOL_NAME = "custom_tool"
```

2. Navigate to `src/tiny_agent/tiny_agent_tools.py` and define your tool.

```python
def get_custom_tool(...) -> list[Tool]:
async def tool_coroutine(...) -> str: # Needs to return a string
...
return ...

    custom_tool = Tool(
        name=TinyAgentToolName.CUSTOM_TOOL_NAME,
        func=tool_coroutine,
        description=(
            f"{TinyAgentToolName.CUSTOM_TOOL_NAME.value}(...) -> str\n"
            "<The description of the tool>"
        )
    )

    return [custom_tool]
```

3. Add your tools to the `get_tiny_agent_tools` function.

```python
def get_tiny_agent_tools(...):
...
tools += get_custom_tools(...)
...
```

> Note: Adding your own tools only works for GPT models since our open-source models and ToolRAG were only fine-tuned on the original TinyAgent toolset.

### Adding your own sub-agents

You can also add your own custom subagents. To do so, please follow these steps:

1. All sub-agents inherit from `SubAgent` class in `src/tiny_agent/sub_agents/sub_agent.py`. Your custom agent should inherit from this abstract class and define the `__call__` method.

```python
class CustomSubAgent(SubAgent):

    async def __call__(self, ...) -> str:
        ...
        return response
```

2. Add your custom agent to `TinyAgent` in `src/tiny_agent/tiny_agent.py`

```python
from src.tiny_agent.sub_agents.custom_agent import CustomAgent

class TinyAgent:
...
custom_agent: CustomAgent

    def __init__(...):
        ...
        self.custom_agent = CustomAgent(
            sub_agent_llm, config.sub_agent_config, config.custom_instructions
        )
        ...
```

3. After defining your custom agent and adding it to TinyAgent, you should create a tool that calls this agent. Please refer to the [Adding your own tools](#adding-your-own-tools) section to see how to do so

## Citation

```
@inproceedings{erdogan2024tinyagent,
  title={TinyAgent: Function Calling at the Edge},
  author={Erdogan, Lutfi and Lee, Nicholas and Jha, Siddharth and Kim, Sehoon and Tabrizi, Ryan and Moon, Suhong and Hooper, Coleman and Anumanchipalli, Gopala and Keutzer, Kurt and Gholami, Amir},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  pages={80--88},
  year={2024}
}
```
