# OpenAI Bot Utils

**OpenAI Bot Utils** is a lightweight Python module for building chat‑based bots using OpenAI models.  It supports both the legacy Chat Completions API and the newer Responses API.

## Features

- **Bot class**: send prompts to OpenAI models with optional history. You can choose between the legacy chat completions endpoint and the newer responses endpoint.
- **API mode switch**: set `api_mode="responses"` to use the new Responses API or leave it as `"chat_completions"` for backward compatibility.
- **Memory option**: enable `memory=True` to keep a local history of messages so the bot can remember past interactions.
- **JSON output**: set `response_format="json_object"` to have the bot return JSON output.
- **BooleanConsensusAgent**: ask a yes/no question multiple times and return the result via majority vote or confidence threshold.
- **MultiClassConsensusAgent**: ask a multiple‑choice question repeatedly and choose the option with the most votes.

## Install

Clone this repository into your project folder:

```bash
git clone https://github.com/arazilab/openai_bot_utils.git
```

Add the cloned folder to your Python path or install it locally.  For example:

```python
import sys
sys.path.append('./openai_bot_utils')
```

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=your_openai_key_here
```

## Quick Start

### Stateless bot

```python
from bot_utils.core import Bot

bot = Bot(
    system_prompt="You are a helpful assistant.",
    model="gpt-4o",
    temperature=0.7,
    response_format="text",
    max_completion_tokens=512,
    memory=False,
    api_mode="responses"  # use the new API
)

print(bot.receive_output("Hello! What can you do?"))
```

### Stateful bot (with memory)

```python
from bot_utils.core import Bot

bot = Bot(
    system_prompt="You are a helpful assistant.",
    memory=True,
    api_mode="responses"
)

print(bot.receive_output("Remember my name is Alex."))
print(bot.receive_output("What is my name?"))
```

### JSON output

```python
from bot_utils.core import Bot

bot = Bot(
    system_prompt="Return a JSON with one key 'answer'.",
    response_format="json_object",
    api_mode="responses"
)

text = "Is water wet? Answer true or false."
print(bot.receive_output(text))  # e.g. {"answer": true}
```

### Yes/No with voting

```python
from bot_utils.boolean_consensus_agent import BooleanConsensusAgent

agent = BooleanConsensusAgent(
    question="Is this review positive?",
    n_votes=7,          # majority vote
    model="gpt-4o",
    api_mode="responses"
)

print(agent("This product is amazing. I love it."))  # True or False

# Confidence mode
agent2 = BooleanConsensusAgent(
    question="Is this phishing?",
    target_confidence=0.8,
    min_votes=5,
    max_votes=11,
    api_mode="responses"
)

print(agent2("Click this link to claim your prize!"))
```

### Multiple‑choice voting

```python
from bot_utils.multiclass_consensus_agent import MultiClassConsensusAgent

options = {
    "A": "positive sentiment",
    "B": "neutral sentiment",
    "C": "negative sentiment"
}

agent = MultiClassConsensusAgent(
    question="What is the sentiment of this review?",
    options=options,
    n_votes=5,
    model="gpt-4o",
    api_mode="responses"
)

print(agent("I love this product!"))  # returns 1, 2 or 3
```

## API modes

- `api_mode="responses"` uses `client.responses.create`.
- `api_mode="chat_completions"` keeps the legacy behaviour and calls `client.chat.completions.create`.
- The default is `"chat_completions"` for backward compatibility.

## Requirements

- Python 3.9 or later
- `openai` library version 1.0.0 or later
- An OpenAI API key in `OPENAI_API_KEY`

## License

MIT
