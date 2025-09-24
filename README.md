# OpenAI Bot Utils

`openai_bot_utils` helps you build chat bots with OpenAI models. It now supports the new **Responses API** and keeps **backward compatibility** with **Chat Completions**.

## Features

- Simple `Bot` class for chatting.
- `api_mode` switch: `"responses"` or `"chat_completions"`.
- Optional `memory=True` to keep history.
- JSON output support.
- `BooleanConsensusAgent` for yes/no with voting.

## Install

Clone the repo into your project folder.

```bash
git clone https://github.com/arazilab/openai_bot_utils.git
```

Add to `sys.path` or install as a local package.

```python
import sys
sys.path.append('./openai_bot_utils')
```

Set your OpenAI key.

```bash
export OPENAI_API_KEY=YOUR_KEY
```

## Quick start

### Stateless bot (default)

```python
from bot_utils.core import Bot

bot = Bot(
    system_prompt="You are a helpful assistant.",
    model="gpt-4.1-nano",
    temperature=0.7,
    response_format="text",
    max_completion_tokens=512,
    memory=False,                # no history
    api_mode="responses"         # use new API; omit to keep legacy
)

print(bot.receive_output("Hello! What can you do?"))
```

### Stateful bot (keeps history)

```python
from bot_utils.core import Bot

bot = Bot(
    system_prompt="You are a helpful assistant.",
    memory=True,                 # keep history
    api_mode="responses"
)

print(bot.receive_output("Remember my name is Alex."))
print(bot.receive_output("What is my name?"))
```

### JSON output

**Chat Completions mode** (legacy) uses `response_format={"type": "json_object"}`.
**Responses mode** maps this to the new structured output setting for you.

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

## Yes/No with voting

`BooleanConsensusAgent` repeats the question and aggregates results.

```python
from bot_utils.boolean_consensus_agent import BooleanConsensusAgent

agent = BooleanConsensusAgent(
    question="Is this review positive?",
    n_votes=7,                   # majority vote
    model="gpt-4.1-nano",
    api_mode="responses"
)

print(agent("This product is amazing. I love it."))  # True or False
```

Confidence mode:

```python
from bot_utils.boolean_consensus_agent import BooleanConsensusAgent

agent = BooleanConsensusAgent(
    question="Is this phishing?",
    target_confidence=0.8,
    min_votes=5,
    max_votes=11,
    api_mode="responses"
)

print(agent("Click this link to claim your prize!"))
```

## API modes

- `api_mode="responses"` uses `client.responses.create`.
- `api_mode="chat_completions"` keeps the old `client.chat.completions.create`.
- The default is `"chat_completions"` for backward compatibility.
- Set `memory=True` to keep local message history across calls.
- Set `store=False` is already used on legacy calls to avoid server storage.

## Migration notes

- Switch bots by setting `api_mode="responses"`.
- If you used `response_format="json_object"` before, keep it. The wrapper maps it correctly for the new API.
- No other changes needed for basic text chat.

## Minimal examples

```python
# responses API
bot = Bot("You are concise.", api_mode="responses")
print(bot.receive_output("Summarize: OpenAI builds AI tools."))

# legacy chat completions API
bot = Bot("You are concise.", api_mode="chat_completions")
print(bot.receive_output("Summarize: OpenAI builds AI tools."))
```

## Project structure

```
bot_utils/
  __init__.py
  core.py                   # Bot and Message
  boolean_consensus_agent.py
README.md
```

## Requirements

- Python 3.9+
- openai >= 1.0.0
- An OpenAI API key in `OPENAI_API_KEY`

## FAQ

**Does my old code break?**  
No. The default mode is the old Chat Completions. Your code keeps working.

**How do I opt in to the new API?**  
Pass `api_mode="responses"` when you create `Bot` or when you create `BooleanConsensusAgent`.

**Why JSON sometimes fails?**  
If the model returns invalid JSON, check your prompt. Keep it short and explicit. In Responses mode the wrapper requests strict JSON for you.

## License

MIT
