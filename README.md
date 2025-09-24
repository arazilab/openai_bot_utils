# OpenAI Bot Utils

A lightweight Python module for building chat-based bots using OpenAI's API. This utility keeps track of message history if enabled and simplifies working with the Chat Completions endpoint.

## 🧠 Features

- Class-based structure
- Optional memory mode (disabled by default) — enable it for stateful interactions
- Customizable temperature, model, and token limits
- Supports text or json_object response formats
- Easy to import and re-use across multiple projects
- Includes a voting-based `BooleanConsensusAgent` for structured yes/no questions with repeat sampling

## 📦 Installation

Clone the repo into your project:

```bash
git clone https://github.com/arazilab/openai_bot_utils.git
```

## 🧩 How to Use

Add to `sys.path` and import:

```python
import sys
sys.path.append('./openai_bot_utils')
from bot_utils.core import Bot
from bot_utils.boolean_consensus_agent import BooleanConsensusAgent
```

### Create a bot instance and chat

By default, the bot is stateless (it does not remember previous messages):

```python
# Stateless bot (default): does not keep track of past messages
bot = Bot(
    system_prompt="You are a helpful assistant.",
    model="gpt-4.1-nano",
    temperature=0.7,
    response_format="text",
    max_completion_tokens=1024
)

response = bot.receive_output("Hello, who are you?")
print(response)
```

### (Optional) Enable memory for stateful conversations

```python
# Stateful bot: keeps history across turns
stateful_bot = Bot(
    system_prompt="You are a helpful assistant.",
    memory=True
)

print(stateful_bot.receive_output("What's your name?"))
```

### Use `BooleanConsensusAgent` for reliable yes/no answers

This special agent repeatedly asks the model a yes/no question and aggregates the results to improve reliability.

You can specify:
- `n_votes`: how many times to ask and return majority
- `target_confidence`: minimum confidence required, based on entropy (needs `min_votes` and `max_votes` too)

#### Example: Majority vote

```python
agent = BooleanConsensusAgent(
    question="Is this email likely to be a phishing attempt?",
    n_votes=7
)
result = agent("This is a limited time offer! Click here to claim your prize.")
print(result)  # True or False
```

#### Example: Confidence-based vote

```python
agent = BooleanConsensusAgent(
    question="Is this review positive?",
    target_confidence=0.8,
    min_votes=5,
    max_votes=11
)
result = agent("This product is amazing. I use it every day.")
print(result)  # True or False
```

## 🧱 Classes

### `Message`

Represents a single message in the conversation.

```python
Message(role, content)
```

- `role`: 'system', 'user', or 'assistant'
- `content`: string

### `Bot`

Handles full chat interaction and history (if enabled).

```python
Bot(
    system_prompt,
    model="gpt-4.1-nano",
    temperature=1.0,
    response_format="text",
    max_completion_tokens=2048,
    memory=False
)
```

**Method**:

```python
bot.receive_output(user_input)  # returns assistant's reply
```

### `BooleanConsensusAgent`

Makes yes/no decisions by repeating prompts and aggregating answers.

```python
BooleanConsensusAgent(
    question,
    n_votes=None,
    target_confidence=None,
    min_votes=3,
    max_votes=11,
    model="gpt-4.1-nano"
)
```

**Call**:

```python
agent(datapoint)  # returns True or False
```
