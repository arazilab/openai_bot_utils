# OpenAI Bot Utils

A lightweight Python module for building chat-based bots using OpenAI's API. This utility keeps track of message history and simplifies working with the Chat Completions endpoint.

## 🧠 Features

- Class-based structure
- Optional memory mode (enabled by default) — disable it for stateless interactions
- Customizable temperature, model, and token limits
- Supports `text` or `json_object` response formats
- Easy to import and re-use across multiple projects

## 📦 Installation

Clone the repo into your project:

```bash
git clone https://github.com/arazilab/openai_bot_utils.git
```

## 🧩 How to Use

1. Add to sys.path

```python
import sys
sys.path.append('./openai_bot_utils')
from bot_utils.core import Bot
```
2. Create a bot instance and chat

```python
# Create a stateful bot (default): remembers previous messages
bot = Bot(
    system_prompt="You are a helpful assistant.",  # Initial instruction to guide the bot's behavior
    model="gpt-4o-mini",                           # OpenAI model to use
    temperature=0.7,                               # Controls randomness: higher = more creative
    response_format="text",                        # Can also be "json_object" if needed
    max_completion_tokens=1024                     # Limits how long the response can be
)

# Start chatting
response = bot.receive_output("Hello, who are you?")
print(response)
```

3. (Optional) Create a stateless bot

If you want a one-turn interaction — where the bot does not remember anything from past messages — set `memory=False`:

```
# Stateless bot: does not keep track of previous inputs or outputs
stateless_bot = Bot(
    system_prompt="You are a helpful assistant.",
    memory=False
)

# Every call is treated like the first interaction
print(stateless_bot.receive_output("What's your name?"))  # Bot has no memory of earlier messages
```

## 🧱 Classes

### Message

Represents a single message in the conversation. Automatically formats messages for OpenAI’s API.

Constructor:

```python
Message(role, content)
```

- role: `'system'`, `'user'`, or `'assistant'`
- content: string content

### Bot

Handles the full chat interaction, stores the history, and returns model responses.

Constructor:

```python
Bot(system_prompt, model="gpt-4o-mini", temperature=1.0, response_format="text", max_completion_tokens=2048, memory=True)
```

- system_prompt: Instruction for the assistant’s behavior
- model: Optional, OpenAI model name
- temperature: Optional, controls randomness
- response_format: Optional, "text" or "json_object"
- max_completion_tokens: Optional, max response tokens

Method:

```python
bot.receive_output(user_input)  # returns assistant's reply
```
