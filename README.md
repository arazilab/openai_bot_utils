# OpenAI Bot Utils

A lightweight Python module for building chat-based bots using OpenAI's API. This utility keeps track of message history if enabled and simplifies working with the Chat Completions endpoint.

🧠 Features

- Class-based structure
- Optional memory mode (disabled by default) — enable it for stateful interactions
- Customizable temperature, model, and token limits
- Supports text or json_object response formats
- Easy to import and re-use across multiple projects

📦 Installation

Clone the repo into your project:

```bash
git clone https://github.com/arazilab/openai_bot_utils.git
```

🧩 How to Use

Add to `sys.path` and import:

```python
import sys
sys.path.append('./openai_bot_utils')
from bot_utils.core import Bot
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

If you want the bot to remember past messages, set `memory=True`:

```python
# Stateful bot: keeps history across turns
stateful_bot = Bot(
    system_prompt="You are a helpful assistant.",
    memory=True
)

print(stateful_bot.receive_output("What's your name?"))
```

🧱 Classes

### `Message`

Represents a single message in the conversation.

**Constructor**:

```python
Message(role, content)
```

- `role`: 'system', 'user', or 'assistant'
- `content`: string

### `Bot`

Handles full chat interaction and history (if enabled).

**Constructor**:

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
