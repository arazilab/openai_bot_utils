# OpenAI Bot Utils

A lightweight Python module for building chat-based bots using OpenAI's API. This utility keeps track of message history and simplifies working with the Chat Completions endpoint.

## 🧠 Features

- Class-based structure with memory (chat history)
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
2. Initialize and Chat

```python
bot = Bot(
    system_prompt="You are a helpful assistant.",
    model="gpt-4o-mini",
    temperature=0.7,
    response_format="text",
    max_completion_tokens=1024
)

response = bot.receive_output("Hello, who are you?")
print(response)
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
Bot(system_prompt, model="gpt-4o-mini", temperature=1.0, response_format="text", max_completion_tokens=2048)
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

## 🔒 License

This is a private utility. Not meant for public distribution.
