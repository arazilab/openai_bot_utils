# OpenAI Bot Utils

A lightweight Python module for building bots on top of OpenAI's **Responses API**.  
It supports stateless and stateful conversations, structured outputs, and repeated-vote agents for reliable yes/no decisions.

## 🧠 Features

- Modernized for the **Responses API** (`client.responses.create`)
- Class-based design for clean reuse
- Optional memory mode for stateful interactions
- Configurable model, temperature, token limits, reasoning effort, and verbosity
- Supports **text** and **json_object** response formats
- Includes a voting-based `BooleanConsensusAgent` for reliable yes/no answers
- Uses `tqdm` progress bars to visualize vote loops

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

By default, the bot is stateless (does not remember past turns):

```python
bot = Bot(
    system_prompt="You are a helpful assistant.",
    model="gpt-5",
    temperature=0.7,
    response_format="text",
    max_completion_tokens=1024
)

response = bot.receive_output("Hello, who are you?")
print(response)
```

### (Optional) Enable memory for stateful conversations

```python
stateful_bot = Bot(
    system_prompt="You are a helpful assistant.",
    model="gpt-5",
    memory=True
)

print(stateful_bot.receive_output("What's your name?"))
```

### Use `BooleanConsensusAgent` for reliable yes/no answers

This agent repeats the yes/no question until either:
- A fixed number of votes are collected (majority mode), or
- The confidence level reaches the target threshold (confidence mode).

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

### `Bot`

Wrapper for `client.responses.create`. Handles conversation state, formatting, and response extraction.

```python
Bot(
    system_prompt,
    model="gpt-5",
    temperature=1.0,
    response_format="text",       # "text" or "json_object"
    max_completion_tokens=2048,
    memory=False,
    reasoning_effort="minimal",   # minimal | medium | high
    verbosity="medium",           # low | medium | high
    store=False,
    include=None
)
```

**Method**:

```python
bot.receive_output(user_input)  # returns assistant's reply as str
```

### `BooleanConsensusAgent`

Yes/no agent that uses repeated model calls to improve reliability.  
Runs in **majority vote** or **confidence vote** mode.

```python
BooleanConsensusAgent(
    question,
    n_votes=None,               # set this for fixed number of votes
    target_confidence=None,     # set this for entropy-based confidence
    min_votes=3,
    max_votes=11,
    model="gpt-5"
)
```

**Call**:

```python
agent(datapoint)  # returns True or False
```
