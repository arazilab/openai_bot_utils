"""
core.py - Updated Bot utilities to support OpenAI's Responses API.

This module defines two primary classes: ``Message`` and ``Bot``. The
``Message`` class is a simple container for a chat message with a role
(``system``, ``user`` or ``assistant``) and textual content. The ``Bot``
class wraps the OpenAI Python client to send user prompts and receive
assistant replies. It now supports both the legacy chat completions
endpoint and the newer responses API. Backwards compatibility is
maintained through the ``api_mode`` parameter; by default the
``Bot`` operates as before, using the chat completions API.

Usage example:

```
from openai import OpenAI
from bot_utils.core import Bot

# Create a stateless bot using the responses API
bot = Bot(
    system_prompt="You are a helpful assistant.",
    model="gpt-4o",
    api_mode="responses",
    temperature=0.7,
    memory=False,
)

# Send a prompt and print the response
print(bot.receive_output("Tell me a joke."))

# Create a stateful bot using the chat completions API (legacy)
legacy_bot = Bot(
    system_prompt="You remember everything I tell you.",
    model="gpt-4",
    memory=True,
    api_mode="chat_completions",
)

# Engage in a conversation
print(legacy_bot.receive_output("Hello!"))
print(legacy_bot.receive_output("What did I just say?"))
```

When ``api_mode`` is set to ``"responses"`` the bot internally calls
``client.responses.create`` and uses the ``input`` parameter instead of
``messages``. JSON output can still be requested via the
``response_format`` parameter; in this case the bot maps
``response_format="json_object"`` to a ``text.format`` specification
compatible with the responses API.

"""

from openai import OpenAI


# Initialize OpenAI API client once per module
client = OpenAI()


class Message:
    """
    Represents a message in a chat conversation for the OpenAI API.

    Each message contains a role and content. The ``get_dict`` method
    converts the instance into the dictionary format expected by the
    OpenAI API. In responses mode this representation is passed
    directly to the ``input`` parameter; in chat completions mode it
    goes into the ``messages`` array.
    """

    def __init__(self, role: str, content: str) -> None:
        """
        Initialize a Message instance.

        Args:
            role: Role of the sender. Valid values are 'system', 'user',
                or 'assistant'.
            content: The textual content of the message.
        """
        self.role = role
        self.content = content

    def get_dict(self) -> dict:
        """
        Convert the Message into a dictionary format compatible with
        both the Chat Completions API and the Responses API.  The
        content is returned as a plain string rather than a list of
        parts.  The OpenAI client automatically wraps plain text as
        needed.

        Returns:
            A dictionary with keys 'role' and 'content'.  The 'content'
            value is the raw text of the message.
        """
        return {
            "role": self.role,
            "content": self.content,
        }


class Bot:
    """
    Wrapper around OpenAI's Chat and Responses APIs with optional
    message history (memory) and response formatting.

    The Bot can operate in two modes controlled by the ``api_mode``
    parameter:

    - ``"chat_completions"`` (default): uses ``client.chat.completions.create``.
      This preserves backwards compatibility with earlier versions of
      ``openai_bot_utils``.
    - ``"responses"``: uses ``client.responses.create``, unlocking built-in
      tools and the new agentic features described in the migration
      guide.

    Attributes:
        system_prompt: The initial system message guiding the assistant's
            behaviour.
        model: The name of the model to use (e.g. 'gpt-4o' or 'gpt-3.5-turbo').
        temperature: Sampling temperature controlling randomness.
        response_format: 'text' (default) for plain text responses or
            'json_object' to enable JSON output. In responses mode this
            is mapped to the appropriate ``text.format`` configuration.
        max_completion_tokens: The maximum number of tokens the model may
            generate in its completion. When calling the responses API
            this value is passed as ``max_output_tokens``.
        memory: If True, the conversation history is retained and sent
            with each call; otherwise each call is stateless.
        api_mode: Which API endpoint to use: 'chat_completions' or
            'responses'.
    """

    def __init__(
        self,
        system_prompt: str,
        model: str = "gpt-4.1-nano",
        temperature: float = 1.0,
        response_format: str = "text",
        max_completion_tokens: int = 2048,
        memory: bool = False,
        api_mode: str = "chat_completions",
    ) -> None:
        # Store configuration
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.response_format = response_format
        self.max_completion_tokens = max_completion_tokens
        self.memory = memory
        self.api_mode = api_mode

        # Internal message history list
        self.messages: list[dict] = []

        # If memory is enabled, seed the history with the system prompt
        if self.memory:
            self.messages.append(Message("system", system_prompt).get_dict())

    def _build_prompt_messages(self, user_input: str) -> list[dict]:
        """
        Construct the list of message dictionaries to send to the API
        based on whether memory is enabled.

        Args:
            user_input: The latest user message to append.

        Returns:
            A list of message dictionaries ready for the OpenAI client.
        """
        if self.memory:
            # When remembering, append the new user message and use the
            # full conversation history
            self.messages.append(Message("user", user_input).get_dict())
            prompt_messages = self.messages
        else:
            # Stateless conversation: include system prompt and user input only
            prompt_messages = [
                Message("system", self.system_prompt).get_dict(),
                Message("user", user_input).get_dict(),
            ]
        return prompt_messages

    def _map_response_format(self) -> dict | None:
        """
        Map the ``response_format`` attribute to the parameters expected
        by the responses API. For chat completions mode we simply
        return ``{"type": response_format}`` to pass directly to
        ``response_format``. For responses mode we build a ``text``
        dictionary specifying a JSON object if required.

        Returns:
            For chat completions: a ``dict`` suitable for the
            ``response_format`` parameter.
            For responses: a ``dict`` suitable for the ``text``
            parameter, or ``None`` if no special formatting is needed.
        """
        if self.response_format == "json_object":
            if self.api_mode == "chat_completions":
                # Chat completions still accepts response_format
                return {"type": "json_object"}
            else:
                # Responses API requires text.format configuration
                return {"format": {"type": "json_object"}}
        else:
            # Plain text doesn't need special formatting
            if self.api_mode == "chat_completions":
                return {"type": "text"}
            else:
                return None

    def receive_output(self, user_input: str) -> str:
        """
        Send a user message to the OpenAI API and return the assistant's reply.

        This method constructs the appropriate payload depending on the
        configured ``api_mode``. When ``memory`` is enabled, the entire
        conversation history is sent; otherwise only the system and user
        messages are used. JSON mode is supported by setting
        ``response_format="json_object"``.

        Args:
            user_input: The user's message to the bot.

        Returns:
            The assistant's reply as a string.
        """
        prompt_messages = self._build_prompt_messages(user_input)

        if self.api_mode == "chat_completions":
            # Build response_format parameter
            response_format_param = self._map_response_format()
            # Call the chat completions endpoint
            response = client.chat.completions.create(
                model=self.model,
                messages=prompt_messages,
                response_format=response_format_param,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                store=False,  # Never persist server-side memory in legacy mode
            )
            # Extract the assistant's reply
            output = response.choices[0].message.content

        elif self.api_mode == "responses":
            # Build text formatting for JSON if necessary
            text_param = self._map_response_format()
            # Assemble parameters for the responses API.  Unlike the chat
            # completions endpoint, frequency_penalty and presence_penalty
            # are not currently supported on responses.create, so they are
            # omitted.  max_output_tokens limits the reply length.
            kwargs = {
                "model": self.model,
                "input": prompt_messages,
                "temperature": self.temperature,
                "max_output_tokens": self.max_completion_tokens,
                "top_p": 1,
                "store": False,  # do not persist reasoning or tool context
            }
            if text_param:
                kwargs["text"] = text_param
            # Call the responses endpoint
            response = client.responses.create(**kwargs)
            # Extract output text from the response
            output = response.output_text
        else:
            raise ValueError(
                f"Unsupported api_mode: {self.api_mode}. Choose 'chat_completions' or 'responses'."
            )

        # If memory is enabled, store the assistant's reply locally
        if self.memory:
            self.messages.append(Message("assistant", output).get_dict())

        return output
