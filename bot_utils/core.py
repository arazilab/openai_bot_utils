from openai import OpenAI

# Initialize OpenAI API client
client = OpenAI()


class Message:
    """
    Represents a message in a chat conversation for the OpenAI API.

    Attributes:
        role (str): Role of the message sender. One of 'system', 'user', or 'assistant'.
        content (str): Text content of the message.
    """

    def __init__(self, role, content):
        """
        Initializes a Message instance.

        Args:
            role (str): Role of the sender.
            content (str): Message content.
        """
        self.role = role
        self.content = content

    def get_dict(self):
        """
        Converts the Message object into a dictionary format required by OpenAI API.

        Returns:
            dict: Dictionary with role and content formatted for OpenAI API.
        """
        return {
            "role": self.role,
            "content": [
                {
                    "type": "text",
                    "text": self.content
                }
            ]
        }


class Bot:
    """
    Wrapper for OpenAI's Chat API with optional message history (memory).

    Attributes:
        system_prompt (str): The initial prompt that sets the behavior of the assistant.
        model (str): OpenAI model to use (e.g., 'gpt-4o-mini').
        temperature (float): Controls randomness in output. Higher values = more random.
        response_format (str): Format of the response, either 'text' or 'json_object'.
        max_completion_tokens (int): Maximum number of tokens in the completion.
        memory (bool): Whether to keep message history across calls.
        messages (list): Stores conversation history if memory is enabled.
    """

    def __init__(
        self,
        system_prompt,
        model="gpt-4o-mini",
        temperature=1.0,
        response_format="text",
        max_completion_tokens=2048,
        memory=True
    ):
        """
        Initializes the Bot instance.

        Args:
            system_prompt (str): Initial system message to set assistant behavior.
            model (str, optional): Model to use. Default is 'gpt-4o-mini'.
            temperature (float, optional): Randomness in output. Default is 1.0.
            response_format (str, optional): 'text' or 'json_object'. Default is 'text'.
            max_completion_tokens (int, optional): Max tokens for a reply. Default is 2048.
            memory (bool, optional): Whether to remember past messages. Default is True.
        """
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.response_format = response_format
        self.max_completion_tokens = max_completion_tokens
        self.memory = memory
        self.messages = []

        # If memory is enabled, store the initial system prompt
        if self.memory:
            self.messages.append(Message('system', system_prompt).get_dict())

    def receive_output(self, input):
        """
        Sends user input to the OpenAI API and returns the assistant's reply.

        Args:
            input (str): The user message to send.

        Returns:
            str: The assistant's response message.
        """
        # If memory is enabled, append the user message and reuse full message history
        if self.memory:
            self.messages.append(Message('user', input).get_dict())
            prompt_messages = self.messages
        else:
            # If memory is off, only use system and user messages for each call
            prompt_messages = [
                Message('system', self.system_prompt).get_dict(),
                Message('user', input).get_dict()
            ]

        # Call OpenAI Chat API with the messages
        response = client.chat.completions.create(
            model=self.model,
            messages=prompt_messages,
            response_format={"type": self.response_format},
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            store=False  # Do not store conversation server-side
        )

        # Extract the assistant's reply from the response
        output = response.choices[0].message.content

        # If memory is on, store the assistant's reply as well
        if self.memory:
            self.messages.append(Message('assistant', output).get_dict())

        return output
