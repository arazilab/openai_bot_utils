from openai import OpenAI

# Initialize OpenAI API client
client = OpenAI()

class Message:
    """
    Represents a message in a chat, used for OpenAI API calls.

    Attributes:
        role (str): The role of the message sender ('system', 'user', or 'assistant').
        content (str): The text content of the message.
    """
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def get_dict(self):
        """
        Returns the message formatted as required by the OpenAI API.
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
    A reusable wrapper for OpenAI's Chat API with message history support.

    Attributes:
        messages (list): Chat history, including system/user/assistant messages.
        model (str): Model to use (e.g., 'gpt-4o-mini').
        temperature (float): Sampling temperature.
        response_format (str): Either 'text' or 'json_object'.
        max_completion_tokens (int): Maximum number of tokens for the response.
    """
    def __init__(
        self,
        system_prompt,
        model="gpt-4o-mini",
        temperature=1.0,
        response_format="text",
        max_completion_tokens=2048
    ):
        self.messages = [Message('system', system_prompt).get_dict()]
        self.model = model
        self.temperature = temperature
        self.response_format = response_format
        self.max_completion_tokens = max_completion_tokens

    def receive_output(self, input):
        """
        Sends the user input to the OpenAI API and returns the assistant's response.

        Args:
            input (str): User input message.

        Returns:
            str: Assistant's response message.
        """
        # Add user's message to chat history
        self.messages.append(Message('user', input).get_dict())

        # Send request to OpenAI API
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            response_format={"type": self.response_format},
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            store=False
        )

        # Get assistant's reply
        output = response.choices[0].message.content

        # Add assistant's message to chat history
        self.messages.append(Message('assistant', output).get_dict())
        return output