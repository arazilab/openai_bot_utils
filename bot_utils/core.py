from openai import OpenAI

# Initialize OpenAI API client
client = OpenAI()


def _pack_content(role: str, text: str) -> dict:
    """
    Build a single message block for the Responses API.
    - developer/user use type 'input_text'
    - assistant uses type 'output_text'
    """
    if role == "assistant":
        return {"role": "assistant", "content": [{"type": "output_text", "text": text}]}
    elif role in ("developer", "user"):
        return {"role": role, "content": [{"type": "input_text", "text": text}]}
    else:
        raise ValueError("role must be 'developer', 'user', or 'assistant'")


def _extract_output_text(resp) -> str:
    """
    Safely get the text from Responses API result.
    Works for current SDKs that expose .output_text,
    and falls back to the raw structure.
    """
    # Preferred aggregate field if available
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text

    # Fallback: walk the first output item
    try:
        outs = getattr(resp, "output", None) or []
        if outs and "content" in outs[0] and outs[0]["content"]:
            first = outs[0]["content"][0]
            # Some SDKs use {"type": "output_text", "text": "..."}
            if first.get("type") in ("output_text", "text"):
                return first.get("text", "")
    except Exception:
        pass
    return ""


class Bot:
    """
    Wrapper for OpenAI Responses API with optional message history (memory).

    Args:
        system_prompt: initial instructions. Goes to role 'developer'.
        model: e.g. 'gpt-5'.
        temperature: randomness.
        response_format: 'text' or 'json_object' (mapped to text.format.type).
        max_completion_tokens: mapped to 'max_output_tokens'.
        memory: keep conversation history if True.
        store: set Responses API 'store' flag.
        include: list of fields for 'include' param.
        reasoning_effort: 'minimal' | 'medium' | 'high'.
        verbosity: 'low' | 'medium' | 'high' for text verbosity.
    """

    def __init__(
        self,
        system_prompt: str,
        model: str = "gpt-5",
        temperature: float = 1.0,
        response_format: str = "text",
        max_completion_tokens: int = 2048,
        memory: bool = False,
        store: bool = False,
        include: list | None = None,
        reasoning_effort: str = "minimal",
        verbosity: str = "medium",
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.response_format = response_format
        self.max_output_tokens = max_completion_tokens  # renamed for Responses API
        self.memory = memory
        self.store = store
        self.include = include or []
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity

        # Internal message buffer in Responses format
        self.messages: list[dict] = []
        if self.memory:
            self.messages.append(_pack_content("developer", self.system_prompt))

    def _format_block(self, role: str, text: str) -> dict:
        return _pack_content(role, text)

    def _text_format(self) -> dict:
        # Map old response_format to Responses text.format
        # 'text' -> {"type": "text"}; 'json_object' -> {"type": "json"}
        fmt_type = "text" if self.response_format != "json_object" else "json"
        return {"format": {"type": fmt_type}, "verbosity": self.verbosity}

    def receive_output(self, user_input: str) -> str:
        # Build the input list
        if self.memory:
            self.messages.append(self._format_block("user", user_input))
            input_blocks = self.messages
        else:
            input_blocks = [
                self._format_block("developer", self.system_prompt),
                self._format_block("user", user_input),
            ]

        # Create the response
        resp = client.responses.create(
            model=self.model,
            input=input_blocks,
            text=self._text_format(),
            reasoning={"effort": self.reasoning_effort},
            tools=[],                    # add tools if you use them
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            store=self.store,
            include=self.include,
        )

        output = _extract_output_text(resp)

        # If memory, save assistant turn too
        if self.memory:
            self.messages.append(self._format_block("assistant", output))

        return output
