# bot_utils/multiclass_consensus_agent.py

from __future__ import annotations
import json
from collections import Counter
from typing import Dict, List
from .core import Bot

class MultiClassConsensusAgent:
    """
    Agent that answers a multiple-choice question by aggregating votes from
    repeated model calls. The agent calls a Bot `n_votes` times, each time
    asking the model to select one of the provided options and return a JSON
    object with key `answer` and an integer value corresponding to the chosen
    option. It returns the option that receives the most votes.
    """

    def __init__(
        self,
        question: str,
        options: Dict[str, str],
        n_votes: int = 5,
        model: str = "gpt-4.1-nano",
        api_mode: str = "chat_completions",
    ) -> None:
        if n_votes < 1:
            raise ValueError("n_votes must be at least 1")
        if not options:
            raise ValueError("options must not be empty")

        self.question = question
        self.options = options
        self.n_votes = n_votes
        self.model = model
        self.api_mode = api_mode

        # Build option listing and maintain index mapping
        option_lines: List[str] = []
        for idx, (short, description) in enumerate(options.items(), start=1):
            option_lines.append(f"{idx}) {short}: {description}")
        options_str = "\n".join(option_lines)

        # Construct a detailed system prompt
        self.system_prompt = (
            f"Choose the best option for this question based on the given "
            f"data and return the response in JSON format with one key 'answer'.\n"
            f"Question: {question}\n\n"
            f"Options:\n{options_str}\n\n"
            "# Steps\n\n"
            "1. Analyze the provided data to understand the context and details "
            "needed to answer the question.\n"
            "2. Determine which option best addresses the question based on the "
            "information available in the data.\n"
            "3. Select the option by its number (1, 2, 3, …) that corresponds to "
            "the best choice.\n\n"
            "# Output Format\n\n"
            "Return the answer in the following JSON format:\n\n"
            "```json\n"
            "{\"answer\": <option:int>}\n"
            "```\n\n"
            "Replace <option:int> with the integer index of the chosen option."
        )

    def __call__(self, datapoint: str) -> int:
        """
        Query the model `n_votes` times and return the option index that
        receives the majority of votes. Early exits if a majority is reached.
        """
        votes: List[int] = []
        bot = Bot(
            system_prompt=self.system_prompt,
            model=self.model,
            response_format="json_object",
            memory=False,
            api_mode=self.api_mode,
        )

        for _ in range(self.n_votes):
            response_str = bot.receive_output(datapoint)
            try:
                response = json.loads(response_str.strip())
                answer = response.get("answer")
                if isinstance(answer, int):
                    votes.append(answer)
            except Exception:
                # Ignore malformed responses but print for debugging
                print("Malformed response:", response_str)
                print("Votes so far:", votes)

            # Early termination if any option already has a majority
            counts = Counter(votes)
            for opt_idx, count in counts.items():
                if count >= (self.n_votes // 2) + 1:
                    return opt_idx

        # Fallback: return the most common result if no majority was found
        if not votes:
            raise RuntimeError("No valid votes were collected from the model")
        return Counter(votes).most_common(1)[0][0]
