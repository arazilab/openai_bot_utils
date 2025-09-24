"""
boolean_consensus_agent.py - Voting-based yes/no decision helper.

This module defines the ``BooleanConsensusAgent``, an agent that
answers a yes/no question by making multiple calls to an underlying
``Bot`` instance and aggregating the results. The agent can operate
either in fixed-vote mode or confidence-based mode. It supports both
the legacy Chat Completions API and the newer Responses API via the
``api_mode`` parameter, allowing seamless migration while preserving
backwards compatibility.

Example usage:

```
from bot_utils.boolean_consensus_agent import BooleanConsensusAgent

# Majority vote using the responses API
agent = BooleanConsensusAgent(
    question="Is the sky blue?",
    n_votes=5,
    model="gpt-4o",
    api_mode="responses",
)
decision = agent("The sky appears azure today.")
print(decision)  # True or False

# Confidence-based voting using the chat completions API
agent2 = BooleanConsensusAgent(
    question="Is water wet?",
    target_confidence=0.8,
    min_votes=3,
    max_votes=7,
    model="gpt-3.5-turbo",
    api_mode="chat_completions",
)
decision2 = agent2("A description of water's properties...")
print(decision2)
```

The agent ensures the prompt instructs the model to output a JSON
object with a single key ``answer`` whose value is either ``true`` or
``false``. It then parses each response, counts the boolean votes and
either returns the majority or continues until the entropy-based
confidence threshold is met.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Optional

from .core import Bot


def shannon_entropy(probabilities: list[float]) -> float:
    """
    Calculate the Shannon entropy for a list of probabilities.

    Args:
        probabilities: A list of probabilities that sum to 1.0.

    Returns:
        The entropy value in bits.
    """
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


class BooleanConsensusAgent:
    """
    Agent that answers a yes/no question by aggregating multiple model calls.

    The agent can either collect a fixed number of votes (``n_votes``)
    and return the majority result or continue sampling until the
    confidence (based on the entropy of the votes) exceeds a target
    threshold.

    Parameters:
        question: The yes/no question to be answered. Should be
            provided as a statement; the model will return ``True`` if
            the answer is yes and ``False`` otherwise.
        n_votes: If set, a fixed number of model calls will be made and
            the majority result returned. Cannot be specified together
            with ``target_confidence``.
        target_confidence: If set, the agent will continue sampling
            until the confidence (1 - entropy) exceeds this threshold
            or the maximum number of votes is reached. Cannot be
            specified together with ``n_votes``.
        min_votes: Minimum number of votes to collect before
            evaluating confidence in confidence-based mode. Must be at
            least 3 and less than ``max_votes``.
        max_votes: Maximum number of votes to collect in
            confidence-based mode. Must be an odd number greater than
            ``min_votes``.
        model: The OpenAI model to use for each call.
        api_mode: Which API endpoint to use: 'chat_completions' or
            'responses'. Defaults to 'chat_completions' for backwards
            compatibility.
    """

    def __init__(
        self,
        question: str,
        n_votes: Optional[int] = None,
        target_confidence: Optional[float] = None,
        min_votes: int = 3,
        max_votes: int = 11,
        model: str = "gpt-4.1-nano",
        api_mode: str = "chat_completions",
    ) -> None:
        # Validate mutually exclusive parameters
        if (n_votes is not None and target_confidence is not None) or (
            n_votes is None and target_confidence is None
        ):
            raise ValueError("Specify exactly one of n_votes or target_confidence")

        if target_confidence is not None and (
            min_votes < 3 or max_votes <= min_votes or max_votes % 2 == 0
        ):
            raise ValueError(
                "In confidence mode: min_votes >= 3 and max_votes must be an odd number greater than min_votes"
            )

        self.question = question
        self.n_votes = n_votes
        self.target_confidence = target_confidence
        self.min_votes = min_votes
        self.max_votes = max_votes
        self.model = model
        self.api_mode = api_mode

        # System prompt instructing the model to answer in JSON
        self.system_prompt = f"""Answer the yes/no question based on the given data and return the response in JSON format with one key 'answer'.
Question: {question}

# Steps

1. Analyze the provided data to understand the context and details needed to answer the question.
2. Determine the correct answer based on the information available in the data.
3. Decide on 'True' if the answer to the question is yes or 'False' if the answer is no.

# Output Format

Return the answer in the following JSON format:

```json
{{"answer": [True/False]}}
```

Replace [True/False] with the determined boolean value."""

    def __call__(self, datapoint: str) -> bool:
        """
        Run the agent on a given data input and return the boolean decision.

        Args:
            datapoint: The user input or context to use for answering the question.

        Returns:
            ``True`` or ``False`` based on majority or confidence criteria.
        """
        if self.n_votes is not None:
            return self._run_majority_vote(datapoint)
        else:
            return self._run_confidence_vote(datapoint)

    def _run_majority_vote(self, datapoint: str) -> bool:
        """
        Run a fixed number of model calls and return the majority decision.

        Args:
            datapoint: The input string to evaluate.

        Returns:
            The majority boolean decision from the collected votes.
        """
        votes: list[bool] = []
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
                decision = response.get("answer", False)
                if isinstance(decision, bool):
                    votes.append(decision)
            except Exception as exception:
                # Print the exception and current votes for debugging
                print(exception)
                print(f"Votes so far:\n{votes}\n")

            # Early termination if majority already reached
            counts = Counter(votes)
            if counts[True] >= (self.n_votes // 2) + 1:
                return True
            if counts[False] >= (self.n_votes // 2) + 1:
                return False

        # Fallback: return the most common result
        return Counter(votes).most_common(1)[0][0]

    def _run_confidence_vote(self, datapoint: str) -> bool:
        """
        Continue sampling until confidence threshold is met or max_votes is reached.

        Args:
            datapoint: The input string to evaluate.

        Returns:
            The boolean decision once the confidence criterion is satisfied.
        """
        votes: list[bool] = []
        bot = Bot(
            system_prompt=self.system_prompt,
            model=self.model,
            response_format="json_object",
            memory=False,
            api_mode=self.api_mode,
        )

        for _ in range(self.max_votes):
            response_str = bot.receive_output(datapoint)
            try:
                response = json.loads(response_str.strip())
                decision = response.get("answer", False)
                if isinstance(decision, bool):
                    votes.append(decision)
            except Exception as exception:
                print(exception)
                print(f"Votes so far:\n{votes}\n")

            # Evaluate confidence when enough votes collected
            if len(votes) >= self.min_votes:
                counts = Counter(votes)
                total = sum(counts.values())
                probs = [v / total for v in counts.values()]
                entropy = shannon_entropy(probs)
                confidence = 1 - entropy
                if confidence >= (self.target_confidence or 0):
                    return counts.most_common(1)[0][0]

        # Exhausted all votes: return majority result
        return Counter(votes).most_common(1)[0][0]
