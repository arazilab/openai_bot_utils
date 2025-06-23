import json
import math
from collections import Counter
from bot_utils.core import Bot

def shannon_entropy(probabilities):
    """
    Calculates Shannon entropy for a list of probabilities.

    Args:
        probabilities (list of float): Probabilities summing to 1.

    Returns:
        float: Entropy value.
    """
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


class BooleanConsensusAgent:
    """
    Agent that answers a yes/no question based on voting by multiple GPT outputs.

    You can specify:
    - `n_votes`: fixed number of responses and return the majority.
    - `target_confidence`: desired confidence level based on entropy and continue until it's met.

    Attributes:
        question (str): Yes/no question to be answered.
        n_votes (int): Number of model calls (used for majority voting).
        target_confidence (float): Required confidence threshold.
        min_votes (int): Minimum votes before checking entropy.
        max_votes (int): Max number of calls before giving up in confidence mode.
        model (str): OpenAI model name to use.
    """

    def __init__(self, question, n_votes=None, target_confidence=None, min_votes=3, max_votes=11, model="gpt-4.1-nano"):
        if (n_votes is not None and target_confidence is not None) or (n_votes is None and target_confidence is None):
            raise ValueError("Specify exactly one: n_votes or target_confidence")

        if target_confidence is not None and (min_votes < 3 or max_votes <= min_votes or max_votes % 2 == 0):
            raise ValueError("For confidence mode: min_votes >= 3 and max_votes must be an odd number greater than min_votes")

        self.question = question
        self.n_votes = n_votes
        self.target_confidence = target_confidence
        self.min_votes = min_votes
        self.max_votes = max_votes
        self.model = model

        # System prompt used for all responses
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

    def __call__(self, datapoint):
        """
        Run the agent on a data input and return the decision (True or False).

        Args:
            datapoint (str): The user input or data to use in the decision.

        Returns:
            bool: Final decision based on voting.
        """
        if self.n_votes:
            return self._run_majority_vote(datapoint)
        else:
            return self._run_confidence_vote(datapoint)

    def _run_majority_vote(self, datapoint):
        """
        Run majority vote using a fixed number of calls.

        Args:
            datapoint (str): Input to be evaluated.

        Returns:
            bool: Result from majority of responses.
        """
        votes = []
        bot = Bot(system_prompt=self.system_prompt, model=self.model, response_format="json_object", memory=False)

        for _ in range(self.n_votes):
            response = bot.receive_output(datapoint)
            try:
                response = json.loads(response_str.strip())
                decision = response.get("answer", False)
                if isinstance(decision, bool):
                    votes.append(decision)
            except Exception:
                continue

            # Early majority check
            c = Counter(votes)
            if c[True] >= (self.n_votes // 2) + 1:
                return True
            if c[False] >= (self.n_votes // 2) + 1:
                return False

        # Fallback to majority
        return Counter(votes).most_common(1)[0][0]

    def _run_confidence_vote(self, datapoint):
        """
        Run vote loop until confidence threshold is met or max_votes reached.

        Args:
            datapoint (str): Input to be evaluated.

        Returns:
            bool: Result from most confident majority.
        """
        votes = []
        bot = Bot(system_prompt=self.system_prompt, model=self.model, response_format="json_object", memory=False)

        for _ in range(self.max_votes):
            response = bot.receive_output(datapoint)
            try:
                response = json.loads(response_str.strip())
                decision = response.get("answer", False)
                if isinstance(decision, bool):
                    votes.append(decision)
            except Exception:
                continue

            # Check confidence if we have enough votes
            if len(votes) >= self.min_votes:
                c = Counter(votes)
                total = sum(c.values())
                probs = [v / total for v in c.values()]
                entropy = shannon_entropy(probs)
                confidence = 1 - entropy

                if confidence >= self.target_confidence:
                    return c.most_common(1)[0][0]

        # Return majority after exhausting all attempts
        return Counter(votes).most_common(1)[0][0]
