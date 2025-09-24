import json
import math
from collections import Counter
from tqdm import trange
from bot_utils.core import Bot


def shannon_entropy(probabilities):
    """Shannon entropy in bits."""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


class BooleanConsensusAgent:
    """
    Yes/no agent that votes across multiple model calls.

    Use either:
      - n_votes: fixed number of calls, return majority.
      - target_confidence: keep calling until 1 - entropy >= threshold.
    """

    def __init__(
        self,
        question: str,
        n_votes: int | None = None,
        target_confidence: float | None = None,
        min_votes: int = 3,
        max_votes: int = 11,
        model: str = "gpt-5",
    ):
        if (n_votes is not None and target_confidence is not None) or \
           (n_votes is None and target_confidence is None):
            raise ValueError("Specify exactly one of n_votes or target_confidence")

        if target_confidence is not None and (min_votes < 3 or max_votes <= min_votes or max_votes % 2 == 0):
            raise ValueError("For confidence mode: min_votes >= 3 and max_votes must be odd and > min_votes")

        self.question = question
        self.n_votes = n_votes
        self.target_confidence = target_confidence
        self.min_votes = min_votes
        self.max_votes = max_votes
        self.model = model

        # System prompt for JSON boolean output
        self.system_prompt = (
            "Answer the yes/no question based on the given data and return JSON with one key 'answer'.\n"
            f"Question: {question}\n\n"
            "# Steps\n"
            "1. Analyze the provided data.\n"
            "2. Decide True for yes or False for no.\n\n"
            "# Output Format\n"
            '{\"answer\": true} or {\"answer\": false}\n'
        )

    def __call__(self, datapoint: str) -> bool:
        if self.n_votes is not None:
            return self._run_majority_vote(datapoint)
        return self._run_confidence_vote(datapoint)

    def _new_bot(self) -> Bot:
        # JSON mode maps to Responses API text.format.type='json' in the refactored Bot
        return Bot(
            system_prompt=self.system_prompt,
            model=self.model,
            response_format="json_object",
            memory=False,
            temperature=0.2,  # steadier votes
            reasoning_effort="minimal",
            verbosity="low",
            store=False,
        )

    def _parse_decision(self, response_str: str) -> bool | None:
        try:
            data = json.loads(response_str.strip())
            val = data.get("answer", None)
            if isinstance(val, bool):
                return val
            # allow "true"/"false" as strings, just in case
            if isinstance(val, str):
                v = val.strip().lower()
                if v in ("true", "false"):
                    return v == "true"
        except Exception:
            pass
        return None

    def _run_majority_vote(self, datapoint: str) -> bool:
        votes: list[bool] = []
        bot = self._new_bot()
        needed = (self.n_votes // 2) + 1

        pbar = trange(self.n_votes, desc="Voting", leave=False)
        for i in pbar:
            pbar.set_description(f"Voting round {i + 1}/{self.n_votes}")
            response_str = bot.receive_output(datapoint)
            decision = self._parse_decision(response_str)
            if decision is not None:
                votes.append(decision)

            c = Counter(votes)
            if c[True] >= needed:
                pbar.set_description("Early stop. Majority True")
                return True
            if c[False] >= needed:
                pbar.set_description("Early stop. Majority False")
                return False

        if not votes:
            # Fallback if all parses failed
            return False
        return Counter(votes).most_common(1)[0][0]

    def _run_confidence_vote(self, datapoint: str) -> bool:
        votes: list[bool] = []
        bot = self._new_bot()

        pbar = trange(self.max_votes, desc="Confidence loop", leave=False)
        for i in pbar:
            pbar.set_description(f"Call {i + 1}/{self.max_votes}")
            response_str = bot.receive_output(datapoint)
            decision = self._parse_decision(response_str)
            if decision is not None:
                votes.append(decision)

            if len(votes) >= self.min_votes:
                c = Counter(votes)
                total = sum(c.values())
                probs = [v / total for v in c.values()] if total else [0.5, 0.5]
                entropy = shannon_entropy(probs)  # max 1.0 for binary
                confidence = 1 - entropy
                pbar.set_description(
                    f"n={total} True={c[True]} False={c[False]} conf={confidence:.3f}"
                )
                if confidence >= self.target_confidence:
                    return c.most_common(1)[0][0]

        if not votes:
            return False
        return Counter(votes).most_common(1)[0][0]
