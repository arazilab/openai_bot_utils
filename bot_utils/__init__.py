"""
bot_utils

Lightweight helpers for building bots on OpenAI's Responses API.
Exports:
- Bot: wrapper around client.responses.create with optional memory
- BooleanConsensusAgent: repeated-vote yes/no decision helper
"""

from .core import Bot
from .boolean_consensus_agent import BooleanConsensusAgent

__all__ = ["Bot", "BooleanConsensusAgent"]
