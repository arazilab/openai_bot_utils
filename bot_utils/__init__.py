"""
bot_utils

A lightweight module for building chat-based bots using OpenAI's APIs.

This package exposes three objects:

- ``Bot``: a wrapper around either the Chat Completions API or the
  Responses API. It handles construction of message payloads,
  memory/state management and optional JSON output formatting.
- ``Message``: a simple data class representing a single chat message.
- ``BooleanConsensusAgent``: an agent that repeatedly queries a model
  to obtain a boolean verdict via majority or confidence-based
  voting.

The default behaviour uses the legacy chat completions API for
backwards compatibility. To migrate to the new responses API, set
``api_mode="responses"`` when instantiating a ``Bot`` or
``BooleanConsensusAgent``.
"""

from .core import Bot, Message
from .boolean_consensus_agent import BooleanConsensusAgent

__all__ = ["Bot", "Message", "BooleanConsensusAgent"]
