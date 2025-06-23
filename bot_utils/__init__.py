"""
bot_utils

A lightweight module for building chat-based bots using OpenAI's API.
Provides the Bot and Message classes with support for message history
and customizable generation parameters. Includes a voting-based BooleanConsensusAgent
for structured yes/no decisions with confidence control.
"""

from .core import Bot, Message
from .boolean_consensus_agent import BooleanConsensusAgent

__all__ = ["Bot", "Message", "BooleanConsensusAgent"]
