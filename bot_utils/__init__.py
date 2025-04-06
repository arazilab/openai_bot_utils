"""
bot_utils

A lightweight module for building chat-based bots using OpenAI's API.
Provides the Bot and Message classes with support for message history
and customizable generation parameters.
"""

from .core import Bot, Message

__all__ = ["Bot", "Message"]