"""
TapeAgents - A framework for building LLM agents with structured, replayable logs.

TapeAgents leverages a structured, replayable log (Tape) of the agent session to 
facilitate all stages of the LLM Agent development lifecycle.
"""

# Core components
from .core import *
from .agent import Agent, Node
from .dialog_tape import DialogTape, UserStep, AssistantStep, SystemStep
from .llms import LLM, LiteLLM, LLMStream

# Autonomous learning components
from . import autonomous_learning

__version__ = "0.1.0"

__all__ = [
    # Core
    "Agent",
    "Node", 
    "DialogTape",
    "UserStep",
    "AssistantStep", 
    "SystemStep",
    "LLM",
    "LiteLLM",
    "LLMStream",
    # Autonomous learning
    "autonomous_learning"
]