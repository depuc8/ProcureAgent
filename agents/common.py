"""Shared model + generation config for all agents."""

from __future__ import annotations

from typing import Callable, Union

from google.adk.models.base_llm import BaseLlm
from google.genai import types

from settings import get_settings

ModelLike = Union[str, BaseLlm]
ModelFactory = Callable[[str], ModelLike]


def default_model_factory(agent_name: str) -> ModelLike:
    """Every agent uses PROCURE_MODEL unless a factory says otherwise (tests inject scripted models)."""
    return get_settings().model


def gen_config(temperature: float = 0.2) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(temperature=temperature)


REACT_PREAMBLE = (
    "Work in a ReAct loop: briefly reason about what you still need, call exactly the tool that "
    "provides it, read the observation, and repeat until you can answer. Never answer from memory "
    "or invent figures - every number you report must come from a tool result."
)
