"""
Module: src.llm_utils

This module contains utility functions for interacting with LLMs using the Litellm API.

Functions:
- completion_with_backoff(model_full: str, prompt: str, max_tokens: int, use_temperature: bool, use_high_reasoning: bool): Call the LLM API with exponential backoff on failures.

Authors: Hokyung (Andy) Lee
Emails: techandy42@gmail.com (H. Lee)
Date: May 31, 2025
"""


from tenacity import retry, wait_exponential, stop_after_attempt
from litellm import text_completion
import tiktoken


def get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    tokens = encoder.encode(text)
    return len(tokens)


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5)
)
def completion_with_backoff(model_full: str, prompt: str, max_tokens: int, use_temperature: bool, use_high_reasoning: bool):
    """Call the LLM API with exponential backoff on failures."""
    params = {
        'model': model_full,
        'prompt': prompt,
        'max_tokens': max_tokens,
    }
    if use_temperature:
        params['temperature'] = 0.0
    if use_high_reasoning:
        del params['max_tokens']
        params['reasoning_effort'] = 'high'
    return text_completion(**params)
