"""
llm_api - a tiny, dependency-free "large language model" for teaching purposes.

It mimics the shape of a real LLM API (`generate_text(model, prompt, max_tokens)`)
but under the hood it is a simple n-gram model trained on a small built-in corpus.
It shows the core idea of Generative AI: learn patterns from data, then predict
the next token given the context.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Final

logger = logging.getLogger(__name__)

__all__ = ["tokenize", "generate_text", "list_models", "DEFAULT_MODEL"]

DEFAULT_MODEL: Final[str] = "large-language-model"

Token = str
Context = tuple[Token, ...]

# Small training corpus: "learns patterns from datasets".
_CORPUS: Final[str] = """
The future of AI is a brighter place for everyone.
The future of AI is a brighter world for developers.
The future of AI is a brighter path to new ideas.
The future of AI is bright and full of possibilities.
The future of software is powered by AI assistants.
The future of code is written together with AI.
AI assistants help programmers write code faster.
AI models learn patterns from large datasets.
Generative AI creates new text, code, images, audio, and video.
Large language models predict the next token based on context.
A token is a small piece of text such as a word or part of a word.
Chatbots, image generators, and code assistants are examples of generative AI.
Developers use AI to write functions, tests, and documentation.
A brighter tomorrow starts with the tools we build today.
Learning how AI works helps us build better software.
"""

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\w+|[^\w\s]")
_PUNCTUATION_RE: Final[re.Pattern[str]] = re.compile(r"[^\w\s]")


def tokenize(text: str) -> list[Token]:
    """Split text into tokens (words and punctuation)."""
    return _TOKEN_RE.findall(text)


@dataclass
class NGramModel:
    """Predicts the next token from the previous `order - 1` tokens (with backoff)."""

    corpus: str
    order: int = 4
    # counts[context][next_token] -> frequency
    counts: dict[Context, dict[Token, int]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        tokens = tokenize(self.corpus)
        for n in range(1, self.order):  # context sizes 1 .. order-1
            for i in range(len(tokens) - n):
                context = tuple(t.lower() for t in tokens[i : i + n])
                next_token = tokens[i + n]
                bucket = self.counts.setdefault(context, {})
                bucket[next_token] = bucket.get(next_token, 0) + 1
        logger.debug(
            "Trained n-gram model: order=%d, contexts=%d", self.order, len(self.counts)
        )

    def next_token(self, tokens: list[Token], rng: random.Random) -> Token | None:
        """Back off from the longest seen context to the shortest until a match is found."""
        for n in range(min(self.order - 1, len(tokens)), 0, -1):
            context = tuple(t.lower() for t in tokens[-n:])
            candidates = self.counts.get(context)
            if candidates:
                words = list(candidates.keys())
                weights = list(candidates.values())
                return rng.choices(words, weights=weights, k=1)[0]
        return None


# Registry of available "models".
_MODELS: Final[dict[str, NGramModel]] = {DEFAULT_MODEL: NGramModel(_CORPUS)}


def list_models() -> list[str]:
    """Return the names of the available models."""
    return list(_MODELS)


def generate_text(
    model: str,
    prompt: str,
    max_tokens: int = 16,
    seed: int | None = None,
) -> str:
    """Generate up to `max_tokens` new tokens that continue `prompt`.

    Returns only the newly generated text (not the prompt), like a completion API.
    """
    lm = _MODELS.get(model)
    if lm is None:
        raise ValueError(f"Unknown model {model!r}. Available: {list_models()}")

    rng = random.Random(seed)
    tokens = tokenize(prompt)
    generated: list[Token] = []

    for _ in range(max_tokens):
        next_tok = lm.next_token(tokens, rng)
        if next_tok is None:
            break
        generated.append(next_tok)
        tokens.append(next_tok)

    # Re-join tokens: no space before punctuation.
    parts: list[str] = []
    for tok in generated:
        parts.append(tok if _PUNCTUATION_RE.fullmatch(tok) else f" {tok}")
    return "".join(parts).strip()
