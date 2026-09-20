"""
llm_api - a tiny, dependency-free "large language model" for teaching purposes.

It mimics the shape of a real LLM API (`generate_text(model, prompt, max_tokens)`)
but under the hood it is a simple n-gram model trained on a small built-in corpus.
It shows the core idea of Generative AI: learn patterns from data, then predict
the next token given the context.
"""

import random
import re
from collections import Counter, defaultdict

# Small training corpus: "learns patterns from datasets".
_CORPUS = """
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

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    """Split text into tokens (words and punctuation)."""
    return _TOKEN_RE.findall(text)


class _NGramModel:
    """Predicts the next token from the previous `order - 1` tokens (with backoff)."""

    def __init__(self, corpus: str, order: int = 4):
        self.order = order
        # counts[context_tuple][next_token] -> frequency
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        tokens = tokenize(corpus)
        for n in range(1, order):  # context sizes 1 .. order-1
            for i in range(len(tokens) - n):
                context = tuple(t.lower() for t in tokens[i : i + n])
                self.counts[context][tokens[i + n]] += 1

    def next_token(self, tokens: list[str], rng: random.Random) -> str | None:
        # Back off from the longest context to the shortest until we find one.
        for n in range(min(self.order - 1, len(tokens)), 0, -1):
            context = tuple(t.lower() for t in tokens[-n:])
            candidates = self.counts.get(context)
            if candidates:
                words, freqs = zip(*candidates.items())
                return rng.choices(words, weights=freqs, k=1)[0]
        return None


# Registry of available "models".
_MODELS = {"large-language-model": _NGramModel(_CORPUS)}


def generate_text(
    model: str,
    prompt: str,
    max_tokens: int = 16,
    seed: int | None = None,
) -> str:
    """Generate up to `max_tokens` new tokens that continue `prompt`.

    Returns only the newly generated text (not the prompt), like a completion API.
    """
    if model not in _MODELS:
        raise ValueError(f"Unknown model '{model}'. Available: {list(_MODELS)}")

    lm = _MODELS[model]
    rng = random.Random(seed)
    tokens = tokenize(prompt)
    generated: list[str] = []

    for _ in range(max_tokens):
        nxt = lm.next_token(tokens, rng)
        if nxt is None:
            break
        generated.append(nxt)
        tokens.append(nxt)

    # Re-join tokens: no space before punctuation.
    text = ""
    for tok in generated:
        text += tok if re.fullmatch(r"[^\w\s]", tok) else f" {tok}"
    return text.strip()
