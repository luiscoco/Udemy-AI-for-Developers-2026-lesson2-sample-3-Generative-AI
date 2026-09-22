# Generative AI

A small, runnable Python app that illustrates the core idea behind **Generative AI**: *predicting the next token* from a text prompt, based on patterns learned from data.

It follows the slide "3 – Generative AI: Creates new content", which shows this snippet:

```python
import llm_api

# Input text prompt
prompt = "The future of AI is a brighter"

# LLM predicts the next token based on context
response = llm_api.generate_text(
    model="large-language-model",
    prompt=prompt,
    max_tokens=1
)
print(f"Next token generated: {response}")
```

## Key concepts from the slide

- Generative AI **produces** text, code, images, audio, or video.
- It **learns patterns** from large datasets.
- Examples: chatbots, image generators, code assistants.
- A text prompt goes in, and the model generates new content by predicting the **next token**, one at a time.

## Project structure

| File | Purpose |
|------|---------|
| `app.py` | The application: the slide's snippet plus a small bonus loop, exposed as a small CLI (`argparse`). |
| `llm_api.py` | A local, dependency-free stand-in for the `llm_api` module the slide imports. |
| `README.md` | This document. |

## How to run

Requirements: Python 3.10+ (no external packages).

```bash
python app.py
```

Example output (it changes between runs, because the prediction is random):

```
Next token generated: path
Full generated text: The future of AI is a brighter path to new ideas. The future of
```

`app.py` also accepts CLI options (via `argparse`) to try different prompts without editing the file:

```bash
python app.py --prompt "AI assistants help" --tokens 10 --seed 42
```

| Flag | Meaning |
|------|---------|
| `--prompt` | The input text prompt. |
| `--model` | Which model to use (see `llm_api.list_models()`). |
| `--tokens` | How many extra tokens to generate in the bonus loop. |
| `--seed` | Random seed, for repeatable output. |

## Code explanation: `app.py`

### 1. Import the model API

```python
import llm_api
```

The slide uses a fictional `llm_api` package. In this sample it is the local file `llm_api.py`, so the app works without an internet connection or API key.

### 2. Parse CLI options

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny Generative AI demo (n-gram model).")
    parser.add_argument("--prompt", default="The future of AI is a brighter", ...)
    parser.add_argument("--model", default=llm_api.DEFAULT_MODEL, choices=llm_api.list_models(), ...)
    parser.add_argument("--tokens", type=int, default=8, ...)
    parser.add_argument("--seed", type=int, default=None, ...)
    return parser.parse_args()
```

Instead of hardcoding the prompt, the app uses [`argparse`](https://docs.python.org/3/library/argparse.html) (the standard library's CLI-parsing module) so you can try different inputs from the command line. See [How to run](#how-to-run) for the flag reference.

The prompt still defaults to `"The future of AI is a brighter"`, deliberately stopping mid-sentence so the model has something obvious to continue.

### 3. Ask the model for the next token

```python
response = llm_api.generate_text(
    model=args.model,
    prompt=args.prompt,
    max_tokens=1,
    seed=args.seed,
)
```

| Argument | Meaning |
|----------|---------|
| `model` | Which model to use. Here there is a single one, `"large-language-model"`. |
| `prompt` | The input text that provides the context. |
| `max_tokens=1` | Generate only **one** token, so we see a single prediction step. |
| `seed` | Optional; makes the random choice repeatable when set. |

The function returns only the newly generated text, not the prompt (like a real completion API).

### 4. Print the result

```python
print(f"Next token generated: {response}")
```

Prints the predicted token, for example `path`, `place` or `world`.

### 5. Bonus: generate a full sentence

```python
text = args.prompt
for _ in range(args.tokens):
    token = llm_api.generate_text(model=args.model, prompt=text, max_tokens=1, seed=args.seed)
    if not token:
        break
    text += token if token in ".,;:!?" else f" {token}"
print(f"Full generated text: {text}")
```

This is the key insight of LLMs: **long text is just repeated next-token prediction**. Each new token is appended to the text, and the longer text becomes the prompt for the next prediction (this is called *autoregressive* generation). The loop stops after `--tokens` iterations (8 by default) or if the model has nothing to predict.

### 6. Entry point

```python
if __name__ == "__main__":
    main()
```

Wrapping the logic in `main()` behind this guard is the standard Python idiom for a script that can also be imported without side effects.

## Code explanation: `llm_api.py`

This file is a toy language model. Real LLMs use huge neural networks (transformers), but the principle is the same: learn from data, then predict what comes next.

### Training corpus

```python
_CORPUS = """
The future of AI is a brighter place for everyone.
The future of AI is a brighter world for developers.
...
"""
```

A short built-in text acts as the "large dataset". The model can only produce patterns it has seen here.

### Tokenizer

```python
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")

def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)
```

A **token** is a small piece of text. Here the tokenizer splits text into words and punctuation marks, e.g. `"AI is bright."` → `["AI", "is", "bright", "."]`. Real LLMs use sub-word tokens, but the idea is the same.

### The n-gram model (`NGramModel`)

`NGramModel` is a [`@dataclass`](https://docs.python.org/3/library/dataclasses.html): it declares its fields (`corpus`, `order`, `counts`) instead of writing a manual `__init__`, and Python generates the constructor for us.

**Training** (`__post_init__`): a dataclass's post-init hook runs right after the generated constructor, so this is where training happens. For every position in the corpus, the model records which token followed each context of 1, 2 and 3 previous tokens, and how many times:

```
("a", "brighter")  ->  {"place": 1, "world": 1, "path": 1}
```

A `logging.debug(...)` call reports how many contexts were learned — run with `logging.basicConfig(level=logging.DEBUG)` to see it.

**Prediction** (`next_token`):

1. Take the last 3 tokens of the prompt as the context.
2. Look up which tokens followed that context in the corpus.
3. If the context was never seen, **back off** to a shorter one (2 tokens, then 1).
4. Pick one candidate at random, **weighted by frequency**: tokens seen more often are more likely.

This is a tiny version of what an LLM does: it produces a probability distribution over possible next tokens and samples from it. That is why the same prompt can give different answers on different runs.

### Model registry: `list_models()` and `DEFAULT_MODEL`

```python
_MODELS: Final[dict[str, NGramModel]] = {DEFAULT_MODEL: NGramModel(_CORPUS)}

def list_models() -> list[str]:
    """Return the names of the available models."""
    return list(_MODELS)
```

`DEFAULT_MODEL` names the single built-in model (`"large-language-model"`), and `list_models()` exposes the registry — `app.py` uses it to validate the `--model` CLI flag instead of hardcoding the name.

### `generate_text(model, prompt, max_tokens=16, seed=None)`

The public function used by `app.py`:

1. Looks up the `model` name in the registry and raises `ValueError` if it's unknown.
2. Tokenizes the prompt.
3. Repeats up to `max_tokens` times: predict the next token, append it to the context.
4. Joins the generated tokens back into a string (no space before punctuation) and returns it.

The optional `seed` parameter makes the randomness repeatable:

```python
llm_api.generate_text("large-language-model", "The future of AI is a brighter", 1, seed=42)
```

## Try it yourself

- Change `prompt` to `"Large language models predict"` or `"AI assistants help"`.
- Increase `max_tokens` to generate longer continuations.
- Add sentences to `_CORPUS` and see how the predictions change: the model only "knows" what is in its training data.
- Try a prompt with words not in the corpus: the model has no context and returns an empty string.

## Limitations

This is an educational toy, not a real LLM:

- It has no understanding of meaning, only word-sequence statistics.
- It can only reproduce patterns from its tiny corpus.
- Real LLMs (Claude, GPT, Gemini, ...) work at a vastly larger scale, but follow the same loop: **context in → probabilities for the next token → pick one → repeat**.

To use a real model, replace `llm_api.py` with a call to a provider SDK (this requires an API key).
