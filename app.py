import argparse

import llm_api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny Generative AI demo (n-gram model).")
    parser.add_argument(
        "--prompt",
        default="The future of AI is a brighter",
        help="Input text prompt.",
    )
    parser.add_argument(
        "--model",
        default=llm_api.DEFAULT_MODEL,
        choices=llm_api.list_models(),
        help="Which model to use.",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=8,
        help="How many extra tokens to generate for the full-sentence bonus.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed, for repeatable output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # LLM predicts the next token based on context
    response = llm_api.generate_text(
        model=args.model,
        prompt=args.prompt,
        max_tokens=1,
        seed=args.seed,
    )
    print(f"Next token generated: {response}")

    # --- Bonus: keep predicting one token at a time (this is how text is generated) ---
    text = args.prompt
    for _ in range(args.tokens):
        token = llm_api.generate_text(model=args.model, prompt=text, max_tokens=1, seed=args.seed)
        if not token:
            break
        text += token if token in ".,;:!?" else f" {token}"
    print(f"Full generated text: {text}")


if __name__ == "__main__":
    main()
