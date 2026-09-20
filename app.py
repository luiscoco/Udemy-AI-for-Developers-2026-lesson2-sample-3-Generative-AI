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

# --- Bonus: keep predicting one token at a time (this is how text is generated) ---
text = prompt
for _ in range(8):
    token = llm_api.generate_text(model="large-language-model", prompt=text, max_tokens=1)
    if not token:
        break
    text += token if token in ".,;:!?" else f" {token}"
print(f"Full generated text: {text}")
