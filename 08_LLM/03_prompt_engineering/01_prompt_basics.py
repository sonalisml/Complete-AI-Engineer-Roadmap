import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()


def generate_text(prompt, max_new_tokens=60):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )


# Prompt 1: Vague instruction
prompt_1 = "Explain AI."

# Prompt 2: Specific instruction
prompt_2 = """
Explain artificial intelligence to a beginner.
Use simple English.
Give three practical applications.
Keep the answer concise.
"""

print("PROMPT 1")
print(generate_text(prompt_1))

print("\nPROMPT 2")
print(generate_text(prompt_2))