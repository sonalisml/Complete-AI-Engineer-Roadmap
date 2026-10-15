import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#---------Load Tokenizer And Causal LM-----------------
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = autoModelForCausalLM.from_pretrained("distilgpt2")

#---------------Text-------------------
prompt = "Artificial Intelligence Will"

#-----Tokenize the prompt
inputs = tokenizer(
    prompt,
    return_tensors = "pt"
)
# -----------------------------------
# 1. Greedy decoding
# -----------------------------------
with torch.no_grad(
     greedy_output = model.generate(
         **inputs,
         max_new_tokens = 30,
         do_sample = False
    )
)
print("\n--- GREEDY ---")
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

# --------------------------------------------------
# 2. Sampling - temperature 0.3
# --------------------------------------------------

with torch.no_grad():
    low_temp_output = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.3,
        top_p=0.9
    )

print("\n--- SAMPLING: temperature=0.3 ---")
print(tokenizer.decode(low_temp_output[0], skip_special_tokens=True))

# --------------------------------------------------
# 3. Sampling - temperature 1.0
# --------------------------------------------------

with torch.no_grad():
    medium_temp_output = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=1.0,
        top_p=0.9
    )

print("\n--- SAMPLING: temperature=1.0 ---")
print(tokenizer.decode(medium_temp_output[0], skip_special_tokens=True))

# --------------------------------------------------
# 4. Sampling - temperature 1.5
# --------------------------------------------------

with torch.no_grad():
    high_temp_output = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=1.5,
        top_p=0.9
    )

print("\n--- SAMPLING: temperature=1.5 ---")
print(tokenizer.decode(high_temp_output[0], skip_special_tokens=True))