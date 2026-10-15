
# A reusable prompt template

template = """
Act as an AI research assistant.

Task: Summarize the following research text.
Audience: A computer science researcher.
Constraints:
- Use clear, concise language.
- Identify the main contribution.
- Mention one limitation.

Research text:
{text}

Summary:
"""

# Example 1
paper_1 = """
Federated learning trains machine learning models across
multiple devices without collecting their raw data centrally.
"""

prompt_1 = template.format(text=paper_1)
print("PROMPT 1:")
print(prompt_1)

# Example 2
paper_2 = """
Vision transformers use self-attention to learn relationships
between image patches for image classification.
"""

prompt_2 = template.format(text=paper_2)
print("\nPROMPT 2:")
print(prompt_2)
