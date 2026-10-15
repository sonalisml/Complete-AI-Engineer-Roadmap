
template = """
Classify the sentiment of the customer review.

Review: {review}

Choose exactly one label:
Positive
Negative
Neutral

Return only the label. Do not explain your answer.

Label:
"""

reviews = [
    "The product is excellent.",
    "The product stopped working after one day.",
    "The product arrived yesterday."
]

for review in reviews:
    prompt = template.format(review=review)
    print("Review:", review)
    print("Prompt:\n", prompt)
    print("-" * 40)
