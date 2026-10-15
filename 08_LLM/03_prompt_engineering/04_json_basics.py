
import json

# Example LLM response
response = """
{
    "title": "Lung Cancer Classification",
    "method": "Convolutional Neural Network",
    "dataset": "CT scan images"
}
"""

# Convert JSON text into a Python dictionary
data = json.loads(response)

print("Title:", data["title"])
print("Method:", data["method"])
print("Dataset:", data["dataset"])

# Validate required keys
required_keys = ["title", "method", "dataset"]

for key in required_keys:
    if key not in data:
        raise ValueError(f"Missing required field: {key}")

print("\nAll required fields are present.")
