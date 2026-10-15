from pydantic import BaseModel, Field, ValidationError
import json

class ResearchPaper(BaseModel):
    title:str
    method:str
    accuracy: float = Field(ge=0, le = 100)
    dataset: str | None

# Simulated LLM response
response = """
{
    "title": "Lung Cancer Classification",
    "method": "CNN",
    "accuracy": 98.5,
    "dataset": null
}
"""

try:
    # Step 1: Parse JSON
    data = json.loads(response)

    # Step 2: Validate structure and types
    paper = ResearchPaper.model_validate(data)

    print("Title:", paper.title)
    print("Method:", paper.method)
    print("Accuracy:", paper.accuracy)
    print("Dataset:", paper.dataset)

except (json.JSONDecodeError, ValidationError) as error:
    print("Invalid response:", error)