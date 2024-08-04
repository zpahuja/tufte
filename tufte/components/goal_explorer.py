import json
import logging
import openai
from typing import Dict, List

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """
As an experienced data analyst, you are tasked with generating insightful hypotheses and goals for the analysis and visualization of a dataset. Each goal you create should encompass the following components:

1. reasoning: Provide a justification for the insights expected from the visualization and specify which dataset fields will be utilized.
2. question: Pose a specific analytical question to be answered, such as 'What is the distribution of X?'
3. visualization: Recommend a type of visualization and identify the dataset fields to be visualized, for example, 'Histogram of Y, where Y is the exact field name from the dataset.'
4. statistic: Determine a key statistic to be calculated, like 'Mean of Z.'

The output should be structured as a list of dictionaries, each representing a goal. This list should be nested within a dictionary under the key 'goals'.

Please ensure that the dataset, summarized in JSON format where each key denotes a field name or column, is accurately represented in your goals. Follow visualization best practices, including Tufte's principles, and favor the use of bar charts over pie charts.
""".strip()

logger = logging.getLogger(__name__)


class GoalExplorer:
    def __init__(self, model: str = DEFAULT_OPENAI_MODEL) -> None:
        self.oai_model = model
        self.oai_client = openai.OpenAI()

    def generate_goals(self, summary: dict, n_goals: int = 5) -> List[Dict]:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Generate {n_goals} goals given the following data summary: {json.dumps(summary)}",
            },
        ]

        response = self.oai_client.chat.completions.create(
            model=self.oai_model,
            messages=messages,
            response_format={"type": "json_object"},
        )

        goals = json.loads(response.choices[0].message.content)
        assert "goals" in goals, "Expected 'goals' key in the response"
        assert isinstance(goals["goals"], list), "Expected a list of goals"
        assert (
            len(goals["goals"]) == n_goals
        ), f"Expected {n_goals} goals, but got {len(goals)} goals"
        return goals["goals"]
