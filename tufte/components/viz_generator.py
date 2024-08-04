import json
import re
from openai import OpenAI
from typing import Dict

from .scaffold import Scaffold

GENERAL_INSTRUCTIONS_PROMPT = """
You are an expert data analyst and programmer. Your task is to generate python code for a visualization based on a dataset summary and a goal.
The code must follow visualization best practices, meet the specified goal, apply the right transformation, use the right visualization type, use the right data encoding, and use the right aesthetics.
Think step by step. First generate a brief plan for how you would solve the task e.g. what transformations you would apply, what fields you would use, what visualization type you would use, what aesthetics you would use, etc.
Then write a complete python program that generates the visualization. The program should be enclosed in backticks (```python) and should start with an import statement.
The program should only use data fields that exist in the dataset (field_names) or fields that are transformations based on existing field_names.
Generated code should be created by modifying the specified parts of the provided template.

If using a field where semantic_type is date, apply the following transform before using that column:
1. convert date fields to date types using pd.to_datetime(data[<field>], errors='coerce')
2. drop the rows with NaN values using data = data[pd.notna(data[<field>])]
3. convert the field to the right time format for plotting

Ensure that the x-axis labels are legible (e.g., rotate when needed).

Do not write code to load the data. The data is already loaded and available in the variable data.
""".strip()


class VizGenerator(object):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.oai_model = model
        self.oai_client = OpenAI()
        self.scaffold = Scaffold()

    def _extract_code(self, text: str):
        pattern = r'```python(.*?)```'
        code = re.findall(pattern, text, re.DOTALL)
        return code

    def generate_code(self, summary: Dict, goal: Dict, library: str = "altair"):
        code_template, additional_instructions = self.scaffold.get_template(library)
        messages = [
            {
                "role": "system", 
                "content": f"""
                {GENERAL_INSTRUCTIONS_PROMPT}
                plot(data) method should generate {goal['visualization']} using {library} that addresses this goal: {goal['question']}.
                {additional_instructions}
                Code template: {code_template}
                """
            },
            {
                "role": "system",
                "content": f"Dataset summary is : {json.dumps(summary)}"
            },
        ]

        completions = self.oai_client.chat.completions.create(
            model=self.oai_model,
            messages=messages,
            temperature=0,
        )
        response = completions.choices[0].message.content
        return self._extract_code(response)
