import json
import logging
import pandas as pd
import warnings
from openai import OpenAI
from typing import Dict, List, Union

from .utils import read_dataframe

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """
You are an experienced data analyst. You have been tasked to summarize a dataset given statistics in a JSON format where each key is the field name or column.

Respond in JSON format as follows:
{{
    "description": "A brief description of the dataset",
    "fields": {{
        field_name: {{
            "description: "A brief description of the field",
            "semantc_type": "single word semantic type given its values, e.g. date, company, city, number, category, supplier, location, gender, longitude, latitude, url, zipcode, email",
        }}
    }}
}}
""".strip()


class Summarizer:
    def __init__(self, model: str = DEFAULT_OPENAI_MODEL) -> None:
        self.oai_model = model
        self.oai_client = OpenAI()

    def _get_column_properties(self, df: pd.DataFrame, n_samples: int = 3) -> List[Dict]:
        properties_dict = {}

        def add_date_properties(column):
            try:
                properties["min"] = df[column].min()
                properties["max"] = df[column].max()
            except TypeError:
                cast_date_col = pd.to_datetime(df[column], errors="coerce")
                properties["min"] = cast_date_col.min()
                properties["max"] = cast_date_col.max()

        def add_samples(column):
            non_null_values = df[column][df[column].notnull()].unique()
            n_samples_adjusted = min(n_samples, len(non_null_values))
            return (
                pd.Series(non_null_values)
                .sample(n_samples_adjusted, random_state=42)
                .tolist()
            )

        def convert_np_dtype(value, dtype: str):
            if "float" in str(dtype):
                return float(value)
            elif "int" in str(dtype):
                return int(value)
            return value

        for column in df.columns:
            dtype = df[column].dtype
            properties = {"dtype": str(dtype)}

            if dtype in [int, float, complex]:
                properties.update(
                    {
                        "dtype": "number",
                        "mean": round(float(df[column].mean()), 4),
                        "std": round(float(df[column].std()), 4),
                        "min": convert_np_dtype(df[column].min(), dtype),
                        "max": convert_np_dtype(df[column].max(), dtype),
                    }
                )
            elif dtype == bool:
                properties["dtype"] = "boolean"
            elif dtype == object:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(df[column], errors="raise")
                        properties["dtype"] = "date"
                except ValueError:
                    unique_ratio = df[column].nunique() / len(df[column])
                    properties["dtype"] = "category" if unique_ratio < 0.5 else "string"
            elif pd.api.types.is_categorical_dtype(df[column]):
                properties["dtype"] = "category"
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                properties["dtype"] = "date"

            if properties["dtype"] == "date":
                add_date_properties(column)

            properties["samples"] = add_samples(column)
            properties["num_unique_values"] = df[column].nunique()
            properties_dict[column] = properties

        return properties_dict

    def _enrich(self, data_properties: List[Dict]) -> Dict:
        logger.info("Enriching data properties using LLM")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(data_properties)},
        ]
        response = self.oai_client.chat.completions.create(
            model=self.oai_model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        enriched_descriptions = json.loads(response.choices[0].message.content)
        dataset_description, property_descriptions = (
            enriched_descriptions["description"],
            enriched_descriptions["fields"]
        )
        enriched_properties = {
            key: {**data_properties.get(key, {}), **property_descriptions.get(key, {})}
            for key in set(data_properties) | set(property_descriptions)
        }
        return {"description": dataset_description, "fields": enriched_properties}

    def summarize(self, data: Union[pd.DataFrame, str], n_samples: int = 3, enrich: bool = False) -> Dict:
        if isinstance(data, str) and data.endswith(".csv"):
            data = read_dataframe(data)
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame or a path to a CSV file")
        data_properties = self._get_column_properties(data, n_samples)
        return self._enrich(data_properties) if enrich else {"fields": data_properties}
