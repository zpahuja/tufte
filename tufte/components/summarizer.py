import json
import logging
import pandas as pd
import warnings
from openai import OpenAI
from typing import Dict, List, Union

logger = logging.getLogger(__name__)

OPENAI_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = """
You are an experience data analyst. You have been tasked to summarize a dataset so we can understand the dataset and tasks that can be performed on it.

Response in the following JSON format:
{{
    field_name: {
        "description: "A brief description of the field",
        "semantc_type": "single word semantic type given its values, e.g. date, company, city, number, category, supplier, location, gender, longitude, latitude, url, ip address, zip code, email",
}}
""".strip()


class Summarizer:
    def __init__(self) -> None:
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
                        "std": convert_np_dtype(df[column].std(), dtype),
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
            model=OPENAI_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )
        property_descriptions = json.loads(response.choices[0].message.content)
        enriched_properties = {
            key: {**data_properties.get(key, {}), **property_descriptions.get(key, {})}
            for key in set(data_properties) | set(property_descriptions)
        }
        return enriched_properties

    def summarize(self, data: Union[pd.DataFrame, str], n_samples: int = 3) -> Dict:
        if isinstance(data, str) and data.endswith(".csv"):
            data = pd.read_csv(data)
        data_properties = self._get_column_properties(data, n_samples)
        data_summary = self._enrich(data_properties)
        return data_summary
