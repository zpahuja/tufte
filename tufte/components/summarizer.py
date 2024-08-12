from pydantic import BaseModel
import json
import logging
import pandas as pd
import warnings
from openai import OpenAI
from typing import Dict, List, Union
from .utils import read_dataframe

class sum_Output(BaseModel):
  dType: str
  mean: float
  Standard_Deviation: float
  Min: float
  Max: float
  Samples: list[float]
  num_unique_values: int
  Description : str
class all_Ouputs(BaseModel):
  Columns_Properties : list[sum_Output]


logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"



class Summarizer:
    def __init__(self, model: str = DEFAULT_OPENAI_MODEL) -> None:
        self.oai_model = model
        self.oai_client = OpenAI()

    def _get_column_properties(self, df: pd.DataFrame, n_samples: int = 3) -> List[Dict]:
        properties_dict = {}

        def add_date_properties(column):
            try:
                properties["min"] = df[column].min().isoformat()
                properties["max"] = df[column].max().isoformat()
            except TypeError:
                cast_date_col = pd.to_datetime(df[column], errors="coerce")
                properties["min"] = cast_date_col.min().isoformat()
                properties["max"] = cast_date_col.max().isoformat()

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
        SYSTEM_PROMPT = """
        You are an experienced data analyst. You have been tasked to summarize a dataset given statistics where each key is the field name or column. You will be given statistics of this dataset in a JSON format. The dictionary contains all the relevant properties of the data.



        """.strip()
        logger.info("Enriching data properties using LLM")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(data_properties)},
        ]
        completion = self.oai_client.beta.chat.completions.parse(
          model = self.oai_model,
          messages=messages,
          response_format=all_Ouputs
        )



        
        response = completion.choices[0].message.parsed 

        return response



    def summarize(self, data: Union[pd.DataFrame, str], n_samples: int = 3, enrich: bool = False) -> Dict:
        if isinstance(data, str) and data.endswith(".csv"):
            data = read_dataframe(data)
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame or a path to a CSV file")
        data_properties = self._get_column_properties(data, n_samples)
        if enrich:
          return self._enrich(data_properties)
        else:
          return {"fields": data_properties}


summ = Summarizer()

summ1 = summ.summarize('Titanic-Dataset.csv', enrich = True)

print(summ1)
print(summ1.Columns_Properties[0].Description)
