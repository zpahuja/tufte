import logging
import pandas as pd
import random
import openai
import getpass
import os

os.environ['OPENAI_API_KEY'] = 'sk-proj-vmQlTmyyk72uy3RAjWPbT3BlbkFJaJSXwGlLEZ1UNXZIiU06'




logger = logging.getLogger(__name__)



class Summarizer():
  def __init__(self) -> None:
        self.summary = None

  def enrich(self, summary: dict) -> dict:
    full_summary = summary
    system_prompt = f"""
    You are a data analyst. You are given a pandas dataframe that you have to summarize

    Here is the dataset

    {summary}

    """
    messages = [{
    "role": "system",
    "content": system_prompt
    }, {"role": "user", "content": "Summarize the dataset"}]
    model = openai.OpenAI()

    summ = model.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    )

    full_summary.update({'AI_Summarization' : str(summ.choices[0].message.content)})
    return full_summary

  def get_column_properties(self, df: pd.DataFrame, n_samples: int = 3) -> list[dict]:
    properties_l = []

    for col in df.columns:
      dataType = df[col].dtype
      col_properties = {'Column_Name' : col, 'datatype' : dataType}
      if dataType == object:
        try:
          pd.to_datetime(df[col])
          col_properties["dtype"] = "date"
          col_properties.update({"Minimum": df[col].min(), "Maximum": df[col].max()})
        except ValueError:
          if df[col].nunique() / len(df[col]) >= 0.75:
            col_properties["dtype"] = "string"
          else:
            col_properties["dtype"] = 'catagorical'
      elif dataType in [int, float]:
        standardDeviation = df[col].std()
        Min = df[col].min()
        Max = df[col].max()
        col_properties.update({'Standard Deviation' : standardDeviation, 'Minimum' : Min, 'Maximum' : Max})
      col_properties.update({'Random_Samples' : [df[col][random.randint(0, len(df[col]) - 1)] for i in range(n_samples)]})
      properties_l.append(col_properties)
    return(properties_l)
  def summarize(self, data, file_name="", n_samples: int = 5):
    if isinstance(data, str):
      name = data.split("/")[-1]
      data = pd.read_csv(data)
    properties = self.get_column_properties(data, n_samples)
    summary = {"name": file_name, "file_name": file_name, "fields": properties}
    summary["field_names"] = data.columns.tolist()

    summ = self.enrich(summary)

    return summ







summarizer = Summarizer()
csv_file_path = "Titanic-Dataset.csv"
summary = summarizer.summarize(
    data=csv_file_path,
    file_name = 'Titanic'
)

print(summary)

print(summary['AI_Summarization'])
