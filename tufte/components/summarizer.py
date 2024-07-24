import logging
import pandas as pd

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
""".strip()


class Summarizer():
    """
    A class to summarize properties of dataframes and enrich summaries with descriptions using an LLM.
    """

    def __init__(self) -> None:
        """
        Initializes the Summarizer with an empty summary.
        """
        self.summary = None

    def get_column_properties(self, df: pd.DataFrame, n_samples: int = 3) -> list[dict]:
        """
        Extracts properties from each column of the dataframe such as data type, standard deviation, min, max values, etc.

        Args:
            df (pd.DataFrame): The dataframe from which to extract column properties.
            n_samples (int): Number of random samples to retrieve for each column.

        Returns:
            list[dict]: A list of dictionaries where each dictionary contains properties of a column.
        """
        properties_list = []
        for column in df.columns:
            dtype = df[column].dtype
            properties = {"dtype": str(dtype)}
            if dtype in [int, float, complex]:
                properties.update({"std": df[column].std(), "min": df[column].min(), "max": df[column].max()})
            elif dtype == object:
                try:
                    pd.to_datetime(df[column], errors='raise')
                    properties["dtype"] = "date"
                except ValueError:
                    properties["dtype"] = "category" if df[column].nunique() / len(df[column]) < 0.5 else "string"
            if properties["dtype"] == "date":
                properties.update({"min": df[column].min(), "max": df[column].max()})
            properties["samples"] = df[column].dropna().sample(n=min(n_samples, df[column].nunique()), random_state=42).tolist()
            properties["num_unique_values"] = df[column].nunique()
            properties_list.append({"column": column, "properties": properties})
        return properties_list

    def summarize(
            self, data: Union[pd.DataFrame, str],
            file_name="", n_samples: int = 3,
            encoding: str = 'utf-8') -> dict:
        """
        Summarizes data from a pandas DataFrame or a file location. This method constructs a summary based on the data properties.

        Args:
            data (Union[pd.DataFrame, str]): The data to summarize, either as a DataFrame or a file path.
            file_name (str): The name of the file if data is a file path.
            n_samples (int): Number of samples to include for each column in the summary.
            encoding (str): Encoding type if data is read from a file.

        Returns:
            dict: A dictionary containing the summary of the data.
        """
        # if data is a file path, read it into a pandas DataFrame, set file_name to the file name
        if isinstance(data, str):
            file_name = data.split("/")[-1]
            data = pd.read_csv(data, encoding=encoding)
        data_properties = self.get_column_properties(data, n_samples)
        base_summary = {"name": file_name, "file_name": file_name, "fields": data_properties}
        base_summary["field_names"] = data.columns.tolist()
        return base_summary
