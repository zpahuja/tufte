def sum_Output(BaseModel):
  dType : str
  mean : float
  Standard_Deviation : float
  Min : float
  Max : float
  Samples : list[float]
  num_unique_values : int
  Description : str



import logging
import pandas as pd
import re

logger = logging.getLogger(__name__)


def read_dataframe(filepath: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Read a dataframe from a given filepath.
    Sample 100,000 rows if it exceeds that limit.
    """
    file_extension = filepath.split('.')[-1]

    read_funcs = {
        'json': lambda: pd.read_json(filepath, orient='records'),
        'csv': lambda: pd.read_csv(filepath),
        'xls': lambda: pd.read_excel(filepath),
        'xlsx': lambda: pd.read_excel(filepath),
        'tsv': lambda: pd.read_csv(filepath, sep='\t'),
    }

    if file_extension not in read_funcs:
        raise ValueError('Unsupported file type')

    try:
        df = read_funcs[file_extension]()
    except Exception as e:
        logger.error(f"Failed to read file: {filepath}. Error: {e}")
        raise

    df.columns = [re.sub(r'[^0-9a-zA-Z_]', '_', col_name) for col_name in df.columns]

    if len(df) > 100000:
        logger.info(
            "Dataframe has more than 100,000 rows. We will sample 100,000 rows.")
        df = df.sample(1e5)

    return df
