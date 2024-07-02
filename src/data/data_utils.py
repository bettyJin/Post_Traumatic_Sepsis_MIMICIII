import os

import numpy as np
import pandas as pd


# Read data from BigQuery(sql) into pandas dataframes.
def run_query(query, project_id):
  """
  Executes a SQL query on Google BigQuery and returns the result as a DataFrame.

  Args:
  - query (str): The SQL query to execute.
  - project_id (str): The Google Cloud project ID used for accessing BigQuery.

  Returns:
  - DataFrame: The result of the query as a pandas DataFrame.
  """
  os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
  return pd.io.gbq.read_gbq(
      query,
      project_id=project_id,
      dialect='standard')




