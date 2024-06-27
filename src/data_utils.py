import os
# import re
# import gzip

import numpy as np
import pandas as pd
# from datetime import datetime


# Read data from BigQuery(sql) into pandas dataframes.
def run_query(query, project_id):
  os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
  return pd.io.gbq.read_gbq(
      query,
      project_id=project_id,
      dialect='standard')

