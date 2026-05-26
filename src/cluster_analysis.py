#@title Please input your project id
import pandas as pd
import numpy as np
import altair as alt
from google.cloud import bigquery
# Provide credentials to the runtime
from google.colab import auth
from google.cloud.bigquery import magics

auth.authenticate_user()
print('Authenticated')
project_id = 'akiba-356820' #@param {type: "string"}
# Set the default project id for %bigquery magic
magics.context.project = project_id

# Use the client to run queries constructed from a more complicated function.
client = bigquery.Client(project=project_id)
