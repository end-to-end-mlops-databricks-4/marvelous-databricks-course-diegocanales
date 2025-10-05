# Databricks notebook source
## MAGIC %pip install mlops_course-0.0.2-py3-none-any.whl

# COMMAND ----------

## MAGIC %restart_python

# COMMAND ----------

# !pip install -e ..
# %restart_python

# COMMAND ----------

from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------
import os
import time

import requests
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from mlops_course.config import ProjectConfig
from mlops_course.serving.model_serving import ModelServing

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

w = WorkspaceClient()
os.environ["DBR_HOST"] = w.config.host
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=1200).token_value

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
# Initialize model serving
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.frost_prediction_model_basic", endpoint_name="frost-prediction-model-serving"
)


# COMMAND ----------
# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()  ## Wait for the endpoint to be ready

# COMMAND ----------
# Get the numerical features from the config
required_columns = config.num_features

# COMMAND ----------
# Sample data from the test set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# Sample records from the test set
sampled_records = test_set[required_columns].sample(n=10, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------
# Function to call the model serving endpoint
def call_endpoint(record) -> tuple[int, str]:
    """Call the model serving endpoint with a given input record."""
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/frost-prediction-model-serving/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text

# COMMAND ----------
# Test with one sample record
status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# Load test with multiple records
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2)
