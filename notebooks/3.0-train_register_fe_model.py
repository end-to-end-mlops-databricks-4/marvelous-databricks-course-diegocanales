# Databricks notebook source
## MAGIC %pip install mlops_course-0.0.2-py3-none-any.whl

# COMMAND ----------

## MAGIC %restart_python

# COMMAND ----------

# !pip install -e ..
# %restart_python

# COMMAND ----------

# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------

# Configure tracking uri
from loguru import logger
from pyspark.sql import SparkSession

from mlops_course.config import ProjectConfig, Tags
from mlops_course.models.feature_lookup_model import FeatureLookUpModel

# Configure tracking uri
# mlflow.set_tracking_uri("databricks")
# mlflow.set_registry_uri("databricks-uc")

spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "feat/fe-model"}
tags = Tags(**tags_dict)

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

# Create feature table
fe_model.create_feature_table()

# COMMAND ----------

# Define median temperature feature function
fe_model.define_feature_function()

# COMMAND ----------

# Load data
fe_model.load_data()

# COMMAND ----------

# Perform feature engineering
fe_model.feature_engineering()

# COMMAND ----------

# Train the model
fe_model.train()

# COMMAND ----------

# Register the model
latest_version = fe_model.register_model()
logger.info(f"Model registered with version: {latest_version}")

# COMMAND ----------

# Let's run prediction on the latest model
# Load test set from Delta table
spark = SparkSession.builder.getOrCreate()

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

# Drop target column for prediction
X_test = test_set.drop(config.target)

# COMMAND ----------

# Convert temperature columns to double type
from pyspark.sql.functions import col

# Convert all temperature columns (t00-t23) to double
for i in range(24):
    col_name = f"t{str(i).zfill(2)}"
    X_test = X_test.withColumn(col_name, col(col_name).cast("double"))

# COMMAND ----------

# Initialize model for prediction
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# Make predictions
predictions = fe_model.load_latest_model_and_predict(X_test)

# COMMAND ----------
logger.info(predictions)
