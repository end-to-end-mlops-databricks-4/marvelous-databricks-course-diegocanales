# Databricks notebook source

# !pip install -e ..
# %restart_python

# COMMAND ----------

# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------
# %load_ext autoreload
# %autoreload 2

# COMMAND ----------
from loguru import logger
from pyspark.sql import SparkSession

from mlops_course.config import ProjectConfig
from mlops_course.data.ingestion import load_weather_data_sample
from mlops_course.data_processor import DataProcessor

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

df = load_weather_data_sample(data_sample_path="../tests/test_data/sample.csv")
df.head()

# COMMAND ----------

data_processor = DataProcessor(df, config, spark)
data_processor.preprocess()

logger.info("Data preprocessing is completed.")

# COMMAND ----------
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# COMMAND ----------

logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

# COMMAND ----------
logger.info("Enable change data feed")
data_processor.enable_change_data_feed()

# COMMAND ----------
