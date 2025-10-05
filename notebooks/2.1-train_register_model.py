# Databricks notebook source
# COMMAND ----------
# %load_ext autoreload
# %autoreload 2

# COMMAND ----------
import os

import mlflow
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import SparkSession

from mlops_course.config import ProjectConfig, Tags
from mlops_course.models.sklearn_model import SklearnModel
from mlops_course.utils import is_databricks

# COMMAND ----------
if not is_databricks():
    logger.info("Databricks environment is not detected")
    load_dotenv()
    profile = os.environ.get("PROFILE", "DEFAULT")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")
    logger.info(f"Setting Databricks tracking and registry URIs for profile: {profile}")

# COMMAND ----------
logger.info("Loading project config")
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})


# COMMAND ----------
# Initialize model with the config path
logger.info("Initializing model")
model = SklearnModel(
    config=config,
    tags=tags,
    spark=spark,
)

# COMMAND ----------
# Load data and prepare features
logger.info("Loading data and preparing features")
model.load_data()
model.prepare_features()
logger.info("Data and features loaded and prepared")

# COMMAND ----------
# Train model and log model
logger.info("Training model")
model.train()
logger.info("Model trained")

# COMMAND ----------
logger.info("Logging model")
model.log_model()
logger.info("Model logged")

# COMMAND ----------
# Register model
logger.info("Registering model")
model.register_model()

logger.info("Model training and registration completed")

# COMMAND ----------
