"""Spark configuration for testing."""

from dataclasses import dataclass


@dataclass
class SparkConfig:
    """Spark configuration for testing."""

    master: str = "local[*]"
    app_name: str = "mlops_course_test"
    spark_executor_cores: str = "1"
    spark_executor_instances: str = "1"
    spark_sql_shuffle_partitions: str = "1"
    spark_driver_bindAddress: str = "127.0.0.1"


spark_config = SparkConfig()
