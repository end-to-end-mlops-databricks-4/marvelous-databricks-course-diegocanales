import os

import duckdb
import pandas as pd
from dotenv import find_dotenv, load_dotenv


def setup_duckdb_s3_connection():
    """Set up DuckDB connection with S3 credentials from environment variables.

    Returns:
        DuckDB connection object

    """
    load_dotenv(find_dotenv())

    conn = duckdb.connect()
    conn.execute(f"""
        SET s3_access_key_id='{os.getenv("AWS_ACCESS_KEY_ID")}';
        SET s3_secret_access_key='{os.getenv("AWS_SECRET_ACCESS_KEY")}';
        SET s3_endpoint='{os.getenv("AWS_ENDPOINT_URL")}';
    """)

    return conn


def load_weather_data(id_region: int, id_city: int, id_station: int, conn=None) -> pd.DataFrame:
    """Load weather data from S3 for a specific station.

    Args:
        id_region: Region ID
        id_city: City ID
        id_station: Station ID
        conn: Optional DuckDB connection (will create one if not provided)

    Returns:
        DataFrame with raw weather data

    """
    if conn is None:
        conn = setup_duckdb_s3_connection()

    query = f"""
        SELECT *
        FROM 's3://portfolio-projects-data/weather-stations/agromet/processed/*/id_region={id_region}/id_city={id_city}/id_station={id_station}/start_date=*_end_date=*.parquet'
    """

    df = conn.execute(query).df()
    return df


def load_weather_data_sample(data_sample_path: str) -> pd.DataFrame:
    """Load weather data from a sample file.

    Args:
        data_sample_path: Path to the sample file

    Returns:
        DataFrame with weather data

    """
    df = pd.read_csv(data_sample_path)
    return df
