import numpy as np
import pandas as pd


def clean_temperature_range(
    df: pd.DataFrame, column: str = "average_air_temperature", min_val: float = -15, max_val: float = 45
) -> pd.DataFrame:
    """Replace temperature values outside acceptable range with NaN.

    Args:
        df: DataFrame containing temperature data
        column: Column name for temperature values
        min_val: Minimum acceptable temperature (°C)
        max_val: Maximum acceptable temperature (°C)

    Returns:
        DataFrame with cleaned temperature values

    """
    df = df.copy()
    df.loc[~df[column].between(min_val, max_val), column] = np.nan
    return df


def clean_humidity_range(
    df: pd.DataFrame,
    column: str = "average_relative_humidity",
    min_val: float = 0,
    max_val: float = 100,
) -> pd.DataFrame:
    """Replace humidity values outside acceptable range with NaN.

    Args:
        df: DataFrame containing humidity data
        column: Column name for humidity values
        min_val: Minimum acceptable humidity (%)
        max_val: Maximum acceptable humidity (%)

    Returns:
        DataFrame with cleaned humidity values

    """
    df = df.copy()
    df.loc[~df[column].between(min_val, max_val), column] = np.nan
    return df


def clean_radiation_range(
    df: pd.DataFrame,
    column: str = "max_solar_radiation",
    min_val: float = 0,
    max_val: float = 1300,
) -> pd.DataFrame:
    """Replace solar radiation values outside acceptable range with NaN.

    Args:
        df: DataFrame containing solar radiation data
        column: Column name for solar radiation values
        min_val: Minimum acceptable radiation (W/m²)
        max_val: Maximum acceptable radiation (W/m²)

    Returns:
        DataFrame with cleaned radiation values

    """
    df = df.copy()
    df.loc[~df[column].between(min_val, max_val), column] = np.nan
    return df


def interpolate_time_series(
    df: pd.DataFrame,
    columns: list[str],
    group_by: str = "id_station",
    time_col: str = "ts",
) -> pd.DataFrame:
    """Perform linear interpolation on time series data, grouped by station.

    Args:
        df: DataFrame containing time series data
        columns: List of columns to interpolate
        group_by: Column to group by (typically station ID)
        time_col: Column containing timestamps

    Returns:
        DataFrame with interpolated values

    """
    df = df.copy()
    df = df.sort_values([group_by, time_col])

    def _interp_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.set_index(time_col)
        g[columns] = g[columns].interpolate(method="time", limit_area="inside")
        return g.reset_index()

    return df.groupby(group_by, group_keys=False).apply(_interp_group)


def add_time_features(df: pd.DataFrame, timestamp_col: str = "ts") -> pd.DataFrame:
    """Add date and hour features from timestamp.

    Args:
        df: DataFrame containing timestamp data
        timestamp_col: Column name for timestamp

    Returns:
        DataFrame with added time features

    """
    df = df.copy()
    df["date"] = df[timestamp_col].dt.floor("D")
    df["hour"] = df[timestamp_col].dt.hour
    return df


def create_hourly_temperature_features(df: pd.DataFrame, temp_col: str = "average_air_temperature") -> pd.DataFrame:
    """Create hourly temperature features (t00-t23) from time series data.

    Args:
        df: DataFrame with temperature and hour columns
        temp_col: Column name for temperature values

    Returns:
        DataFrame with hourly temperature columns, indexed by station and date

    """
    temps = (
        df.pivot_table(
            index=["id_station", "date"],
            columns="hour",
            values=temp_col,
            aggfunc="mean",
        ).reindex(columns=range(24))  # ensure columns 0..23 exist
    )
    temps.columns = [f"t{h:02d}" for h in temps.columns]
    return temps


def create_frost_target(
    df: pd.DataFrame,
    temp_col: str = "min_temp_next_day",
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Create binary frost target variable based on minimum temperature.

    Args:
        df: DataFrame with minimum temperature column
        temp_col: Column name for minimum temperature
        threshold: Temperature threshold for frost in Celsius (default: 0.0)

    Returns:
        DataFrame with added frost target column

    """
    df = df.copy()
    df["frost_next_day"] = (df[temp_col] < threshold).astype(int)
    return df


def create_target_min_temp_next_day(daily_df: pd.DataFrame, temp_cols: list[str] = None) -> pd.DataFrame:
    """Create target variable: minimum temperature of the next day.

    Args:
        daily_df: DataFrame with hourly temperature columns
        temp_cols: List of temperature column names (t00-t23)

    Returns:
        DataFrame with added target variable

    """
    df = daily_df.copy()

    if temp_cols is None:
        temp_cols = [f"t{h:02d}" for h in range(24)]

    df["min_temp_today"] = df[temp_cols].min(axis=1)
    df["min_temp_next_day"] = df.groupby(level=0)["min_temp_today"].shift(-1)

    return df


def clean_and_filter_daily_features(df: pd.DataFrame, required_columns: list[str] = None) -> pd.DataFrame:
    """Clean daily features by removing rows with missing values in required columns.

    Args:
        df: DataFrame with daily features
        required_columns: List of columns that must not be null

    Returns:
        DataFrame with rows filtered by required columns

    """
    if required_columns is None:
        temp_cols = [f"t{h:02d}" for h in range(24)]
        required_columns = temp_cols + ["min_temp_next_day"]

    return df.dropna(subset=required_columns)


def select_and_order_columns(df: pd.DataFrame, include_keys: bool = False) -> pd.DataFrame:
    """Select and order columns for the final dataset.

    Args:
        df: DataFrame with all features
        include_keys: Whether to include index keys (id_station, date) as columns

    Returns:
        DataFrame with selected and ordered columns

    """
    temp_cols = [f"t{h:02d}" for h in range(24)]
    feature_cols = temp_cols + [
        "min_temp_next_day",  # TODO: remove this column
        "frost_next_day",
    ]

    result = df[feature_cols]

    if include_keys:
        result = result.reset_index()

    return result


def preprocess_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess weather station data to create daily features for modeling.

    Args:
        df: DataFrame with raw weather station data

    Returns:
        DataFrame with processed daily features

    """
    # Ensure timestamp is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], format="%Y-%m-%d %H:%M:%S", utc=False, errors="coerce")

    # 1. Clean data (apply range rules)
    df = clean_temperature_range(df)
    df = clean_humidity_range(df)
    df = clean_radiation_range(df)

    # 2. Interpolate missing values
    cols_to_interp = [
        "average_air_temperature",
        "average_relative_humidity",
        "max_solar_radiation",
    ]
    df = interpolate_time_series(df, columns=cols_to_interp)

    # 3. Add time features
    df = add_time_features(df)

    # 4. Create hourly temperature features
    temps = create_hourly_temperature_features(df)

    # 9. Join all daily features
    daily = temps.join([])

    # 10. Create target variable
    daily = create_target_min_temp_next_day(daily)

    # 11. Create frost target
    daily = create_frost_target(daily)

    # 12. Clean and filter daily features
    daily = clean_and_filter_daily_features(daily)

    # 13. Select and order columns
    daily = select_and_order_columns(daily)

    daily = daily.reset_index(drop=True)

    daily["id"] = daily.index + 1

    daily = daily[["id"] + [col for col in daily.columns if col != "id"]]

    return daily
