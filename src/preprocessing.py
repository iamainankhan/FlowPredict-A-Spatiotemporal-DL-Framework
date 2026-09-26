import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


TARGET = "avg_speed_kmph"

FEATURES = [
    "avg_speed_kmph",
    "density_veh_per_km",
    "avg_wait_time_s",
    "occupancy_pct",
    "flow_veh_per_hr",
    "queue_length_veh",
    "avg_accel_ms2",
    "heading_deg",
    "signal_state_num",
    "incident_num",
    "temp_c",
    "visibility_km",
    "rain_intensity_mmph",
    "channel_busy"
]


def load_data(path):

    df = pd.read_csv(path)

    print("Dataset shape:", df.shape)

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort data
    df = df.sort_values(
        ["timestamp", "road_segment_id"]
    ).reset_index(drop=True)

    return df


def clean_data(df):

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Check required columns
    required = [
        "timestamp",
        "road_segment_id"
    ] + FEATURES

    missing_columns = [
        col for col in required
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Convert numerical columns
    for col in FEATURES:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Fill missing numerical values
    df[FEATURES] = df[FEATURES].interpolate(
        method="linear"
    )

    df[FEATURES] = df[FEATURES].ffill().bfill()

    # Remove impossible values
    df = df[
        df["avg_speed_kmph"] >= 0
    ]

    return df


def chronological_split(df):

    timestamps = sorted(
        df["timestamp"].unique()
    )

    n = len(timestamps)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_times = timestamps[:train_end]
    val_times = timestamps[train_end:val_end]
    test_times = timestamps[val_end:]

    train_df = df[
        df["timestamp"].isin(train_times)
    ].copy()

    val_df = df[
        df["timestamp"].isin(val_times)
    ].copy()

    test_df = df[
        df["timestamp"].isin(test_times)
    ].copy()

    print("\nData split:")
    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    return train_df, val_df, test_df


def scale_data(train_df, val_df, test_df):

    scaler = StandardScaler()

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    scaler.fit(
        train_df[FEATURES]
    )

    train_df[FEATURES] = scaler.transform(
        train_df[FEATURES]
    )

    val_df[FEATURES] = scaler.transform(
        val_df[FEATURES]
    )

    test_df[FEATURES] = scaler.transform(
        test_df[FEATURES]
    )

    return train_df, val_df, test_df, scaler