"""baselines.py -- learn each road's normal speed.

1. Loads traffic_train.csv (uses data_loader.py if it works, else cleans it here).
2. Saves traffic_clean.parquet (so later scripts load fast).
3. Computes median + std of speed for every
   (segment_id, day_of_week, time_slot) and saves baselines.csv.

Run this INSIDE the flowpilot folder (next to data_loader.py):
    py baselines.py
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# --- find the dataset folder ---------------------------------------------
CANDIDATES = [
    os.path.join(HERE, "NEURAX_SMART_CITIES_TRAINING_V2"),
    os.path.join(HERE, "data"),
    HERE,
]
DATA_DIR = next(
    (d for d in CANDIDATES if os.path.exists(os.path.join(d, "traffic_train.csv"))),
    None,
)
if DATA_DIR is None:
    sys.exit(
        "ERROR: traffic_train.csv not found. "
        "Put baselines.py in the flowpilot folder next to the dataset."
    )


# --- cleaning (fallback if data_loader.py can't be used) ------------------
NUMERIC_COLS = [
    "speed_kmh",
    "flow_vph",
    "occupancy_pct",
    "travel_time_min",
    "delay_min",
    "queue_length_veh",
    "congestion_index",
]


def clean_manually(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["segment_id", "timestamp"])
    df = df.drop_duplicates(subset=["segment_id", "timestamp"], keep="first")
    for col in NUMERIC_COLS:  # impossible negatives -> missing
        if col in df.columns:
            df[col] = df[col].mask(df[col] < 0)
    if "occupancy_pct" in df.columns:  # occupancy can never exceed 100%
        df["occupancy_pct"] = df["occupancy_pct"].mask(df["occupancy_pct"] > 100)
    # forward-fill up to three 5-minute slots, inside each segment only
    df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)
    fill_cols = [c for c in df.columns if c not in ("segment_id", "timestamp")]
    df[fill_cols] = df.groupby("segment_id")[fill_cols].ffill(limit=3)
    return df


def load_clean_traffic():
    try:
        import data_loader as dl  # noqa: E402

        for name in ("load_traffic", "load_and_clean_traffic", "load_clean", "get_traffic"):
            if hasattr(dl, name):
                print(f"using data_loader.{name}()")
                return getattr(dl, name)()
        print("data_loader has no known loader function; cleaning manually...")
    except Exception as e:  # noqa: BLE001
        print(f"data_loader not usable ({e}); cleaning manually...")
    return clean_manually(os.path.join(DATA_DIR, "traffic_train.csv"))


# --- main -----------------------------------------------------------------
def main():
    df = load_clean_traffic()
    print(f"traffic: {len(df):,} rows | segments: {df['segment_id'].nunique()}")

    pq = os.path.join(HERE, "traffic_clean.parquet")
    df.to_parquet(pq, index=False)
    print(f"cached {os.path.basename(pq)}")

    ts = pd.to_datetime(df["timestamp"])
    df["day_of_week"] = ts.dt.dayofweek          # Monday=0 ... Sunday=6
    df["time_slot"] = ts.dt.hour * 12 + ts.dt.minute // 5  # 0..287

    base = (
        df.groupby(["segment_id", "day_of_week", "time_slot"])["speed_kmh"]
        .agg(speed_median="median", speed_std="std")
        .reset_index()
    )
    base["speed_std"] = base["speed_std"].fillna(0)

    out = os.path.join(HERE, "baselines.csv")
    base.to_csv(out, index=False)
    print(f"baselines: {len(base):,} rows -> {os.path.basename(out)}")
    print("DONE. Next step: anomaly detector.")


if __name__ == "__main__":
    main()
