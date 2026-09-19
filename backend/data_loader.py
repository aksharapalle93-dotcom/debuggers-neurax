"""
FlowPilot - data_loader.py
Loads + cleans the NeuraX Smart Cities dataset.
Run: py data_loader.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent / "NEURAX_SMART_CITIES_TRAINING_V2"

def clean_traffic(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # 1. fix shuffled rows
    df = df.sort_values(["segment_id", "timestamp"])
    # 2. drop duplicate readings
    df = df.drop_duplicates(subset=["timestamp", "segment_id"], keep="last")
    # 3. impossible negative readings -> NaN
    num_cols = ["speed_kmh", "flow_vph", "occupancy_pct", "travel_time_min",
                "free_flow_time_min", "delay_min", "queue_length_veh",
                "congestion_index"]
    for c in num_cols:
        df.loc[df[c] < 0, c] = np.nan
    df.loc[df["occupancy_pct"] > 100, "occupancy_pct"] = np.nan
    # 4. fill tiny gaps: forward-fill within each segment (max 3 slots = 15 min)
    df = df.set_index(["segment_id", "timestamp"]).sort_index()
    df = df.groupby(level="segment_id").ffill(limit=3)
    return df.reset_index()

def load_all():
    traffic = clean_traffic(pd.read_csv(DATA / "traffic_train.csv"))
    network = pd.read_csv(DATA / "network.csv")
    nodes = pd.read_csv(DATA / "nodes.csv")
    incidents = pd.read_csv(DATA / "incidents_train.csv",
                            parse_dates=["start_time", "end_time"])
    context = pd.read_csv(DATA / "context_train.csv", parse_dates=["timestamp"])
    targets = pd.read_csv(DATA / "forecast_targets_train.csv", parse_dates=["timestamp"])
    plans = pd.read_csv(DATA / "planning_candidates.csv")
    return traffic, network, nodes, incidents, context, targets, plans

if __name__ == "__main__":
    traffic, network, nodes, incidents, context, targets, plans = load_all()
    print("traffic:", traffic.shape, "| segments:", traffic["segment_id"].nunique())
    print("network:", network.shape, "| nodes:", nodes.shape)
    print("incidents:", incidents.shape, "| plans:", plans.shape)
    print("context:", context.shape, "| targets:", targets.shape)
    print("date range:", traffic["timestamp"].min(), "->", traffic["timestamp"].max())
