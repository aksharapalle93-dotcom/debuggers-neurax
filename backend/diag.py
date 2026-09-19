"""diag.py -- find out why the anomaly detector found nothing."""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

t = pd.read_parquet(
    os.path.join(HERE, "traffic_clean.parquet"),
    columns=["timestamp", "segment_id", "speed_kmh"],
)
b = pd.read_csv(os.path.join(HERE, "baselines.csv"))

print("=== traffic dtypes ===")
print(t.dtypes)
print("=== baselines dtypes ===")
print(b.dtypes)
print("=== traffic head ===")
print(t.head(3).to_string())
print("=== baselines head ===")
print(b.head(3).to_string())
print("=== speed_kmh describe ===")
print(t["speed_kmh"].describe().to_string())
print("=== speed_std describe ===")
print(b["speed_std"].describe().to_string())
print("unique segments:", t["segment_id"].nunique(), "vs", b["segment_id"].nunique())

tmp = t.head(100000).copy()
ts = pd.to_datetime(tmp["timestamp"])
tmp["day_of_week"] = ts.dt.dayofweek
tmp["time_slot"] = ts.dt.hour * 12 + ts.dt.minute // 5
m = tmp.merge(b, on=["segment_id", "day_of_week", "time_slot"],
              how="left", indicator=True)
print("=== merge match rate (first 100k rows) ===")
print(m["_merge"].value_counts())
