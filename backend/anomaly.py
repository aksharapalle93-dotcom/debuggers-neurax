"""anomaly.py -- flag abnormal jams: live speed vs each road's own baseline.

Rule (simple + explainable):
    flag when speed is 30%+ below that road's normal for that day & time,
    AND at least 5 km/h below normal (ignores tiny wiggles).

Why not z-score: each (road, day, time) group has only ~2 readings,
so a z-score can never reach the threshold. A relative drop is robust.

Also saves anomalies.csv and VALIDATES against the 49 labeled incidents.

Run inside the flowpilot folder:
    py anomaly.py
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

CANDIDATES = [
    os.path.join(HERE, "NEURAX_SMART_CITIES_TRAINING_V2"),
    os.path.join(HERE, "data"),
    HERE,
]
DATA_DIR = next(
    (d for d in CANDIDATES if os.path.exists(os.path.join(d, "incidents_train.csv"))),
    None,
)

DROP_PCT = 0.30   # 30% slower than normal
DROP_ABS = 5.0    # and at least 5 km/h slower


def main():
    traffic = pd.read_parquet(
        os.path.join(HERE, "traffic_clean.parquet"),
        columns=["timestamp", "segment_id", "speed_kmh"],
    )
    base = pd.read_csv(os.path.join(HERE, "baselines.csv"))

    ts = pd.to_datetime(traffic["timestamp"])
    traffic["day_of_week"] = ts.dt.dayofweek
    traffic["time_slot"] = ts.dt.hour * 12 + ts.dt.minute // 5

    m = traffic.merge(base, on=["segment_id", "day_of_week", "time_slot"], how="left")
    m["drop"] = m["speed_median"] - m["speed_kmh"]
    m["drop_pct"] = m["drop"] / m["speed_median"]
    m["is_anomaly"] = (m["drop_pct"] > DROP_PCT) & (m["drop"] > DROP_ABS)

    anom = m[m["is_anomaly"]].copy()
    anom[["timestamp", "segment_id", "speed_kmh", "speed_median",
          "drop", "drop_pct"]].to_csv(
        os.path.join(HERE, "anomalies.csv"), index=False
    )
    print(f"anomalies flagged: {len(anom):,} / {len(m):,} readings "
          f"({100 * len(anom) / len(m):.2f}%)")

    # --- validation against the 49 labeled incidents ---------------------
    if DATA_DIR is None:
        print("incidents_train.csv not found; skipping validation.")
        return
    inc = pd.read_csv(os.path.join(DATA_DIR, "incidents_train.csv"))
    inc["start_time"] = pd.to_datetime(inc["start_time"])
    inc["end_time"] = pd.to_datetime(inc["end_time"])
    anom["timestamp"] = pd.to_datetime(anom["timestamp"])

    caught = 0
    for _, row in inc.iterrows():
        hit = anom[
            (anom["segment_id"] == row["segment_id"])
            & (anom["timestamp"] >= row["start_time"])
            & (anom["timestamp"] <= row["end_time"])
        ]
        if len(hit):
            caught += 1
    print(f"VALIDATION: caught {caught}/{len(inc)} labeled incidents "
          f"({100 * caught / len(inc):.1f}%)")
    print("DONE. Next step: road graph.")


if __name__ == "__main__":
    main()
