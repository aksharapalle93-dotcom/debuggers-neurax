"""forecast.py -- predict each road's speed 30 minutes ahead.

Features: recent speed lags, flow, the road's baseline now & at target time,
          day of week, time slot.
Model: HistGradientBoostingRegressor (fast, no tuning needed).
Train: 15 training days. Validate: the 4 official validation days.
Compares against a naive baseline (predict "speed stays the same").

Saves forecast_model.pkl  -> {"model":..., "features":[...]}

Run:
    py forecast.py     (takes a few minutes)
"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))

CANDIDATES = [
    os.path.join(HERE, "NEURAX_SMART_CITIES_TRAINING_V2"),
    os.path.join(HERE, "data"),
    HERE,
]
DATA_DIR = next(
    (d for d in CANDIDATES if os.path.exists(os.path.join(d, "traffic_validation.csv"))),
    None,
)

HORIZON = 6  # 6 x 5min = 30 minutes ahead


def clean(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["segment_id", "timestamp"])
    df = df.drop_duplicates(subset=["segment_id", "timestamp"], keep="first")
    for col in ("speed_kmh", "flow_vph", "occupancy_pct"):
        if col in df.columns:
            df[col] = df[col].mask(df[col] < 0)
    if "occupancy_pct" in df.columns:
        df["occupancy_pct"] = df["occupancy_pct"].mask(df["occupancy_pct"] > 100)
    fill_cols = [c for c in ("speed_kmh", "flow_vph", "occupancy_pct")
                 if c in df.columns]
    df[fill_cols] = df.groupby("segment_id")[fill_cols].ffill(limit=3)
    return df.reset_index(drop=True)


def add_features(df, base):
    df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)
    for k in range(4):
        df[f"speed_lag{k}"] = df.groupby("segment_id")["speed_kmh"].shift(k)
    if "flow_vph" in df.columns:
        df["flow_lag0"] = df.groupby("segment_id")["flow_vph"].shift(0)
    if "occupancy_pct" in df.columns:
        df["occ_lag0"] = df.groupby("segment_id")["occupancy_pct"].shift(0)
    df["label_30"] = df.groupby("segment_id")["speed_kmh"].shift(-HORIZON)

    ts = pd.to_datetime(df["timestamp"])
    df["day_of_week"] = ts.dt.dayofweek
    df["time_slot"] = ts.dt.hour * 12 + ts.dt.minute // 5

    df = df.merge(base, on=["segment_id", "day_of_week", "time_slot"], how="left")
    df = df.rename(columns={"speed_median": "base_now"})

    tslot = (df["time_slot"] + HORIZON) % 288
    tdow = (df["day_of_week"] + (df["time_slot"] + HORIZON) // 288) % 7
    tgt = base.rename(columns={"speed_median": "base_target",
                               "day_of_week": "tdow", "time_slot": "tslot"})
    df["tdow"] = tdow
    df["tslot"] = tslot
    df = df.merge(tgt[["segment_id", "tdow", "tslot", "base_target"]],
                  on=["segment_id", "tdow", "tslot"], how="left")
    return df.drop(columns=["tdow", "tslot"])


FEATURES = ["speed_lag0", "speed_lag1", "speed_lag2", "speed_lag3",
            "base_now", "base_target", "day_of_week", "time_slot"]


def main():
    base = pd.read_csv(os.path.join(HERE, "baselines.csv"))
    print("building train features...")
    train = add_features(
        pd.read_parquet(os.path.join(HERE, "traffic_clean.parquet"),
                        columns=["timestamp", "segment_id", "speed_kmh",
                                 "flow_vph", "occupancy_pct"]),
        base,
    )
    feats = [f for f in FEATURES if f in train.columns]
    train = train.dropna(subset=feats + ["label_30"])
    print(f"train samples: {len(train):,}")

    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08,
                                          random_state=42)
    model.fit(train[feats], train["label_30"])
    print("model trained.")

    print("building validation features...")
    val = add_features(clean(os.path.join(DATA_DIR, "traffic_validation.csv")), base)
    val = val.dropna(subset=feats + ["label_30"])
    pred = model.predict(val[feats])
    mae = mean_absolute_error(val["label_30"], pred)
    naive = mean_absolute_error(val["label_30"], val["speed_lag0"])
    print(f"VALIDATION 30-min MAE: {mae:.2f} km/h  "
          f"(naive persistence: {naive:.2f} km/h)")

    # where it matters: periods when speed actually changed a lot
    vol = (val["label_30"] - val["speed_lag0"]).abs() > 3.0
    if vol.sum() > 100:
        mae_v = mean_absolute_error(val.loc[vol, "label_30"], pred[vol])
        naive_v = mean_absolute_error(val.loc[vol, "label_30"],
                                      val.loc[vol, "speed_lag0"])
        print(f"VOLATILE periods ({100*vol.mean():.1f}% of data): "
              f"model MAE {mae_v:.2f} vs naive {naive_v:.2f} km/h")

    with open(os.path.join(HERE, "forecast_model.pkl"), "wb") as f:
        pickle.dump({"model": model, "features": feats}, f)
    print("saved forecast_model.pkl")
    print("DONE. Next step: recommender.")


if __name__ == "__main__":
    main()
