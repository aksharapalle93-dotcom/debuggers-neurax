"""recommender.py -- suggest SAFE diversions around a jammed road.

For a jammed segment:
  1. Find alternate paths around it in the road graph.
  2. CAPACITY GUARDRAIL: keep a path only if every road on it still has
     free space AFTER absorbing diverted traffic (flow/capacity < 0.95).
     Never move the jam to another road.
  3. Rank survivors by extra travel time.

Demo: runs on the 3 worst anomalies and prints recommendations.

Run:
    py recommender.py
"""

import os
import pickle
from itertools import islice

import networkx as nx
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

CANDIDATES = [
    os.path.join(HERE, "NEURAX_SMART_CITIES_TRAINING_V2"),
    os.path.join(HERE, "data"),
    HERE,
]
DATA_DIR = next(
    (d for d in CANDIDATES if os.path.exists(os.path.join(d, "network.csv"))),
    None,
)

DIVERT_SHARE = 0.30   # assume 30% of the jammed road's flow diverts
MAX_RATIO = 0.95      # a road is "full" above 95% of capacity


def load_all():
    with open(os.path.join(HERE, "graph.pkl"), "rb") as f:
        G = nx.DiGraph(pickle.load(f))
    net = pd.read_csv(os.path.join(DATA_DIR, "network.csv"))
    anom = pd.read_csv(os.path.join(HERE, "anomalies.csv"),
                       parse_dates=["timestamp"])
    traffic = pd.read_parquet(os.path.join(HERE, "traffic_clean.parquet"),
                              columns=["segment_id", "flow_vph"])
    seg_edge = {}
    for u, v, d in G.edges(data=True):
        seg_edge[d["segment_id"]] = (u, v)
    cap = dict(zip(net["segment_id"], net["capacity_vph"]))
    length = dict(zip(net["segment_id"], net["length_km"]))
    ffs = dict(zip(net["segment_id"], net["free_flow_speed_kmh"]))
    mean_flow = traffic.groupby("segment_id")["flow_vph"].mean().to_dict()
    return G, anom, seg_edge, cap, length, ffs, mean_flow


def recommend(jammed_seg, G, seg_edge, cap, length, ffs, mean_flow, k=4):
    """Return viable diversion paths around jammed_seg (ranked)."""
    if jammed_seg not in seg_edge:
        return []
    src, dst = seg_edge[jammed_seg]
    diverted = DIVERT_SHARE * mean_flow.get(jammed_seg, 0)
    results = []
    try:
        paths = islice(nx.shortest_simple_paths(G, src, dst, weight="length_km"),
                       k + 1)
    except nx.NetworkXNoPath:
        return []
    for path in paths:
        seg_path = []
        ok = True
        for u, v in zip(path[:-1], path[1:]):
            s = G[u][v]["segment_id"]
            if s == jammed_seg:      # path uses the jammed road itself
                ok = False
                break
            seg_path.append(s)
        if not ok or not seg_path:
            continue
        # capacity guardrail: every road must absorb diverted flow
        worst_ratio, worst_seg = 0, None
        for s in seg_path:
            ratio = (mean_flow.get(s, 0) + diverted) / cap.get(s, 1)
            if ratio > worst_ratio:
                worst_ratio, worst_seg = ratio, s
        if worst_ratio >= MAX_RATIO:
            continue  # would create a new jam -> reject
        extra_km = sum(length.get(s, 0) for s in seg_path)
        extra_min = sum(length.get(s, 0) / max(ffs.get(s, 30), 1) * 60
                        for s in seg_path)
        results.append({
            "via": seg_path,
            "extra_km": round(extra_km, 2),
            "extra_min": round(extra_min, 1),
            "worst_ratio": round(worst_ratio, 2),
            "bottleneck": worst_seg,
            "headroom_pct": round((1 - worst_ratio) * 100),
        })
    return sorted(results, key=lambda r: r["extra_min"])


def main():
    G, anom, seg_edge, cap, length, ffs, mean_flow = load_all()
    top = anom.sort_values("drop_pct", ascending=False).head(3)
    for _, row in top.iterrows():
        seg = row["segment_id"]
        print(f"\n=== jammed: {seg} at {row['timestamp']} "
              f"(speed {row['speed_kmh']:.0f} vs normal {row['speed_median']:.0f})")
        recs = recommend(seg, G, seg_edge, cap, length, ffs, mean_flow)
        if not recs:
            print("  no SAFE diversion: every alternate road is too full. "
                  "Advice: hold traffic / alert, don't divert.")
        for i, r in enumerate(recs[:2], 1):
            print(f"  option {i}: via {r['via']}")
            print(f"    +{r['extra_km']} km, +{r['extra_min']} min, "
                  f"bottleneck {r['bottleneck']} at {r['worst_ratio']*100:.0f}% "
                  f"capacity ({r['headroom_pct']}% headroom)")
    print("\nDONE. Next step: simulator.")


if __name__ == "__main__":
    main()
