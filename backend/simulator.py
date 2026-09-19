"""simulator.py -- PROTOTYPE: estimate impact of actions and infra ideas.

Method (calibrated BPR, standard traffic engineering):
  1. During a jam, effective capacity collapses (blocked lanes, rubbernecking).
     We CALIBRATE effective capacity so the BPR formula reproduces the
     observed jam speed exactly -- no invented numbers.
  2. Then we simulate: what if 30% of flow diverts? What if capacity grows?
  travel_time = free_time * (1 + 0.15 * (flow / eff_capacity) ** 4)

Everything here is an estimate. Label: PROTOTYPE SIMULATION.

Run:
    py simulator.py
"""

import os
import pickle

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

CANDIDATES = [
    os.path.join(HERE, "NEURAX_SMART_CITIES_TRAINING_V2"),
    os.path.join(HERE, "data"),
    HERE,
]
DATA_DIR = next(
    (d for d in CANDIDATES if os.path.exists(os.path.join(d, "planning_candidates.csv"))),
    None,
)

DIVERT_SHARE = 0.30


def bpr_minutes(length_km, ffs_kmh, flow, capacity):
    t0 = length_km / max(ffs_kmh, 1) * 60
    ratio = flow / max(capacity, 1)
    return t0 * (1 + 0.15 * ratio ** 4)


def calibrate_capacity(length_km, ffs_kmh, flow, observed_speed, nominal_cap):
    """Effective capacity that makes BPR reproduce the observed jam speed."""
    t0 = length_km / max(ffs_kmh, 1) * 60
    t_obs = length_km / max(observed_speed, 1) * 60
    if t_obs <= t0 * 1.05:
        return nominal_cap  # not really jammed
    ratio = ((t_obs / t0 - 1) / 0.15) ** 0.25
    return flow / max(ratio, 0.05)


def main():
    import recommender  # reuse its loader + recommend()

    G, anom, seg_edge, cap, length, ffs, mean_flow = recommender.load_all()
    worst = anom.sort_values("drop_pct", ascending=False).iloc[0]
    seg = worst["segment_id"]
    flow = mean_flow.get(seg, 0)
    obs_speed = worst["speed_kmh"]

    cap_eff = calibrate_capacity(length[seg], ffs[seg], flow, obs_speed, cap[seg])

    print("=== PROTOTYPE SIMULATION: diversion action ===")
    print(f"jammed {seg}: observed {obs_speed:.0f} km/h "
          f"(normal {worst['speed_median']:.0f}), effective capacity "
          f"{cap_eff:.0f} veh/h vs nominal {cap[seg]:.0f}")
    t_before = bpr_minutes(length[seg], ffs[seg], flow, cap_eff)
    t_after = bpr_minutes(length[seg], ffs[seg], flow * (1 - DIVERT_SHARE), cap_eff)
    s_after = length[seg] / t_after * 60
    print(f"stayers: {t_before:.1f} -> {t_after:.1f} min/veh "
          f"(speed {obs_speed:.0f} -> {s_after:.0f} km/h)")

    recs = recommender.recommend(seg, G, seg_edge, cap, length, ffs, mean_flow)
    if recs:
        r = recs[0]
        diverted = DIVERT_SHARE * flow
        alt_time = sum(
            bpr_minutes(length[s], ffs[s],
                        mean_flow.get(s, 0) + diverted, cap.get(s, 1))
            for s in r["via"]
        )
        print(f"diverters via {r['via']}: {alt_time:.1f} min/veh vs "
              f"{t_before:.1f} staying (save {t_before - alt_time:.1f} min each)")
        total = (flow * (1 - DIVERT_SHARE) * (t_before - t_after)
                 + diverted * max(t_before - alt_time, 0))
        print(f"ESTIMATED TOTAL: {total:,.0f} vehicle-minutes saved per hour")
    else:
        print("no safe diversion available in this scenario.")

    # --- what-if infrastructure lab ------------------------------------
    print("\n=== PROTOTYPE SIMULATION: what-if infrastructure ===")
    plans = pd.read_csv(os.path.join(DATA_DIR, "planning_candidates.csv"))
    plans["_value"] = (plans["capacity_delta_vph"]
                       / plans["cost_index"].replace(0, float("nan")))
    print("top candidates by (capacity gain / cost):")
    for _, p in plans.sort_values("_value", ascending=False).head(3).iterrows():
        print(f"  {p['candidate_id']}: {p['intervention_type']} on "
              f"{p['target_segment']} (+{p['capacity_delta_vph']:.0f} veh/h, "
              f"cost {p['cost_index']}, {p['feasibility_band']})")

    lane_cap = 600.0  # illustrative: one extra lane
    t_lane = bpr_minutes(length[seg], ffs[seg], flow, cap_eff + lane_cap)
    s_lane = length[seg] / t_lane * 60
    print(f"illustrative: +1 lane on {seg} during this jam -> "
          f"{t_before:.1f} -> {t_lane:.1f} min/veh "
          f"(speed {obs_speed:.0f} -> {s_lane:.0f} km/h)")
    print("\nDONE. Backend complete: loader, baselines, anomaly, graph, "
          "forecast, recommender, simulator.")


if __name__ == "__main__":
    main()
