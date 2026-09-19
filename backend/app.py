"""FlowPilot -- control-room dashboard.

Shows the city road network, live alerts with evidence, 30-min forecasts,
capacity-safe diversion recommendations, and a what-if simulator.

Run inside the flowpilot folder:
    py -m streamlit run app.py
"""

import os
import pickle

import pandas as pd
import pydeck as pdk
import streamlit as st

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

st.set_page_config(page_title="FlowPilot", layout="wide")


@st.cache_resource
def load_all():
    import recommender
    import simulator as sim

    G, anom, seg_edge, cap, length, ffs, mean_flow = recommender.load_all()
    net = pd.read_csv(os.path.join(DATA_DIR, "network.csv"))
    nodes = pd.read_csv(os.path.join(DATA_DIR, "nodes.csv"))
    base = pd.read_csv(os.path.join(HERE, "baselines.csv"))
    traffic = pd.read_parquet(
        os.path.join(HERE, "traffic_clean.parquet"),
        columns=["timestamp", "segment_id", "speed_kmh"],
    )
    with open(os.path.join(HERE, "forecast_model.pkl"), "rb") as f:
        fc = pickle.load(f)
    coord = {r["node_id"]: (r["lon"], r["lat"]) for _, r in nodes.iterrows()}
    seg_nodes = {}
    for u, v, d in G.edges(data=True):
        seg_nodes[d["segment_id"]] = (u, v)
    return {
        "G": G, "anom": anom, "seg_edge": seg_edge, "cap": cap,
        "length": length, "ffs": ffs, "mean_flow": mean_flow,
        "net": net, "nodes": nodes, "base": base, "traffic": traffic,
        "model": fc["model"], "features": fc["features"],
        "coord": coord, "seg_nodes": seg_nodes,
        "recommender": recommender, "sim": sim,
    }


def predict_30min(d, seg, T):
    t = d["traffic"]
    rows = t[t["segment_id"] == seg].sort_values("timestamp").reset_index(drop=True)
    idx = rows.index[rows["timestamp"] == T]
    if len(idx) == 0 or idx[0] < 3:
        return None
    i = idx[0]
    lag = [rows.loc[i - k, "speed_kmh"] for k in range(4)]
    ts = pd.Timestamp(T)
    dow, slot = ts.dayofweek, ts.hour * 12 + ts.minute // 5
    b = d["base"]
    bn = b[(b["segment_id"] == seg) & (b["day_of_week"] == dow)
           & (b["time_slot"] == slot)]
    tslot, tdow = (slot + 6) % 288, (dow + (slot + 6) // 288) % 7
    bt = b[(b["segment_id"] == seg) & (b["day_of_week"] == tdow)
           & (b["time_slot"] == tslot)]
    if bn.empty or bt.empty:
        return None
    import pandas as pd  # noqa: F811
    X = pd.DataFrame([{
        "speed_lag0": lag[0], "speed_lag1": lag[1],
        "speed_lag2": lag[2], "speed_lag3": lag[3],
        "base_now": float(bn.iloc[0]["speed_median"]),
        "base_target": float(bt.iloc[0]["speed_median"]),
        "day_of_week": dow, "time_slot": slot,
    }])
    return float(d["model"].predict(X[d["features"]])[0])


d = load_all()
rec = d["recommender"]
sim = d["sim"]

# ---------------- header ----------------
st.title("FlowPilot")
st.caption("See the jam before it forms. Stop it before it spreads.")

# ---------------- pick an alert ----------------
top = (d["anom"].sort_values("drop_pct", ascending=False)
       .drop_duplicates("segment_id").head(10))
labels = [f"{r['segment_id']} @ {r['timestamp']} (-{r['drop_pct']:.0%} speed)"
          for _, r in top.iterrows()]
choice = st.sidebar.selectbox("Active alerts", labels)
sel = top.iloc[labels.index(choice)]
SEG, T = sel["segment_id"], pd.Timestamp(sel["timestamp"])

# ---------------- map ----------------
t = d["traffic"]
now = t[t["timestamp"] == T].set_index("segment_id")["speed_kmh"].to_dict()
ts = pd.Timestamp(T)
dow, slot = ts.dayofweek, ts.hour * 12 + ts.minute // 5
bmap = d["base"].set_index(["segment_id", "day_of_week", "time_slot"])["speed_median"]

lines = []
anom_set = set(d["anom"][d["anom"]["timestamp"] == T]["segment_id"])
for seg, (u, v) in d["seg_nodes"].items():
    if seg not in now or u not in d["coord"] or v not in d["coord"]:
        continue
    sp = now[seg]
    try:
        normal = bmap.loc[(seg, dow, slot)]
    except KeyError:
        continue
    drop = (normal - sp) / normal if normal else 0
    if seg in anom_set:
        color = [220, 30, 30]
    elif drop > 0.15:
        color = [240, 160, 0]
    else:
        color = [30, 180, 80]
    lon1, lat1 = d["coord"][u]
    lon2, lat2 = d["coord"][v]
    lines.append({"source": [lon1, lat1], "target": [lon2, lat2],
                  "color": color, "segment_id": seg,
                  "label": f"{seg}: {sp:.0f} km/h (normal {normal:.0f})"})

lats = [c[1] for c in d["coord"].values()]
lons = [c[0] for c in d["coord"].values()]
view = pdk.ViewState(latitude=sum(lats) / len(lats),
                     longitude=sum(lons) / len(lons), zoom=10.5)
layer = pdk.Layer("LineLayer", data=lines,
                  get_source_position="source", get_target_position="target",
                  get_color="color", get_width=5, pickable=True)
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                         tooltip={"text": "{label}"}))
st.caption(f"Road status at {T} -- red = alert, yellow = slowing, green = normal")

# ---------------- detail tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(["Evidence", "Forecast", "Action", "What-if lab"])

with tab1:
    st.subheader(f"Alert: {SEG}")
    st.metric("Current speed", f"{sel['speed_kmh']:.0f} km/h",
              delta=f"{sel['speed_kmh'] - sel['speed_median']:.0f} vs normal")
    st.write(f"Normal for this road at this time: **{sel['speed_median']:.0f} km/h**")
    st.write(f"Drop: **{sel['drop_pct']:.0%}** -- flagged as abnormal congestion.")

with tab2:
    pred = predict_30min(d, SEG, T)
    if pred is None:
        st.write("Not enough history to forecast this segment.")
    else:
        st.metric("Predicted speed in 30 min", f"{pred:.0f} km/h")
        st.caption("Typical error when traffic is changing: +/-3.2 km/h "
                   "(measured on validation data).")

with tab3:
    recs = rec.recommend(SEG, d["G"], d["seg_edge"], d["cap"], d["length"],
                         d["ffs"], d["mean_flow"])
    if not recs:
        st.warning("No SAFE diversion: every alternate road is too full. "
                   "Recommendation: hold traffic and alert on the ground.")
    for i, r in enumerate(recs[:2], 1):
        st.markdown(f"**Option {i}** -- via {' -> '.join(r['via'])}")
        st.write(f"+{r['extra_km']} km, +{r['extra_min']} min. "
                 f"Bottleneck {r['bottleneck']} at {r['worst_ratio']*100:.0f}% "
                 f"capacity ({r['headroom_pct']}% headroom). "
                 f"Safe: will not create a new jam.")
    if st.button("Simulate: follow option 1"):
        flow = d["mean_flow"].get(SEG, 0)
        cap_eff = sim.calibrate_capacity(d["length"][SEG], d["ffs"][SEG],
                                         flow, sel["speed_kmh"], d["cap"][SEG])
        tb = sim.bpr_minutes(d["length"][SEG], d["ffs"][SEG], flow, cap_eff)
        ta = sim.bpr_minutes(d["length"][SEG], d["ffs"][SEG],
                             flow * 0.7, cap_eff)
        st.write(f"Stayers: {tb:.1f} -> {ta:.1f} min per vehicle.")
        if recs:
            div = flow * 0.3
            alt = sum(sim.bpr_minutes(d["length"][s], d["ffs"][s],
                                      d["mean_flow"].get(s, 0) + div,
                                      d["cap"].get(s, 1)) for s in recs[0]["via"])
            st.write(f"Diverters: {alt:.1f} min vs {tb:.1f} staying.")
            st.success(f"Estimated total: "
                       f"{flow*0.7*(tb-ta) + div*max(tb-alt,0):,.0f} "
                       f"vehicle-minutes saved per hour. (PROTOTYPE)")
        st.caption("PROTOTYPE SIMULATION -- estimates, not guarantees.")

with tab4:
    st.subheader("What-if infrastructure lab")
    st.caption("PROTOTYPE -- test ideas before spending money.")
    wseg = st.selectbox("Road", sorted(d["seg_nodes"].keys()),
                        index=sorted(d["seg_nodes"].keys()).index(SEG))
    delta = st.slider("Extra capacity (veh/h)", 0, 1200, 600, step=50)
    flow = d["mean_flow"].get(wseg, 0)
    tb = sim.bpr_minutes(d["length"][wseg], d["ffs"][wseg], flow, d["cap"][wseg])
    ta = sim.bpr_minutes(d["length"][wseg], d["ffs"][wseg],
                         flow, d["cap"][wseg] + delta)
    st.write(f"{wseg}: {tb:.1f} -> {ta:.1f} min per vehicle "
             f"(save {tb-ta:.1f} min each).")

st.divider()
st.caption("FlowPilot prototype -- built on the organizer's 15-day dataset. "
           "Simulations are estimates for decision support.")
