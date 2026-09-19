"""FlowPilot -- control-room dashboard.

See the jam before it forms. Stop it before it spreads.

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

st.set_page_config(page_title="FlowPilot", page_icon="🚦", layout="wide")

# ---------------- styles ----------------
st.markdown(
    """
    <style>
    .fp-banner { background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
                 padding: 20px 26px; border-radius: 14px; color: white;
                 margin-bottom: 14px; }
    .fp-banner h1 { margin: 0; font-size: 36px; color: white; }
    .fp-tag { opacity: 0.85; font-size: 16px; margin-top: 6px; }
    .fp-legend { display: flex; gap: 18px; margin: 8px 0; font-size: 14px; }
    .dot { display: inline-block; width: 12px; height: 12px;
           border-radius: 50%; margin-right: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="fp-banner"><h1>🚦 FlowPilot</h1>'
    '<div class="fp-tag">See the jam before it forms. '
    "Stop it before it spreads.</div></div>",
    unsafe_allow_html=True,
)


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

# ---------------- sidebar ----------------
st.sidebar.header("🚨 Active alerts")
top = (d["anom"].sort_values("drop_pct", ascending=False)
       .drop_duplicates("segment_id").head(10))
labels = [f"{r['segment_id']} @ {r['timestamp']} (-{r['drop_pct']:.0%} speed)"
          for _, r in top.iterrows()]
choice = st.sidebar.selectbox("Pick an alert to investigate", labels)
sel = top.iloc[labels.index(choice)]
SEG, T = sel["segment_id"], pd.Timestamp(sel["timestamp"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Map legend**")
st.sidebar.markdown(
    '<div class="fp-legend">'
    '<span><span class="dot" style="background:#dc1e1e"></span>Alert</span>'
    '<span><span class="dot" style="background:#f0a000"></span>Slowing</span>'
    '<span><span class="dot" style="background:#1eb450"></span>Normal</span>'
    "</div>",
    unsafe_allow_html=True,
)
st.sidebar.caption("436 roads • 120 junctions • 15 days of data • 5-min resolution")

# ---------------- KPI row ----------------
n_alerts_now = d["anom"][d["anom"]["timestamp"] == T]["segment_id"].nunique()
k1, k2, k3 = st.columns(3)
k1.metric("🚨 Alerts at this moment", n_alerts_now)
k2.metric("🛣️ Roads monitored", "436")
k3.metric("📉 Worst speed drop", f"{top.iloc[0]['drop_pct']:.0%}")

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
        color = [235, 45, 45]
    elif drop > 0.15:
        color = [240, 160, 0]
    else:
        color = [70, 130, 95]
    lon1, lat1 = d["coord"][u]
    lon2, lat2 = d["coord"][v]
    lines.append({"source": [lon1, lat1], "target": [lon2, lat2],
                  "color": color, "segment_id": seg,
                  "width": 7 if seg in anom_set else (3 if drop > 0.15 else 1.2),
                  "label": f"{seg}: {sp:.0f} km/h (normal {normal:.0f})"})

lats = [c[1] for c in d["coord"].values()]
lons = [c[0] for c in d["coord"].values()]
view = pdk.ViewState(latitude=sum(lats) / len(lats),
                     longitude=sum(lons) / len(lons), zoom=10.5)
layer = pdk.Layer("LineLayer", data=lines,
                  get_source_position="source", get_target_position="target",
                  get_color="color", get_width="width", pickable=True,
                  opacity=0.85)
st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view,
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    tooltip={"text": "{label}"}))
st.caption(f"Network status at {T} — control-room schematic view, hover any road for details.")

# ---------------- detail tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Evidence", "🔮 Forecast", "✅ Action", "🔬 What-if lab"])

with tab1:
    st.subheader(f"Alert: {SEG}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Current speed", f"{sel['speed_kmh']:.0f} km/h",
              delta=f"{sel['speed_kmh'] - sel['speed_median']:.0f} vs normal",
              delta_color="inverse")
    m2.metric("Normal for this time", f"{sel['speed_median']:.0f} km/h")
    m3.metric("Speed drop", f"{sel['drop_pct']:.0%}")
    st.info("The system learned this road's normal hour-by-hour, so it knows "
            "this isn't routine traffic — it's abnormal.")

with tab2:
    pred = predict_30min(d, SEG, T)
    if pred is None:
        st.warning("Not enough history to forecast this segment.")
    else:
        st.metric("Predicted speed in 30 min", f"{pred:.0f} km/h",
                  delta=f"{pred - sel['speed_kmh']:.0f} km/h from now")
        st.caption("Typical error when traffic is changing: ±3.2 km/h "
                   "(measured on validation data).")
        if pred < sel["speed_median"] * 0.7:
            st.error("⚠️ Jam expected to persist or worsen — act now.")
        else:
            st.success("✅ Conditions expected to improve.")

with tab3:
    st.subheader("Recommended actions")
    recs = rec.recommend(SEG, d["G"], d["seg_edge"], d["cap"], d["length"],
                         d["ffs"], d["mean_flow"])
    if not recs:
        st.warning("⛔ No SAFE diversion: every alternate road is too full. "
                   "Recommendation: hold traffic and alert ground teams.")
    for i, r in enumerate(recs[:2], 1):
        st.markdown(f"**Option {i}** — via {' → '.join(r['via'])}")
        a, b, c = st.columns(3)
        a.metric("Extra distance", f"+{r['extra_km']} km")
        b.metric("Extra time", f"+{r['extra_min']} min")
        c.metric("Spare capacity", f"{r['headroom_pct']}%")
        st.caption(f"Bottleneck {r['bottleneck']} at "
                   f"{r['worst_ratio']*100:.0f}% capacity — safe, "
                   "will not create a new jam.")
    if recs and st.button("▶ Simulate: follow option 1", type="primary"):
        flow = d["mean_flow"].get(SEG, 0)
        cap_eff = sim.calibrate_capacity(d["length"][SEG], d["ffs"][SEG],
                                         flow, sel["speed_kmh"], d["cap"][SEG])
        tb = sim.bpr_minutes(d["length"][SEG], d["ffs"][SEG], flow, cap_eff)
        ta = sim.bpr_minutes(d["length"][SEG], d["ffs"][SEG],
                             flow * 0.7, cap_eff)
        s1, s2 = st.columns(2)
        s1.metric("Stayers: before", f"{tb:.1f} min/veh")
        s2.metric("Stayers: after", f"{ta:.1f} min/veh",
                  delta=f"-{tb - ta:.1f} min")
        div = flow * 0.3
        alt = sum(sim.bpr_minutes(d["length"][s], d["ffs"][s],
                                  d["mean_flow"].get(s, 0) + div,
                                  d["cap"].get(s, 1)) for s in recs[0]["via"])
        st.write(f"Diverters: **{alt:.1f} min** vs {tb:.1f} staying "
                 f"(save {tb - alt:.1f} min each).")
        total = flow * 0.7 * (tb - ta) + div * max(tb - alt, 0)
        st.success(f"Estimated total: **{total:,.0f} vehicle-minutes "
                   "saved per hour.**")
        st.caption("PROTOTYPE SIMULATION — estimates, not guarantees.")

with tab4:
    st.subheader("What-if infrastructure lab")
    st.caption("PROTOTYPE — test ideas before spending money.")
    wseg = st.selectbox("Road", sorted(d["seg_nodes"].keys()),
                        index=sorted(d["seg_nodes"].keys()).index(SEG))
    delta = st.slider("Extra capacity (veh/h)", 0, 1200, 600, step=50)
    flow = d["mean_flow"].get(wseg, 0)
    tb = sim.bpr_minutes(d["length"][wseg], d["ffs"][wseg], flow, d["cap"][wseg])
    ta = sim.bpr_minutes(d["length"][wseg], d["ffs"][wseg],
                         flow, d["cap"][wseg] + delta)
    w1, w2 = st.columns(2)
    w1.metric("Travel time now", f"{tb:.1f} min/veh")
    w2.metric("With upgrade", f"{ta:.1f} min/veh", delta=f"-{tb - ta:.1f} min")

st.divider()
st.caption("FlowPilot prototype — built on the organizer's 15-day dataset. "
           "Simulations are estimates for decision support.")
