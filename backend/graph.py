"""graph.py -- build the road network graph.

Nodes = junctions (from nodes.csv), edges = road segments (from network.csv).
Used later for: jam spread modeling, diversion routes, map drawing.

Saves graph.pkl in the flowpilot folder.

Run:
    py graph.py
"""

import os
import pickle

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
if DATA_DIR is None:
    raise SystemExit("ERROR: network.csv not found.")


def find_col(df, *keywords):
    for kw in keywords:
        for c in df.columns:
            if kw in str(c).lower():
                return c
    return None


def main():
    net = pd.read_csv(os.path.join(DATA_DIR, "network.csv"))
    nodes = pd.read_csv(os.path.join(DATA_DIR, "nodes.csv"))
    print(f"network cols: {list(net.columns)}")
    print(f"nodes cols: {list(nodes.columns)}")

    node_col = find_col(nodes, "node")
    src_col = find_col(net, "source", "from", "start", "origin", "node_a", "u")
    dst_col = find_col(net, "target", "to", "end", "dest", "node_b", "v")
    seg_col = find_col(net, "segment")
    if not (src_col and dst_col):
        raise SystemExit("Could not find source/target node columns in network.csv")

    G = nx.DiGraph()
    for _, r in nodes.iterrows():
        G.add_node(r[node_col], **{k: r[k] for k in nodes.columns if k != node_col})
    for _, r in net.iterrows():
        G.add_edge(r[src_col], r[dst_col],
                   **{k: r[k] for k in net.columns
                      if k not in (src_col, dst_col)})

    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    und = G.to_undirected()
    comps = nx.number_connected_components(und)
    print(f"connected components: {comps}")
    if comps > 1:
        print("WARNING: graph is not fully connected (ok for prototype).")

    with open(os.path.join(HERE, "graph.pkl"), "wb") as f:
        pickle.dump(G, f)
    print("saved graph.pkl")
    print("DONE. Next step: forecaster.")


if __name__ == "__main__":
    main()
