### COMMENTS OF THE CODE AND VARIABLE NAMES WERE REVIEWED AND IMPROVED BY CHATGPT-5 ###


# -*- coding: utf-8 -*-
# ============================================================
# SCI Analysis from saved files:
# - country selection
# - per-country node export
# - weighted network metrics
# - global distance vs SCI plot
#
# Saves all data to outputs/data/ and plots to outputs/plots/.
# Prints how many and which countries are selected.
# Comments are concise and in English.
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------------
# I/O configuration and params
# -------------------------------
NODE_LIST_PATH        = "node_list.csv"
EDGES_ALL_PATH        = "edges_all.csv"
EDGES_BY_COUNTRY_DIR  = "edges_by_country"          # contains edges_<ISO3>.csv
NODES_BY_COUNTRY_DIR  = "nodes_by_country"          # will write nodes_<ISO3>.csv
CACHE_COVERAGE_PICKLE = "cache/coverage_by_country.pkl"  # optional

# Output folders
OUTPUT_DATA_DIR  = "outputs/data"
OUTPUT_PLOTS_DIR = "outputs/plots"

# Country selection params
EXCLUDE_ISO3       = {"USA", "DEU"}
TOP_K_COUNTRIES    = 100
MIN_NODES_PER_CTRY = 5
MIN_EDGES_PER_CTRY = 10
MIN_COVERAGE_PCT   = 10.0   # used only if coverage pickle is available

# Analysis params (for large graphs)
BETWEENNESS_SAMPLE_NODES = 200
BETWEENNESS_K_THRESHOLD  = 4000   # if nodes > this, use sampling for betweenness

# Global distance vs SCI plot
GLOBAL_SCATTER_MAX_POINTS = 200_000

# -------------------------------
# Utilities
# -------------------------------
def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def read_node_list(path):
    """Read node list and basic columns; drop duplicate nodeIDs."""
    df = pd.read_csv(
        path,
        dtype={"nodeID": int, "nodeLabel": str, "latitude": float, "longitude": float}
    )
    required = {"nodeID", "nodeLabel", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"node_list is missing columns: {missing}")
    return df.drop_duplicates("nodeID")

def read_edges_all(path):
    """Read full edge table (directed rows, weighted)."""
    df = pd.read_csv(
        path,
        dtype={"nodeID_from": int, "nodeID_to": int, "country_ISO3": "string", "weight": float}
    )
    required = {"nodeID_from", "nodeID_to", "country_ISO3", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"edges_all is missing columns: {missing}")
    return df

def read_edges_by_country(dir_path):
    """Read all edges_<ISO3>.csv files into a dict: {ISO3: DataFrame}."""
    out = {}
    pattern = os.path.join(dir_path, "edges_*.csv")
    for fp in glob.glob(pattern):
        iso = os.path.splitext(os.path.basename(fp))[0].replace("edges_", "")
        try:
            df = pd.read_csv(
                fp,
                dtype={"nodeID_from": int, "nodeID_to": int, "country_ISO3": "string", "weight": float}
            )
            if len(df) > 0:
                out[iso] = df
        except Exception as e:
            print(f"[WARN] Could not read {fp}: {e}")
    return out

def read_optional_coverage_pickle(pkl_path):
    """Read coverage_by_country pickle if present; return None if missing or unreadable."""
    if os.path.exists(pkl_path):
        try:
            return pd.read_pickle(pkl_path)
        except Exception as e:
            print(f"[WARN] coverage_by_country pickle not readable ({e}). Continuing without it.")
    return None

def select_top_countries(edges_per_country,
                         coverage_by_country,
                         exclude_iso3,
                         top_k,
                         min_nodes,
                         min_edges,
                         min_cov):
    """
    Build a summary table per country and select the top ones.
    Returns (list_of_iso3, selection_dataframe).
    Also prints the count and the list of selected ISO3.
    """
    rows = []
    for iso, e in edges_per_country.items():
        used_nodes = pd.unique(pd.concat([e["nodeID_from"], e["nodeID_to"]], ignore_index=True))
        rows.append({"country_ISO3": iso, "nodes": len(used_nodes), "edges": len(e)})

    df = pd.DataFrame(rows)

    # Optionally filter by coverage if provided
    if coverage_by_country is not None and len(coverage_by_country) > 0:
        cov = (
            coverage_by_country
            .groupby(["country_ISO3", "country_name"], dropna=False)
            .agg(coverage_pct=("coverage_pct", "max"))
            .reset_index()
        )
        df = df.merge(cov[["country_ISO3", "coverage_pct"]], on="country_ISO3", how="left")
        df["coverage_pct"] = df["coverage_pct"].fillna(0.0)
        df = df[df["coverage_pct"] >= min_cov]
    else:
        df["coverage_pct"] = np.nan  # keep column for consistency; no filter

    # Basic filters and sorting
    df = df[(df["nodes"] >= min_nodes) & (df["edges"] >= min_edges)]
    df = df[~df["country_ISO3"].isin(exclude_iso3)]
    df = df.sort_values(["edges", "nodes"], ascending=False)

    selected = df.head(top_k).copy()
    iso_list = selected["country_ISO3"].tolist()

    # Immediate print
    print(f"Selected {len(iso_list)} countries:")
    if len(iso_list) > 0:
        print(", ".join(iso_list))

    return iso_list, selected

def export_country_nodes(node_list, edges_per_country, iso_list, out_dir):
    """For each selected country, write nodes_<ISO3>.csv containing only used nodes."""
    ensure_dir(out_dir)
    core = node_list[["nodeID", "nodeLabel", "latitude", "longitude"]].drop_duplicates("nodeID")
    for iso in iso_list:
        e = edges_per_country.get(iso)
        if e is None or e.empty:
            continue
        used = pd.unique(pd.concat([e["nodeID_from"], e["nodeID_to"]], ignore_index=True))
        nodes_iso = core[core["nodeID"].isin(used)].copy()
        nodes_iso.to_csv(os.path.join(out_dir, f"nodes_{iso}.csv"), index=False)

# -------------------------------
# Graph construction and metrics
# -------------------------------
def _graph_weighted(edges_df):
    """
    Build an undirected weighted graph without self-loops.
    For each pair, sum weights across directions.
    Store 'weight' and a bounded 'distance' for shortest-path metrics:
      distance = 1 / (1 + weight / median_weight)
    """
    # remove self-loops
    e = edges_df[edges_df["nodeID_from"] != edges_df["nodeID_to"]].copy()

    # undirected pairs (a <= b)
    e["a"] = e[["nodeID_from", "nodeID_to"]].min(axis=1)
    e["b"] = e[["nodeID_from", "nodeID_to"]].max(axis=1)
    agg = e.groupby(["a", "b"], as_index=False)["weight"].sum()

    med_w = float(np.median(agg["weight"])) if len(agg) > 0 else 1.0
    if med_w <= 0:
        med_w = 1.0

    G = nx.Graph()
    for _, row in agg.iterrows():
        a = int(row["a"])
        b = int(row["b"])
        w = float(row["weight"])
        w_norm = w / med_w
        dist = 1.0 / (1.0 + w_norm)  # in (0, 1]
        G.add_edge(a, b, weight=w, distance=dist)
    return G

def _giant_component(G):
    """Return the largest connected component as a new graph."""
    if G.number_of_nodes() == 0:
        return G
    if nx.is_connected(G):
        return G
    largest_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_nodes).copy()

def _betweenness_with_guard(H):
    """
    Compute betweenness centrality using 'distance' as edge weight.
    Use node sampling for large graphs to speed up.
    """
    if H.number_of_nodes() == 0:
        return {}
    if H.number_of_nodes() > BETWEENNESS_K_THRESHOLD:
        k = min(BETWEENNESS_SAMPLE_NODES, H.number_of_nodes())
        return nx.betweenness_centrality(H, normalized=True, weight="distance", k=k, seed=42)
    return nx.betweenness_centrality(H, normalized=True, weight="distance")

def country_stats_weighted(edges_df):
    """
    Compute weighted metrics (undirected):
      - n_nodes, n_edges
      - number of components and giant component size
      - average node strength (sum of incident weights)
      - average weighted clustering (Barrat)
      - betweenness (avg and max) on the giant component, using 'distance'
      - total_sci = sum of undirected edge weights
    """
    G = _graph_weighted(edges_df)
    H = _giant_component(G)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # node strength = sum of incident edge weights
    strength = {u: sum(d.get("weight", 0.0) for _, _, d in G.edges(u, data=True)) for u in G.nodes()}
    avg_strength = float(np.mean(list(strength.values()))) if strength else 0.0

    avg_clustering = nx.average_clustering(G, weight="weight") if n_nodes > 0 else 0.0

    bet = _betweenness_with_guard(H)
    bet_avg = float(np.mean(list(bet.values()))) if bet else 0.0
    bet_max = float(np.max(list(bet.values()))) if bet else 0.0

    total_w = float(np.sum([d.get("weight", 0.0) for _, _, d in G.edges(data=True)]))

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "components": nx.number_connected_components(G) if n_nodes > 0 else 0,
        "giant_comp_size": H.number_of_nodes(),
        "avg_strength": avg_strength,
        "avg_clustering": avg_clustering,
        "betweenness_avg": bet_avg,
        "betweenness_max": bet_max,
        "total_sci": total_w,
    }

# -------------------------------
# Plot helpers
# -------------------------------
def _pretty_label(col):
    """Human-friendly axis labels."""
    mapping = {
        "n_nodes": "Number of nodes",
        "n_edges": "Number of edges",
        "giant_comp_size": "Giant component size",
        "avg_strength": "Average node strength",
        "avg_clustering": "Average clustering",
        "betweenness_avg": "Average betweenness",
        "betweenness_max": "Max betweenness",
        "total_sci": "Total SCI (sum of weights)",
        "distance_km": "Great-circle distance (km)",
        "weight": "SCI (weight)",
    }
    return mapping.get(col, col.replace("_", " ").title())

def scatter_vs_nodes(df, y_cols, out_dir=OUTPUT_PLOTS_DIR, filename_prefix="metrics_"):
    """Scatter plots of various metrics vs number of nodes."""
    ensure_dir(out_dir)
    for y in y_cols:
        if y not in df.columns:
            continue
        plt.figure(figsize=(7, 5))
        plt.scatter(df["n_nodes"], df[y], s=18, alpha=0.7)
        plt.xlabel(_pretty_label("n_nodes"))
        plt.ylabel(_pretty_label(y))
        plt.title(f"{_pretty_label(y)} vs Number of nodes")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        fname = f"{filename_prefix}{y}_vs_n_nodes.png".replace(" ", "_")
        plt.savefig(os.path.join(out_dir, fname), dpi=220)
        plt.close()

def scatter_xy(df, x, y, filename, out_dir=OUTPUT_PLOTS_DIR):
    """Generic scatter plot helper."""
    ensure_dir(out_dir)
    plt.figure(figsize=(7, 5))
    plt.scatter(df[x], df[y], s=18, alpha=0.7)
    plt.xlabel(_pretty_label(x))
    plt.ylabel(_pretty_label(y))
    plt.title(f"{_pretty_label(y)} vs {_pretty_label(x)}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=220)
    plt.close()

# -------------------------------
# Global plot: distance vs SCI
# - removes self-loops
# - great-circle distance in km
# - hexbin density plot with log-binned counts
# - writes Pearson & Spearman correlations to CSV (or just N if scipy missing)
# -------------------------------
def plot_global_distance_vs_sci(node_list, edges_all,
                                max_points=GLOBAL_SCATTER_MAX_POINTS,
                                out_dir=OUTPUT_PLOTS_DIR,
                                filename="global_distance_vs_sci.png"):
    ensure_dir(out_dir)

    coords = node_list.set_index("nodeID")[["latitude", "longitude"]]
    E = edges_all[["nodeID_from", "nodeID_to", "weight", "country_ISO3"]].copy()

    # remove self-loops
    E = E[E["nodeID_from"] != E["nodeID_to"]].copy()

    # join coordinates
    E = E.join(coords, on="nodeID_from").rename(columns={"latitude": "lat_from", "longitude": "lon_from"})
    E = E.join(coords, on="nodeID_to").rename(columns={"latitude": "lat_to", "longitude": "lon_to"})
    E = E.dropna(subset=["lat_from", "lon_from", "lat_to", "lon_to", "weight"])

    # Haversine distance (km)
    R = 6371.0
    lat1 = np.radians(E["lat_from"].values)
    lat2 = np.radians(E["lat_to"].values)
    dlat = lat2 - lat1
    dlon = np.radians(E["lon_to"].values - E["lon_from"].values)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    E["distance_km"] = 2 * R * np.arcsin(np.sqrt(a))

    # sampling for plotting
    if len(E) > max_points:
        E = E.sample(max_points, random_state=42)

    x = E["distance_km"].to_numpy()
    y = E["weight"].to_numpy()

    # hexbin plot
    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(x, y, gridsize=80, bins="log")
    plt.xlabel(_pretty_label("distance_km"))
    plt.ylabel(_pretty_label("weight"))
    plt.title("SCI vs Distance")
    cbar = plt.colorbar(hb)
    cbar.set_label("log10(count)")
    plt.grid(True, alpha=0.2)

    # correlations (also saved to CSV)
    corr_csv = os.path.join(OUTPUT_DATA_DIR, "global_distance_correlations.csv")
    try:
        import scipy.stats as stats
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_rho, spearman_p = stats.spearmanr(x, y)
        txt = f"Pearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_rho:.3f}\nN = {len(x):,}"
        plt.gcf().text(
            0.98, 0.02, txt, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, lw=0.0)
        )
        ensure_dir(OUTPUT_DATA_DIR)
        pd.DataFrame([{
            "pearson_r": pearson_r,
            "pearson_pvalue": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_pvalue": spearman_p,
            "n_points": len(x),
        }]).to_csv(corr_csv, index=False)
    except Exception:
        # scipy not available; still record N
        ensure_dir(OUTPUT_DATA_DIR)
        pd.DataFrame([{"n_points": len(x)}]).to_csv(corr_csv, index=False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=240)
    plt.close()

# ===============================
#               MAIN
# ===============================
if __name__ == "__main__":
    # Prepare output folders
    ensure_dir(OUTPUT_DATA_DIR)
    ensure_dir(OUTPUT_PLOTS_DIR)

    # Load inputs
    node_list = read_node_list(NODE_LIST_PATH)
    edges_all = read_edges_all(EDGES_ALL_PATH)
    edges_per_country = read_edges_by_country(EDGES_BY_COUNTRY_DIR)
    coverage_by_country = read_optional_coverage_pickle(CACHE_COVERAGE_PICKLE)

    # Country selection
    top_iso3, selection_df = select_top_countries(
        edges_per_country=edges_per_country,
        coverage_by_country=coverage_by_country,
        exclude_iso3=EXCLUDE_ISO3,
        top_k=TOP_K_COUNTRIES,
        min_nodes=MIN_NODES_PER_CTRY,
        min_edges=MIN_EDGES_PER_CTRY,
        min_cov=MIN_COVERAGE_PCT,
    )

    # Save selected countries
    pd.Series(top_iso3, name="country_ISO3").to_csv(
        os.path.join(OUTPUT_DATA_DIR, "selected_countries.txt"), index=False
    )
    selection_df.to_csv(os.path.join(OUTPUT_DATA_DIR, "selected_countries_table.csv"), index=False)

    # Export per-country node files
    export_country_nodes(node_list, edges_per_country, top_iso3, NODES_BY_COUNTRY_DIR)
    print(f"Saved node files for {len(top_iso3)} countries in: {NODES_BY_COUNTRY_DIR}/nodes_<ISO3>.csv")

    # Weighted analysis per country
    rows = []
    for iso in top_iso3:
        e = edges_per_country.get(iso)
        if e is None or e.empty:
            continue
        stats = country_stats_weighted(e)
        stats["country_ISO3"] = iso
        rows.append(stats)

    df_w = pd.DataFrame(rows).sort_values("n_nodes", ascending=False)
    df_w.to_csv(os.path.join(OUTPUT_DATA_DIR, "country_stats.csv"), index=False)
    print("Saved: outputs/data/country_stats.csv")

    # Scatter plots (simple axes, no point annotations)
    scatter_vs_nodes(
        df_w,
        y_cols=[
            "avg_clustering",
            "betweenness_avg",
            "betweenness_max",
            "total_sci",
            "avg_strength",
            "giant_comp_size",
            "n_edges",
        ],
        out_dir=OUTPUT_PLOTS_DIR,
        filename_prefix="metrics_",
    )
    print("Saved scatter plots in outputs/plots/")

    # Dedicated: Total SCI vs Number of nodes
    scatter_xy(
        df=df_w,
        x="n_nodes",
        y="total_sci",
        filename="total_sci_vs_n_nodes.png",
        out_dir=OUTPUT_PLOTS_DIR,
    )
    print("Saved: outputs/plots/total_sci_vs_n_nodes.png")

    # Global distance vs SCI (hexbin + correlations)
    plot_global_distance_vs_sci(
        node_list=node_list,
        edges_all=edges_all,
        max_points=GLOBAL_SCATTER_MAX_POINTS,
        out_dir=OUTPUT_PLOTS_DIR,
        filename="global_distance_vs_sci.png",
    )
    print("Saved global plot: outputs/plots/global_distance_vs_sci.png")
    print("Saved correlations: outputs/data/global_distance_correlations.csv")
