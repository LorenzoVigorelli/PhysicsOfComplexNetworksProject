# -*- coding: utf-8 -*-
# ============================================================
# SCI Analysis (from saved files): country selection, per-country node export,
# weighted network metrics, and global distance-vs-SCI plot.
# Saves ALL data to outputs/data/ and ALL plots to outputs/plots/.
# Immediately prints how many & which countries are selected.
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
EDGES_BY_COUNTRY_DIR  = "edges_by_country"          # holds edges_<ISO3>.csv
NODES_BY_COUNTRY_DIR  = "nodes_by_country"          # where to save nodes_<ISO3>.csv
CACHE_COVERAGE_PICKLE = "cache/coverage_by_country.pkl"  # optional

# Output folders
OUTPUT_DATA_DIR  = "outputs/data"
OUTPUT_PLOTS_DIR = "outputs/plots"

# Country selection params
EXCLUDE_ISO3       = {"USA", "DEU"}  # countries to skip in analysis
TOP_K_COUNTRIES    = 100
MIN_NODES_PER_CTRY = 5
MIN_EDGES_PER_CTRY = 10
MIN_COVERAGE_PCT   = 10.0   # used only if coverage_by_country pickle is available

# Analysis params (for large graphs)
BETWEENNESS_SAMPLE_NODES = 200
BETWEENNESS_K_THRESHOLD  = 4000   # above this #nodes, use k-sampling for betweenness

# Global distance vs SCI plot
GLOBAL_SCATTER_MAX_POINTS = 200_000

# -------------------------------
# Utilities
# -------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_node_list(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"nodeID": int, "nodeLabel": str, "latitude": float, "longitude": float})
    need = {"nodeID","nodeLabel","latitude","longitude"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"node_list is missing columns: {missing}")
    return df.drop_duplicates("nodeID")

def read_edges_all(path: str) -> pd.DataFrame:
    # expected: nodeID_from,nodeID_to,country_ISO3,weight
    df = pd.read_csv(path, dtype={"nodeID_from": int, "nodeID_to": int, "country_ISO3": "string", "weight": float})
    need = {"nodeID_from","nodeID_to","country_ISO3","weight"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"edges_all is missing columns: {missing}")
    return df

def read_edges_by_country(dir_path: str) -> dict[str, pd.DataFrame]:
    d: dict[str, pd.DataFrame] = {}
    for fp in glob.glob(os.path.join(dir_path, "edges_*.csv")):
        iso = os.path.splitext(os.path.basename(fp))[0].replace("edges_", "")
        try:
            df = pd.read_csv(fp, dtype={"nodeID_from": int, "nodeID_to": int, "country_ISO3": "string", "weight": float})
            if len(df):
                d[iso] = df
        except Exception as e:
            print(f"[WARN] Could not read {fp}: {e}")
    return d

def read_optional_coverage_pickle(pkl_path: str) -> pd.DataFrame | None:
    if os.path.exists(pkl_path):
        try:
            cov = pd.read_pickle(pkl_path)
            return cov
        except Exception as e:
            print(f"[WARN] coverage_by_country pickle not readable ({e}). Continuing without it.")
    return None

def select_top_countries(edges_per_country: dict[str, pd.DataFrame],
                         coverage_by_country: pd.DataFrame | None,
                         exclude_iso3: set[str],
                         top_k: int,
                         min_nodes: int,
                         min_edges: int,
                         min_cov: float) -> tuple[list[str], pd.DataFrame]:
    """
    Returns (iso3_list, full_selection_df) and prints immediately how many & which countries.
    """
    rows = []
    for iso, e in edges_per_country.items():
        used = pd.unique(pd.concat([e["nodeID_from"], e["nodeID_to"]], ignore_index=True))
        rows.append({"country_ISO3": iso, "nodes": len(used), "edges": len(e)})
    df = pd.DataFrame(rows)

    if coverage_by_country is not None and len(coverage_by_country):
        cov_agg = (coverage_by_country
                   .groupby(["country_ISO3","country_name"], dropna=False)
                   .agg(coverage_pct=("coverage_pct","max"))
                   .reset_index())
        df = df.merge(cov_agg[["country_ISO3","coverage_pct"]], on="country_ISO3", how="left")
        df["coverage_pct"] = df["coverage_pct"].fillna(0.0)
        df = df[df["coverage_pct"] >= min_cov]
    else:
        df["coverage_pct"] = np.nan  # do not filter on coverage if missing

    df = df[(df["nodes"] >= min_nodes) & (df["edges"] >= min_edges)]
    df = df[~df["country_ISO3"].isin(exclude_iso3)]
    df = df.sort_values(["edges","nodes"], ascending=False)
    selected = df.head(top_k).copy()

    top_iso3 = selected["country_ISO3"].tolist()

    # --- Immediate print to terminal ---
    print(f"Selected {len(top_iso3)} countries:")
    if len(top_iso3):
        print(", ".join(top_iso3))

    return top_iso3, selected

def export_country_nodes(node_list: pd.DataFrame,
                         edges_per_country: dict[str, pd.DataFrame],
                         iso_list: list[str],
                         out_dir: str):
    ensure_dir(out_dir)
    node_core = node_list[["nodeID","nodeLabel","latitude","longitude"]].drop_duplicates("nodeID")
    for iso in iso_list:
        e = edges_per_country.get(iso)
        if e is None or e.empty:
            continue
        used = pd.unique(pd.concat([e["nodeID_from"], e["nodeID_to"]], ignore_index=True))
        nodes_iso = node_core[node_core["nodeID"].isin(used)].copy()
        out_path = os.path.join(out_dir, f"nodes_{iso}.csv")
        nodes_iso.to_csv(out_path, index=False)

# -------------------------------
# Graphs: weighted construction & metrics
# -------------------------------
def _graph_weighted(edges_df: pd.DataFrame) -> nx.Graph:
    """Undirected, no self-loops; sum weights for both directions.
       Stores 'weight' and bounded 'distance' = 1/(1 + w_norm) with w_norm = weight / median_weight."""
    # drop self-loops
    e = edges_df[edges_df["nodeID_from"] != edges_df["nodeID_to"]].copy()
    # undirected pair
    e["a"] = e[["nodeID_from","nodeID_to"]].min(axis=1)
    e["b"] = e[["nodeID_from","nodeID_to"]].max(axis=1)
    agg = e.groupby(["a","b"], as_index=False)["weight"].sum()

    med_w = float(np.median(agg["weight"])) if len(agg) else 1.0
    if med_w <= 0:
        med_w = 1.0

    G = nx.Graph()
    for _, row in agg.iterrows():
        a, b, w = int(row["a"]), int(row["b"]), float(row["weight"])
        w_norm = w / med_w
        dist = 1.0 / (1.0 + w_norm)   # in (0,1]
        G.add_edge(a, b, weight=w, distance=dist)
    return G

def _giant_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return G
    if nx.is_connected(G):
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()

def _betweenness_with_guard(H: nx.Graph) -> dict[int, float]:
    """Use sampling for betweenness when graphs are large; 'distance' used as edge distance."""
    if H.number_of_nodes() == 0:
        return {}
    if H.number_of_nodes() > BETWEENNESS_K_THRESHOLD:
        k = min(BETWEENNESS_SAMPLE_NODES, H.number_of_nodes())
        return nx.betweenness_centrality(H, normalized=True, weight="distance", k=k, seed=42)
    return nx.betweenness_centrality(H, normalized=True, weight="distance")

def country_stats_weighted(edges_df: pd.DataFrame) -> dict:
    """
    Weighted metrics (undirected):
      - average node strength (sum of incident weights)
      - weighted clustering (Barrat)
      - betweenness (on giant component, with 'distance')
      - total_sci = sum of undirected edge weights
    """
    G = _graph_weighted(edges_df)
    H = _giant_component(G)
    n, m = G.number_of_nodes(), G.number_of_edges()

    strength = {u: sum(d.get("weight",0.0) for _,_,d in G.edges(u, data=True)) for u in G.nodes()}
    avg_strength = float(np.mean(list(strength.values()))) if strength else 0.0

    clust_w = nx.average_clustering(G, weight="weight") if n > 0 else 0.0

    bet_w = _betweenness_with_guard(H)
    betw_avg = float(np.mean(list(bet_w.values()))) if bet_w else 0.0
    betw_max = float(np.max(list(bet_w.values()))) if bet_w else 0.0

    total_w = float(np.sum([d.get("weight",0.0) for _,_,d in G.edges(data=True)]))

    return dict(
        n_nodes=n, n_edges=m,
        components=nx.number_connected_components(G) if n > 0 else 0,
        giant_comp_size=H.number_of_nodes(),
        avg_strength=avg_strength,
        avg_clustering=clust_w,
        betweenness_avg=betw_avg,
        betweenness_max=betw_max,
        total_sci=total_w,
    )

# -------------------------------
# Plot helpers (simple titles)
# -------------------------------
def _pretty_label(col: str) -> str:
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

def scatter_vs_nodes(df: pd.DataFrame, y_cols: list[str],
                     out_dir: str = OUTPUT_PLOTS_DIR, filename_prefix: str = "metrics_"):
    ensure_dir(out_dir)
    for y in y_cols:
        if y not in df.columns:
            continue
        plt.figure(figsize=(7,5))
        plt.scatter(df["n_nodes"], df[y], s=18, alpha=0.7)
        plt.xlabel(_pretty_label("n_nodes"))
        plt.ylabel(_pretty_label(y))
        plt.title(f"{_pretty_label(y)} vs Number of nodes")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        fname = f"{filename_prefix}{y}_vs_n_nodes.png".replace(" ", "_")
        plt.savefig(os.path.join(out_dir, fname), dpi=220)
        plt.close()

def scatter_xy(df: pd.DataFrame, x: str, y: str, filename: str,
               out_dir: str = OUTPUT_PLOTS_DIR):
    ensure_dir(out_dir)
    plt.figure(figsize=(7,5))
    plt.scatter(df[x], df[y], s=18, alpha=0.7)
    plt.xlabel(_pretty_label(x))
    plt.ylabel(_pretty_label(y))
    plt.title(f"{_pretty_label(y)} vs {_pretty_label(x)}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=220)
    plt.close()

# -------------------------------
# Global plot: distance vs SCI (no self-loops)
#  - uses km
#  - hexbin (density)
#  - shows Pearson & Spearman correlations (and saves them to CSV)
# -------------------------------
def plot_global_distance_vs_sci(node_list: pd.DataFrame,
                                edges_all: pd.DataFrame,
                                max_points: int = GLOBAL_SCATTER_MAX_POINTS,
                                out_dir: str = OUTPUT_PLOTS_DIR,
                                filename: str = "global_distance_vs_sci.png"):
    ensure_dir(out_dir)

    coords = node_list.set_index("nodeID")[["latitude","longitude"]]
    E = edges_all[["nodeID_from","nodeID_to","weight","country_ISO3"]].copy()

    # Remove self-loops
    E = E[E["nodeID_from"] != E["nodeID_to"]].copy()

    # Join coordinates
    E = E.join(coords, on="nodeID_from").rename(columns={"latitude":"lat_from","longitude":"lon_from"})
    E = E.join(coords, on="nodeID_to").rename(columns={"latitude":"lat_to","longitude":"lon_to"})
    E = E.dropna(subset=["lat_from","lon_from","lat_to","lon_to","weight"])

    # Haversine (km)
    R = 6371.0  # km
    lat1 = np.radians(E["lat_from"].values)
    lat2 = np.radians(E["lat_to"].values)
    dlat = lat2 - lat1
    dlon = np.radians(E["lon_to"].values - E["lon_from"].values)
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    E["distance_km"] = 2*R*np.arcsin(np.sqrt(a))

    # Sample after all columns exist
    if len(E) > max_points:
        E = E.sample(max_points, random_state=42)

    x = E["distance_km"].to_numpy()
    y = E["weight"].to_numpy()

    # Hexbin density; log counts help with heavy tails
    plt.figure(figsize=(8,6))
    hb = plt.hexbin(x, y, gridsize=80, bins='log')
    plt.xlabel(_pretty_label("distance_km"))
    plt.ylabel(_pretty_label("weight"))
    plt.title("SCI vs Distance")
    cbar = plt.colorbar(hb)
    cbar.set_label("log10(count)")
    plt.grid(True, alpha=0.2)

    # Correlation annotations (and save to CSV)
    corr_out = os.path.join(OUTPUT_DATA_DIR, "global_distance_correlations.csv")
    try:
        import scipy.stats as st
        rho_p, p_p = st.pearsonr(x, y)
        rho_s, p_s = st.spearmanr(x, y)
        txt = f"Pearson r = {rho_p:.3f}\nSpearman ρ = {rho_s:.3f}\nN = {len(x):,}"
        plt.gcf().text(0.98, 0.02, txt, ha="right", va="bottom", fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, lw=0.0))
        # save to CSV
        ensure_dir(OUTPUT_DATA_DIR)
        pd.DataFrame([{
            "pearson_r": rho_p, "pearson_pvalue": p_p,
            "spearman_rho": rho_s, "spearman_pvalue": p_s,
            "n_points": len(x)
        }]).to_csv(corr_out, index=False)
    except Exception:
        # in case scipy is not available, still save N
        ensure_dir(OUTPUT_DATA_DIR)
        pd.DataFrame([{"n_points": len(x)}]).to_csv(corr_out, index=False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=240)
    plt.close()

# ===============================
#             MAIN
# ===============================
if __name__ == "__main__":
    # Output folders
    ensure_dir(OUTPUT_DATA_DIR)
    ensure_dir(OUTPUT_PLOTS_DIR)

    # Load files
    node_list = read_node_list(NODE_LIST_PATH)
    edges_all = read_edges_all(EDGES_ALL_PATH)
    edges_per_country = read_edges_by_country(EDGES_BY_COUNTRY_DIR)
    coverage_by_country = read_optional_coverage_pickle(CACHE_COVERAGE_PICKLE)

    # Country selection + immediate print
    top_iso3, selection_df = select_top_countries(
        edges_per_country=edges_per_country,
        coverage_by_country=coverage_by_country,
        exclude_iso3=EXCLUDE_ISO3,
        top_k=TOP_K_COUNTRIES,
        min_nodes=MIN_NODES_PER_CTRY,
        min_edges=MIN_EDGES_PER_CTRY,
        min_cov=MIN_COVERAGE_PCT
    )

    # Save selected countries
    pd.Series(top_iso3, name="country_ISO3").to_csv(
        os.path.join(OUTPUT_DATA_DIR, "selected_countries.txt"), index=False
    )
    selection_df.to_csv(os.path.join(OUTPUT_DATA_DIR, "selected_countries_table.csv"), index=False)

    # Export per-country node files
    export_country_nodes(node_list, edges_per_country, top_iso3, NODES_BY_COUNTRY_DIR)
    print(f"✅ Saved node files for {len(top_iso3)} countries in: {NODES_BY_COUNTRY_DIR}/nodes_<ISO3>.csv")

    # -------- Weighted analysis --------
    w_rows = []
    for iso in top_iso3:
        e = edges_per_country.get(iso)
        if e is None or e.empty:
            continue
        stats = country_stats_weighted(e)
        stats["country_ISO3"] = iso
        w_rows.append(stats)
    df_w = pd.DataFrame(w_rows).sort_values("n_nodes", ascending=False)
    df_w.to_csv(os.path.join(OUTPUT_DATA_DIR, "country_stats.csv"), index=False)
    print("✅ Saved: outputs/data/country_stats.csv")

    # Scatter plots (simple titles, no point annotations)
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
        filename_prefix="metrics_"
    )
    print("✅ Saved scatter plots in outputs/plots/")

    # Dedicated: Total SCI vs Number of nodes
    scatter_xy(
        df=df_w,
        x="n_nodes",
        y="total_sci",
        filename="total_sci_vs_n_nodes.png",
        out_dir=OUTPUT_PLOTS_DIR
    )
    print("✅ Saved: outputs/plots/total_sci_vs_n_nodes.png")

    # -------- Global distance vs SCI (hexbin + Pearson/Spearman) --------
    plot_global_distance_vs_sci(
        node_list, edges_all,
        max_points=GLOBAL_SCATTER_MAX_POINTS,
        out_dir=OUTPUT_PLOTS_DIR,
        filename="global_distance_vs_sci.png"
    )
    print("✅ Saved global plot: outputs/plots/global_distance_vs_sci.png")
    print("✅ Saved correlations: outputs/data/global_distance_correlations.csv")
