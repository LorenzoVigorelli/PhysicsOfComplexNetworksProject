### COMMENTS OF THE CODE AND VARIABLE NAMES WERE REVIEWED AND IMPROVED BY CHATGPT-5 ###

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import k_clique_communities

# =============================================================================
# OUTPUT FOLDERS
# =============================================================================
OUT_ROOT = Path("outputs")
CSV_DIR = OUT_ROOT / "csv"
PLOTS_DIR = OUT_ROOT / "plots"
CSV_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# THEORY AND UTILITIES
# =============================================================================
def pc_er_kclique(N: int, k: int) -> float:
    """Asymptotic percolation threshold for k-clique percolation in ER graphs."""
    return ((k - 1) * N) ** (-1.0 / (k - 1))

def pc_er_giant_component(N: int) -> float:
    """Classical ER threshold for the emergence of the giant component (k=2 case)."""
    return 1.0 / N

def graph_edge_density(G: nx.Graph) -> float:
    """Edge density p_eff = 2|E| / (|V|(|V|-1))."""
    return nx.density(G)

# =============================================================================
# K-CLIQUES AND PERCOLATION CLUSTERS
# =============================================================================
def enumerate_k_cliques(G: nx.Graph, k: int) -> List[tuple]:
    """List exactly the k-cliques using enumerate_all_cliques + filter."""
    out = []
    for C in nx.enumerate_all_cliques(G):
        if len(C) < k:
            continue
        if len(C) == k:
            out.append(tuple(sorted(C)))
        # ignore cliques larger than k here
    return out

@dataclass
class KCliqueStats:
    Phi: float                          # fraction of vertices in the giant k-clique cluster
    Psi: float                          # fraction of k-cliques in the giant k-clique cluster
    Nk_tot: int                         # total number of k-cliques
    cluster_sizes_cliques: List[int]    # cluster sizes measured in # of k-cliques
    largest_cluster_vertex_set: set     # vertex set of the largest cluster

def kclique_percolation_stats(G: nx.Graph, k: int, nk_cap: int = 1_000_000) -> KCliqueStats:
    """Compute k-clique percolation statistics using NetworkX k_clique_communities."""
    kcliques = enumerate_k_cliques(G, k)
    Nk = len(kcliques)
    if Nk == 0:
        return KCliqueStats(0.0, 0.0, 0, [], set())

    # Skip heavy counting if the number of k-cliques explodes
    if Nk > nk_cap:
        comms = list(k_clique_communities(G, k))
        if not comms:
            return KCliqueStats(0.0, np.nan, Nk, [], set())
        largest = max(comms, key=len)
        Phi = len(largest) / G.number_of_nodes()
        return KCliqueStats(Phi, np.nan, Nk, [], set(largest))

    comms = list(k_clique_communities(G, k))
    if not comms:
        return KCliqueStats(0.0, 0.0, Nk, [], set())

    comm_sets = [set(c) for c in comms]
    counts = [0] * len(comm_sets)

    # Count how many k-cliques fall in each community (at most one community per clique)
    for C in kcliques:
        Cs = set(C)
        for i, S in enumerate(comm_sets):
            if Cs.issubset(S):
                counts[i] += 1
                break

    if sum(counts) == 0:
        return KCliqueStats(0.0, 0.0, Nk, [], set())

    # Giant cluster index (by number of k-cliques)
    max_i = int(np.argmax(counts))
    Nk_star = counts[max_i]
    Psi = Nk_star / Nk
    Phi = len(comm_sets[max_i]) / G.number_of_nodes()

    cluster_sizes_cliques = sorted([c for c in counts if c > 0], reverse=True)
    largest_vertex_set = comm_sets[max_i]

    return KCliqueStats(Phi, Psi, Nk, cluster_sizes_cliques, largest_vertex_set)

def susceptibility_from_cluster_sizes(sizes: List[int], exclude_giant: bool = True) -> float:
    """Chi = sum s^2 / sum s over clusters measured in # of k-cliques."""
    if not sizes:
        return 0.0
    s = sorted(sizes, reverse=True)
    if exclude_giant and s:
        s = s[1:]
    if not s:
        return 0.0
    arr = np.asarray(s, dtype=float)
    return np.sum(arr**2) / np.sum(arr)

# =============================================================================
# EXPERIMENTS: ERDŐS–RÉNYI ONLY
# =============================================================================
@dataclass
class ExpConfigER:
    N: int
    k: int
    replicas: int
    grid_points: int
    p_window: Tuple[float, float] = (0.6, 1.4)
    seed: int = 123

def run_er(cfg: ExpConfigER) -> pd.DataFrame:
    """Sweep p around p_c(k) for ER graphs; collect Phi, Psi, and chi."""
    rng = np.random.default_rng(cfg.seed)
    pc = pc_er_kclique(cfg.N, cfg.k)
    ps = np.linspace(cfg.p_window[0] * pc, cfg.p_window[1] * pc, cfg.grid_points)

    rows = []
    for rep in range(cfg.replicas):
        for p in ps:
            G = nx.erdos_renyi_graph(cfg.N, p, seed=int(rng.integers(0, 2**31 - 1)))
            stats = kclique_percolation_stats(G, cfg.k)
            chi = susceptibility_from_cluster_sizes(stats.cluster_sizes_cliques, exclude_giant=True)
            rows.append(
                dict(
                    model="ER",
                    N=cfg.N,
                    k=cfg.k,
                    p=p,
                    p_over_pc=p / pc,
                    p_eff=graph_edge_density(G),
                    Phi=stats.Phi,
                    Psi=stats.Psi,
                    Nk_tot=stats.Nk_tot,
                    chi=chi,
                    replicate=rep,
                )
            )
    return pd.DataFrame(rows)

def run_er_classic(N: int, replicas: int, grid_points: int,
                   p_window: Tuple[float, float] = (0.6, 1.4), seed: int = 123) -> pd.DataFrame:
    """Classical ER benchmark: size of the largest connected component S."""
    rng = np.random.default_rng(seed)
    pc = pc_er_giant_component(N)
    ps = np.linspace(p_window[0] * pc, p_window[1] * pc, grid_points)
    rows = []
    for rep in range(replicas):
        for p in ps:
            G = nx.erdos_renyi_graph(N, p, seed=int(rng.integers(0, 2**31 - 1)))
            S = max((len(c) for c in nx.connected_components(G)), default=0) / N
            rows.append(dict(model="ER_classic", N=N, p=p, p_over_pc=p / pc, S=S, replicate=rep))
    return pd.DataFrame(rows)

# =============================================================================
# AGGREGATION AND ESTIMATES
# =============================================================================
def aggregate_over_replicas(df: pd.DataFrame) -> pd.DataFrame:
    """Average across replicas at each control point (p/p_c)."""
    keys = [c for c in ["model", "N", "k", "p_over_pc"] if c in df.columns]
    agg = df.groupby(keys).agg(
        Phi_mean=("Phi", "mean") if "Phi" in df.columns else ("S", "mean"),
        Psi_mean=("Psi", "mean") if "Psi" in df.columns else ("S", "mean"),
        chi_mean=("chi", "mean") if "chi" in df.columns else ("S", "mean"),
        p_eff_mean=("p_eff", "mean") if "p_eff" in df.columns else ("p_over_pc", "mean"),
        n_reps=("replicate", "nunique"),
    ).reset_index()
    return agg

def estimate_psic_per_N_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate Psi at p/p_c ≈ 1 per replica (linear interpolation near 1),
    then average across replicas for each N.
    """
    assert "Psi" in df_raw.columns and "p_over_pc" in df_raw.columns
    rows = []
    for (N, rep), sub in df_raw.groupby(["N", "replicate"]):
        s = sub.sort_values("p_over_pc").reset_index(drop=True)
        x = s["p_over_pc"].values
        y = s["Psi"].values
        idx = np.searchsorted(x, 1.0)
        if 0 < idx < len(x):
            x0, x1 = x[idx - 1], x[idx]
            y0, y1 = y[idx - 1], y[idx]
            y_at_1 = y0 + (y1 - y0) * (1.0 - x0) / (x1 - x0) if x1 != x0 else y0
        else:
            j = np.argmin(np.abs(x - 1.0))
            y_at_1 = y[j]
        rows.append({"N": int(N), "Psi_c_rep": float(y_at_1)})
    per_rep = pd.DataFrame(rows)
    tab = per_rep.groupby("N").agg(
        Psi_c_mean=("Psi_c_rep", "mean"),
        Psi_c_std=("Psi_c_rep", "std"),
        n_reps=("Psi_c_rep", "size"),
    ).reset_index()
    return tab

# =============================================================================
# PLOTTING
# =============================================================================
MARKERS = ["o", "s", "^", "D", "v", ">", "<", "p", "h", "x"]

def plot_order_parameter_ER(dfagg: pd.DataFrame, k: int, ymean: str, ylabel_math: str, outfile: Path):
    plt.figure()
    for i, N in enumerate(sorted(dfagg["N"].unique())):
        sub = dfagg[dfagg["N"] == N].sort_values("p_over_pc")
        plt.plot(sub["p_over_pc"], sub[ymean], marker=MARKERS[i % len(MARKERS)], linestyle="-",
                 label=rf"$N={N}$ (reps≈{int(sub['n_reps'].max())})")
    plt.axvline(1.0, linestyle="--")
    plt.xlabel(r"$p/p_c(k)$")
    plt.ylabel(ylabel_math)
    plt.title(rf"ER — {ylabel_math} vs $p/p_c(k)$  (k={k})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def plot_phi_collapse_multi_alpha(dfagg: pd.DataFrame, k: int, alphas: List[float], outprefix: Path):
    # individual alpha
    for alpha in alphas:
        plt.figure()
        for i, N in enumerate(sorted(dfagg["N"].unique())):
            sub = dfagg[dfagg["N"] == N].copy()
            x = (sub["p_over_pc"].values - 1.0) * (N ** alpha)
            y = sub["Phi_mean"].values
            order = np.argsort(x)
            plt.plot(x[order], y[order], marker=MARKERS[i % len(MARKERS)], linestyle="-", label=rf"$N={N}$")
        plt.xlabel(r"$[p/p_c(k)-1]\,N^\alpha$")
        plt.ylabel(r"$\Phi$")
        plt.title(rf"ER — Data collapse $\Phi$  (k={k}, $\alpha$={alpha})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outprefix.parent / f"{outprefix.stem}_alpha{alpha}.png")
        plt.close()

    # combined panel
    plt.figure()
    for i, N in enumerate(sorted(dfagg["N"].unique())):
        for j, alpha in enumerate(alphas):
            sub = dfagg[dfagg["N"] == N].copy()
            x = (sub["p_over_pc"].values - 1.0) * (N ** alpha)
            y = sub["Phi_mean"].values
            order = np.argsort(x)
            marker = MARKERS[(i + j) % len(MARKERS)]
            plt.plot(x[order], y[order], marker=marker, linestyle="-", label=rf"$N={N}, \alpha={alpha}$")
    plt.xlabel(r"$[p/p_c(k)-1]\,N^\alpha$")
    plt.ylabel(r"$\Phi$")
    plt.title(rf"ER — Data collapse $\Phi$ (k={k}, multi-$\alpha$)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outprefix.parent / f"{outprefix.stem}_multiAlpha.png")
    plt.close()

def plot_psic_vs_N_with_theory(tab: pd.DataFrame, k: int, outfile: Path):
    """Log–log plot of Psi(p_c) vs N with simple slope guides."""
    tab = tab[(tab["Psi_c_mean"] > 0) & (tab["N"] > 1)].copy()
    if len(tab) == 0:
        print("[warn] no positive data for Psi_c; skipping plot.")
        return
    xs = tab["N"].values.astype(float)
    ys = tab["Psi_c_mean"].values.astype(float)
    yerr = tab["Psi_c_std"].fillna(0.0).values.astype(float)

    plt.figure()
    plt.errorbar(xs, ys, yerr=yerr, fmt="o", linestyle="", capsize=3, label="mean ± std")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$\Psi_c$")

    # simple power-law guides (heuristic)
    if k == 3:
        for slope, lab, style in [(-k/6.0, r"$N^{-k/6}$", "--"), (1.0 - k/2.0, r"$N^{1-k/2}$", ":")]:
            A = np.exp(np.mean(np.log(ys) - slope * np.log(xs)))
            xs_line = np.linspace(xs.min(), xs.max(), 200)
            plt.plot(xs_line, A * (xs_line ** slope), style, label=lab)
        title_extra = r"$k=3$"
    else:
        slope = (-k/6.0) if k < 3 else (1.0 - k/2.0)
        A = np.exp(np.mean(np.log(ys) - slope * np.log(xs)))
        xs_line = np.linspace(xs.min(), xs.max(), 200)
        plt.plot(xs_line, A * (xs_line ** slope), "--", label=rf"slope ~ {slope:.2f}")
        title_extra = rf"$k={k}$"

    plt.title(r"$\Psi_c$ vs $N$ — " + title_extra)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

# =============================================================================
# USER CONFIG
# =============================================================================
K_VALUES = [3, 4, 5]
ER_Ns = [100, 300, 1000, 3000]
REPLICAS = 20
GRID_POINTS = 31
ALPHAS_FOR_COLLAPSE = [0.45, 0.5]
SEED = 9

CLASSIC_ENABLE = True
CLASSIC_Ns = [100, 300, 1000]

# =============================================================================
# MAIN
# =============================================================================
def main():
    results_er_raw_by_k: Dict[int, pd.DataFrame] = {}
    results_er_agg_by_kN: Dict[int, Dict[int, pd.DataFrame]] = {}

    for k in K_VALUES:
        dfs_er_raw = []
        dfs_er_agg_byN: Dict[int, pd.DataFrame] = {}

        for N in ER_Ns:
            df_raw = run_er(ExpConfigER(N=N, k=k, replicas=REPLICAS, grid_points=GRID_POINTS, seed=SEED))
            df_raw.to_csv(CSV_DIR / f"ER_raw_N{N}_k{k}.csv", index=False)
            dfs_er_raw.append(df_raw)

            df_agg = aggregate_over_replicas(df_raw)
            df_agg.to_csv(CSV_DIR / f"ER_agg_N{N}_k{k}.csv", index=False)
            dfs_er_agg_byN[N] = df_agg

        df_all_raw = pd.concat(dfs_er_raw, ignore_index=True)
        df_all_agg = pd.concat(dfs_er_agg_byN.values(), ignore_index=True)

        plot_order_parameter_ER(df_all_agg, k, "Phi_mean", r"$\Phi$", PLOTS_DIR / f"ER_phi_k{k}.png")
        plot_order_parameter_ER(df_all_agg, k, "Psi_mean", r"$\Psi$", PLOTS_DIR / f"ER_psi_k{k}.png")
        plot_phi_collapse_multi_alpha(df_all_agg, k, ALPHAS_FOR_COLLAPSE, PLOTS_DIR / f"ER_collapse_phi_k{k}.png")

        psic_tab = estimate_psic_per_N_from_raw(df_all_raw)
        psic_tab.to_csv(CSV_DIR / f"ER_psic_vsN_table_k{k}.csv", index=False)
        plot_psic_vs_N_with_theory(psic_tab, k, PLOTS_DIR / f"ER_psic_vsN_k{k}.png")

        # P(s) near p_c at the largest N
        N_star = max(ER_Ns)
        pc_star = pc_er_kclique(N_star, k)
        dfN = dfs_er_agg_byN[N_star]
        idx = int((dfN["p_over_pc"] - 1.0).abs().idxmin())
        p_star = float(dfN.loc[idx, "p_over_pc"]) * pc_star

        G_star = nx.erdos_renyi_graph(N_star, p_star, seed=999)
        stats_star = kclique_percolation_stats(G_star, k)
        cnt = Counter(stats_star.cluster_sizes_cliques)
        pd.DataFrame({"size_kcliques": list(cnt.keys()), "count": list(cnt.values())}) \
            .to_csv(CSV_DIR / f"ER_Ps_sizes_k{k}_N{N_star}.csv", index=False)

        plt.figure()
        xs = np.array(sorted(cnt.keys()))
        ys = np.array([cnt[v] for v in xs])
        if len(xs) > 0:
            plt.loglog(xs, ys, "o")
        plt.xlabel(r"Cluster size $s$ (#k-cliques)")
        plt.ylabel(r"$P(s)$ (counts)")
        plt.title(f"ER: P(s) near p_c (k={k}, N={N_star})")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"ER_Ps_k{k}_N{N_star}.png")
        plt.close()

        results_er_raw_by_k[k] = df_all_raw
        results_er_agg_by_kN[k] = dfs_er_agg_byN

    # Optional comparison: classical ER (S) vs k=2 (Phi), independent replicas
    if CLASSIC_ENABLE:
        for N in CLASSIC_Ns:
            df_classic = run_er_classic(N, replicas=REPLICAS, grid_points=GRID_POINTS, seed=SEED + 111)
            df_classic.to_csv(CSV_DIR / f"ERclassic_raw_N{N}.csv", index=False)
            agg_classic = aggregate_over_replicas(df_classic)

            df_k2 = run_er(ExpConfigER(N=N, k=2, replicas=REPLICAS, grid_points=GRID_POINTS, seed=SEED + 222))
            df_k2.to_csv(CSV_DIR / f"ER_raw_N{N}_k2.csv", index=False)
            agg_k2 = aggregate_over_replicas(df_k2)

            plt.figure()
            sub = agg_classic.sort_values("p_over_pc")
            plt.plot(sub["p_over_pc"], sub["Phi_mean"], marker="o", linestyle="-", label=fr"classical $S$, $N={N}$")
            sub2 = agg_k2.sort_values("p_over_pc")
            plt.plot(sub2["p_over_pc"], sub2["Phi_mean"], marker="s", linestyle="--", label=fr"$k=2$ $\Phi$, $N={N}$")
            plt.axvline(1.0, linestyle=":")
            plt.xlabel(r"$p/p_c$")
            plt.ylabel("Order parameter")
            plt.title(r"Classical ER $S$ vs $k=2$ (clique-percolation) $\Phi$")
            plt.legend()
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"ER_classic_vs_k2_N{N}.png")
            plt.close()

    print(f"[ok] Plots  -> {PLOTS_DIR.resolve()}")
    print(f"[ok] CSV    -> {CSV_DIR.resolve()}")

if __name__ == "__main__":
    main()
