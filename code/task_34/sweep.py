### COMMENTS OF THE CODE AND VARIABLE NAMES WERE REVIEWED AND IMPROVED BY CHATGPT-5 ###

import argparse
import csv
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# -----------------------------
# Parameters and simple helpers
# -----------------------------

class Params:
    """Container for simulation and plotting parameters."""
    def __init__(
        self,
        n: int = 500,
        avg_k: float = 6.0,        # for ER
        ba_m: int = 3,             # for BA
        seed: Optional[int] = None,
        steps: int = 5000,
        rule: str = "natural_selection",   # or "social_penalty"
        player_type: str = "A",            # ignored if mixed=True
        mixed: bool = False,
        mix_probs: Tuple[float, float, float] = (1/3, 1/3, 1/3),
        bins: int = 40,
        outdir: str = "outputs",
    ):
        self.n = n
        self.avg_k = avg_k
        self.ba_m = ba_m
        self.seed = seed
        self.steps = steps
        self.rule = rule
        self.player_type = player_type
        self.mixed = mixed
        self.mix_probs = mix_probs
        self.bins = bins
        self.outdir = outdir


def sample_strategy(player_type: str, rng: random.Random) -> Tuple[float, float]:
    """
    Draw a (p, q) strategy depending on the player type.
    A: empathetic (q = p)
    B: pragmatic  (q = 1 - p)
    C: independent (p, q independent)
    """
    if player_type == "A":
        p = rng.random()
        q = p
    elif player_type == "B":
        p = rng.random()
        q = 1.0 - p
    elif player_type == "C":
        p = rng.random()
        q = rng.random()
    else:
        raise ValueError("Unknown player type")
    return p, q


def normalize_mix_probs(probs: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalize a 3-tuple of probabilities; fall back to uniform if invalid."""
    s = sum(probs)
    if s <= 0:
        return (1/3, 1/3, 1/3)
    a, b, c = probs
    return (a / s, b / s, c / s)


# -----------------------------
# Simulator
# -----------------------------

class UltimatumSimulator:
    """
    Ultimatum game on an undirected graph.

    - Each undirected edge (i, j) is played twice: i→j and j→i.
    - Payoff rule: if p_i >= q_j then (i gets 1 - p_i, j gets p_i), else 0 to both.
    - Update rules:
        * "natural_selection": pairwise imitation with prob (Pj - Pi) / (2 * max(k_i, k_j)) if Pj > Pi.
        * "social_penalty": pick the node(s) with minimum payoff, randomly choose one,
                            and reset that node and its neighbors with fresh strategies.
    """

    def __init__(self, G: nx.Graph, params: Params):
        self.G = G.copy()
        self.params = params
        self.rng = random.Random(params.seed)

        self.p: Dict[int, float] = {u: 0.0 for u in self.G.nodes()}
        self.q: Dict[int, float] = {u: 0.0 for u in self.G.nodes()}
        self.payoff: Dict[int, float] = {u: 0.0 for u in self.G.nodes()}
        self.types: Dict[int, str] = {u: "A" for u in self.G.nodes()}

        self.last_accept_rate = 0.0
        self._init_strategies()

    def _init_strategies(self) -> None:
        """Initialize strategies for all nodes."""
        if self.params.mixed:
            weights = normalize_mix_probs(self.params.mix_probs)
            for u in self.G.nodes():
                t = self.rng.choices(["A", "B", "C"], weights=weights, k=1)[0]
                self.types[u] = t
                self.p[u], self.q[u] = sample_strategy(t, self.rng)
        else:
            t = self.params.player_type
            for u in self.G.nodes():
                self.types[u] = t
                self.p[u], self.q[u] = sample_strategy(t, self.rng)

    def _play_round(self) -> None:
        """Play one round on all edges and accumulate payoffs."""
        for u in self.G.nodes():
            self.payoff[u] = 0.0

        accepted = 0
        total = 0

        for i, j in self.G.edges():
            # i proposes to j
            total += 1
            if self.p[i] >= self.q[j]:
                self.payoff[i] += (1.0 - self.p[i])
                self.payoff[j] += self.p[i]
                accepted += 1

            # j proposes to i
            total += 1
            if self.p[j] >= self.q[i]:
                self.payoff[j] += (1.0 - self.p[j])
                self.payoff[i] += self.p[j]
                accepted += 1

        self.last_accept_rate = accepted / max(1, total)

    def _natural_selection_update(self) -> None:
        """Pairwise imitation step."""
        new_p = self.p.copy()
        new_q = self.q.copy()

        nodes = list(self.G.nodes())
        self.rng.shuffle(nodes)

        for i in nodes:
            nbrs = list(self.G.neighbors(i))
            if not nbrs:
                continue
            j = self.rng.choice(nbrs)
            Pi, Pj = self.payoff[i], self.payoff[j]
            if Pj > Pi:
                prob = (Pj - Pi) / (2.0 * max(self.G.degree[i], self.G.degree[j]))
                if self.rng.random() < prob:
                    new_p[i] = self.p[j]
                    new_q[i] = self.q[j]
                    self.types[i] = self.types[j]

        self.p = new_p
        self.q = new_q

    def _social_penalty_update(self) -> None:
        """Reset the worst node and its neighbors with fresh strategies."""
        min_payoff = min(self.payoff.values())
        candidates = [u for u, v in self.payoff.items() if v == min_payoff]
        victim = self.rng.choice(candidates)
        to_reset = set([victim]) | set(self.G.neighbors(victim))

        weights = normalize_mix_probs(self.params.mix_probs)
        for u in to_reset:
            if self.params.mixed:
                t = self.rng.choices(["A", "B", "C"], weights=weights, k=1)[0]
            else:
                t = self.params.player_type
            self.types[u] = t
            self.p[u], self.q[u] = sample_strategy(t, self.rng)

    def step(self) -> None:
        """One simulation step: play + update."""
        self._play_round()
        if self.params.rule == "natural_selection":
            self._natural_selection_update()
        else:
            self._social_penalty_update()

    # Accessors
    def get_offers(self) -> np.ndarray:
        return np.array([self.p[u] for u in self.G.nodes()])

    def get_thresholds(self) -> np.ndarray:
        return np.array([self.q[u] for u in self.G.nodes()])


# -----------------------------
# Graph builders
# -----------------------------

def make_er(n: int, avg_k: float, seed: Optional[int] = None) -> nx.Graph:
    """Erdos-Renyi with probability set by desired average degree."""
    p = max(0.0, min(1.0, avg_k / max(1, n - 1)))
    return nx.erdos_renyi_graph(n=n, p=p, seed=seed)


def make_ba(n: int, m: int, seed: Optional[int] = None) -> nx.Graph:
    """Barabasi-Albert with attachment m (clamped to [1, n-1])."""
    m = max(1, min(m, n - 1))
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)


def make_empirical_from_edgelist(path: str, directed: bool = False) -> nx.Graph:
    """Load graph from edgelist (CSV with commas or whitespace-separated)."""
    cu = nx.DiGraph() if directed else nx.Graph()

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        # Fallback to karate club graph if file missing/empty
        return nx.Graph(nx.karate_club_graph())

    with open(path, "r") as f:
        first = f.readline()
        f.seek(0)
        if "," in first:
            # Skip header if it contains letters
            if any(c.isalpha() for c in first):
                next(f)
            return nx.parse_edgelist(f, delimiter=",", nodetype=int, create_using=cu)
        return nx.read_edgelist(path, nodetype=int, create_using=cu)


def build_graph(topology: str, params: Params, edgelist: Optional[str]) -> nx.Graph:
    """Dispatch graph construction by topology tag."""
    if topology == "ER":
        return make_er(params.n, params.avg_k, params.seed)
    if topology == "BA":
        m = params.ba_m if params.ba_m > 0 else max(1, int(round(params.avg_k / 2)))
        return make_ba(params.n, m, params.seed)
    if topology == "EMP":
        if not edgelist:
            raise ValueError("EMP requires --edgelist PATH")
        return make_empirical_from_edgelist(edgelist)
    raise ValueError("Unknown topology")


# -----------------------------
# History utilities
# -----------------------------

def run_with_history(
    sim: UltimatumSimulator,
    steps: int,
    record_times: Optional[List[int]] = None,
    record_every: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Run the simulation for 'steps'. Save (p, q) snapshots at specific times and/or
    at a fixed interval. Returns a dict with:
        - "times": array of saved times
        - "P":     [T x n] offers at saved times
        - "Q":     [T x n] thresholds at saved times
        - "accept":[T]     acceptance rate from the last round before each snapshot
    """
    times_saved: List[int] = []
    Ps: List[np.ndarray] = []
    Qs: List[np.ndarray] = []
    Acc: List[float] = []

    rec_set = set(record_times or [])

    for t in range(1, steps + 1):
        sim.step()

        take = False
        if record_every and record_every > 0 and (t % record_every == 0):
            take = True
        if t in rec_set:
            take = True

        if take:
            times_saved.append(t)
            Ps.append(sim.get_offers().copy())
            Qs.append(sim.get_thresholds().copy())
            Acc.append(sim.last_accept_rate)

    if not Ps:
        # If no snapshot was requested, store only the final state.
        times_saved = [steps]
        Ps = [sim.get_offers().copy()]
        Qs = [sim.get_thresholds().copy()]
        Acc = [sim.last_accept_rate]

    return {
        "times": np.array(times_saved, dtype=int),
        "P": np.stack(Ps, axis=0),
        "Q": np.stack(Qs, axis=0),
        "accept": np.array(Acc, dtype=float),
    }


def save_history_npz(history: Dict, outpath: str) -> None:
    """Save history dict as compressed NPZ."""
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    np.savez_compressed(outpath, **history)


def save_history_csv(
    history: Dict,
    degrees: np.ndarray,
    outpath: str,
    max_rows: int = 5_000_000,
) -> None:
    """
    Save long-format CSV: (t, node, k, p, q).
    Use max_rows to prevent extremely large files.
    """
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    T, n = history["P"].shape

    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "node", "k", "p", "q"])
        rows = 0
        for it, t in enumerate(history["times"]):
            pvec = history["P"][it]
            qvec = history["Q"][it]
            for u in range(n):
                w.writerow([int(t), int(u), int(degrees[u]), float(pvec[u]), float(qvec[u])])
                rows += 1
                if rows >= max_rows:
                    return


# -----------------------------
# Plotting helpers
# -----------------------------

def density_curves_over_times(values_by_time: Dict[int, np.ndarray], bins: int):
    """Compute density histograms over [0,1] for multiple times."""
    times = sorted(values_by_time.keys())
    edges = np.linspace(0, 1, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    curves = []
    for t in times:
        hist, _ = np.histogram(values_by_time[t], bins=edges, density=True)
        curves.append(hist)
    return np.array(times), centers, curves


def plot_multi_time(
    centers: np.ndarray,
    curves: List[np.ndarray],
    times: List[int],
    title: str,
    outpath: str,
    xlabel: str = "p",
    ylabel: str = "D(p)",
) -> None:
    """Plot multiple density curves on the same axes."""
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    markers = ["o", "s", "^", "*", "d", "x", "P", "v"]

    for i, (y, t) in enumerate(zip(curves, times)):
        plt.plot(centers, y, marker=markers[i % len(markers)], linewidth=1.5, label=f"t={t}")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_heatmap_pq(p: np.ndarray, q: np.ndarray, bins: int, title: str, outpath: str) -> None:
    """2D histogram of (p, q)."""
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    H, xedges, yedges = np.histogram2d(p, q, bins=bins, range=[[0, 1], [0, 1]], density=True)
    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, H.T)
    plt.xlabel("p")
    plt.ylabel("q")
    plt.title(title)
    plt.colorbar(label="density")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pk(
    degrees: np.ndarray,
    values_by_time: Dict[int, np.ndarray],
    which: str,
    title: str,
    outpath: str,
) -> None:
    """
    Plot <val>_k vs k at multiple times.
    'which' is either 'p' or 'q' to set axis label.
    """
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)

    times = sorted(values_by_time.keys())
    Ks = np.unique(degrees)
    markers = ["o", "s", "^", "*", "d", "x", "P", "v"]

    for i, t in enumerate(times):
        vec = values_by_time[t]
        means = []
        for k in Ks:
            mask = (degrees == k)
            means.append(vec[mask].mean() if mask.any() else np.nan)
        plt.plot(Ks, means, marker=markers[i % len(markers)], linewidth=1.5, label=f"t={t}")

    plt.xlabel("k (degree)")
    ylabel = r"$\langle p\rangle_k$" if which.lower() == "p" else r"$\langle q\rangle_k$"
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} — {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ultimatum — paper-like figures + history + sweeps")

    # Graph and dynamics
    p.add_argument("--topology", choices=["ER", "BA", "EMP"], default="ER")
    p.add_argument("--rule", choices=["natural_selection", "social_penalty"], default="natural_selection")
    p.add_argument("--type", choices=["A", "B", "C"], default="A")
    p.add_argument("--mixed", action="store_true")
    p.add_argument("--mix-probs", type=float, nargs=3, default=(1/3, 1/3, 1/3), metavar=("pA", "pB", "pC"))
    p.add_argument("-n", type=int, default=500)
    p.add_argument("--avg-k", type=float, default=6.0)
    p.add_argument("--ba-m", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--edgelist", type=str, default=None)

    # Times and histograms
    p.add_argument(
        "--times",
        type=int,
        nargs="+",
        default=[1, 100, 1000, 10000, 20000],
        help="Times at which to build D(p) and D(q). Simulation runs up to max(times).",
    )
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--name", type=str, default=None)

    # History save
    p.add_argument("--save-history-npz", action="store_true", help="Save (p,q) history as NPZ.")
    p.add_argument("--save-history-csv", action="store_true", help="Save (p,q) history as long CSV (large).")
    p.add_argument("--history-every", type=int, default=0, help="If >0, sample every K steps in addition to --times.")

    # Extra plots
    p.add_argument("--plot-q", action="store_true", help="Create D(q) multi-time figure.")
    p.add_argument(
        "--heatmap-times",
        type=int,
        nargs="*",
        default=[],
        help="Times for joint (p,q) heatmap. If empty, only use max(times).",
    )
    p.add_argument("--plot-pk", action="store_true", help="Create <p>_k vs k (and optionally <q>_k).")
    p.add_argument("--plot-qk", action="store_true", help="If set, also create <q>_k vs k.")

    # Sweep
    p.add_argument("--sweep", action="store_true", help="Parameter sweep ER vs BA; include EMP if --edgelist is set.")
    p.add_argument("--sweep-n", type=int, nargs="*", default=[], help="List of n for sweep (ER/BA).")
    p.add_argument("--sweep-avgk", type=float, nargs="*", default=[], help="List of <k> for ER sweep.")
    p.add_argument("--sweep-bam", type=int, nargs="*", default=[], help="List of m for BA sweep.")
    p.add_argument("--sweep-steps", type=int, default=5000, help="Simulation steps in sweep.")
    p.add_argument("--sweep-tailfrac", type=float, default=0.1, help="Final fraction of snapshots used for averaging.")

    return p.parse_args()


# -----------------------------
# MAIN
# -----------------------------

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.sweep:
        # ----------------- PARAMETER SWEEP ER vs BA (+ EMP if provided) -----------------
        n_list = args.sweep_n or [args.n]
        avgk_list = args.sweep_avgk or [args.avg_k]  # ER
        bam_list = args.sweep_bam or [args.ba_m]     # BA

        report_path = os.path.join(args.outdir, "sweep_results.csv")
        with open(report_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "topology", "n", "avg_k", "ba_m", "rule", "type", "mixed",
                "steps", "tailfrac", "seed",
                "mean_p", "std_p", "mean_q", "std_q", "accept_rate"
            ])

            def tail_stats(hist):
                """Average over the last W snapshots."""
                T = hist["P"].shape[0]
                W = max(1, int(np.floor(T * args.sweep_tailfrac)))
                P_tail = hist["P"][-W:, :].reshape(-1)
                Q_tail = hist["Q"][-W:, :].reshape(-1)
                mean_p, std_p = float(np.mean(P_tail)), float(np.std(P_tail))
                mean_q, std_q = float(np.mean(Q_tail)), float(np.std(Q_tail))
                accept = float(np.mean(hist["accept"][-W:]))
                return mean_p, std_p, mean_q, std_q, accept

            for n in n_list:
                seed = args.seed

                # ER configs
                for avgk in avgk_list:
                    params = Params(
                        n=n, avg_k=avgk, ba_m=args.ba_m, seed=seed,
                        steps=args.sweep_steps, rule=args.rule,
                        player_type=args.type, mixed=args.mixed,
                        mix_probs=tuple(args.mix_probs), bins=args.bins, outdir=args.outdir
                    )
                    G = build_graph("ER", params, args.edgelist)
                    sim = UltimatumSimulator(G, params)
                    hist = run_with_history(sim, steps=params.steps,
                                            record_every=max(1, int(params.steps * args.sweep_tailfrac)))
                    mean_p, std_p, mean_q, std_q, accept = tail_stats(hist)
                    w.writerow([
                        "ER", n, avgk, "", args.rule, args.type, args.mixed,
                        params.steps, args.sweep_tailfrac, seed,
                        mean_p, std_p, mean_q, std_q, accept
                    ])

                # BA configs
                for m in bam_list:
                    params = Params(
                        n=n, avg_k=args.avg_k, ba_m=m, seed=seed,
                        steps=args.sweep_steps, rule=args.rule,
                        player_type=args.type, mixed=args.mixed,
                        mix_probs=tuple(args.mix_probs), bins=args.bins, outdir=args.outdir
                    )
                    G = build_graph("BA", params, args.edgelist)
                    sim = UltimatumSimulator(G, params)
                    hist = run_with_history(sim, steps=params.steps,
                                            record_every=max(1, int(params.steps * args.sweep_tailfrac)))
                    mean_p, std_p, mean_q, std_q, accept = tail_stats(hist)
                    w.writerow([
                        "BA", n, "", m, args.rule, args.type, args.mixed,
                        params.steps, args.sweep_tailfrac, seed,
                        mean_p, std_p, mean_q, std_q, accept
                    ])

            # EMP (single row) if edgelist provided
            if args.edgelist:
                params = Params(
                    n=0, avg_k=0.0, ba_m=0, seed=args.seed,
                    steps=args.sweep_steps, rule=args.rule,
                    player_type=args.type, mixed=args.mixed,
                    mix_probs=tuple(args.mix_probs), bins=args.bins, outdir=args.outdir
                )
                G = build_graph("EMP", params, args.edgelist)
                n_emp = G.number_of_nodes()
                avgk_emp = (2.0 * G.number_of_edges()) / max(1, n_emp)
                params.n = n_emp

                sim = UltimatumSimulator(G, params)
                hist = run_with_history(sim, steps=params.steps,
                                        record_every=max(1, int(params.steps * args.sweep_tailfrac)))
                mean_p, std_p, mean_q, std_q, accept = tail_stats(hist)
                w.writerow([
                    "EMP", n_emp, avgk_emp, "", args.rule, args.type, args.mixed,
                    params.steps, args.sweep_tailfrac, args.seed,
                    mean_p, std_p, mean_q, std_q, accept
                ])

        print(f"[SWEEP] Summary saved to: {report_path}")
        return

    # ----------------- SINGLE EXPERIMENT WITH FIGURES -----------------
    params = Params(
        n=args.n, avg_k=args.avg_k, ba_m=args.ba_m, seed=args.seed,
        steps=max(args.times), rule=args.rule, player_type=args.type,
        mixed=args.mixed, mix_probs=tuple(args.mix_probs),
        bins=args.bins, outdir=args.outdir,
    )

    G = build_graph(args.topology, params, args.edgelist)
    sim = UltimatumSimulator(G, params)

    # History at requested times (and optionally at a regular interval)
    rec_every = args.history_every if args.history_every > 0 else None
    history = run_with_history(
        sim, steps=params.steps, record_times=args.times, record_every=rec_every
    )
    times = list(map(int, history["times"]))

    # D(p) at multiple times
    values_p = {t: history["P"][i] for i, t in enumerate(times)}
    tlist, centers, curves = density_curves_over_times(values_p, bins=args.bins)
    fname = args.name or f"Dp_{args.topology}_{'MIX' if args.mixed else args.type}_{args.rule}"
    outpath = os.path.join(args.outdir, f"{fname}.png")
    plot_multi_time(
        centers,
        curves,
        list(tlist),
        f"The Ultimatum Game in complex networks\n{args.topology}, "
        f"{'mixed' if args.mixed else 'type ' + args.type}, {args.rule}",
        outpath,
    )
    print(f"[OK] Saved: {outpath}")

    # D(q) at multiple times (optional)
    if args.plot_q:
        values_q = {t: history["Q"][i] for i, t in enumerate(times)}
        tlist_q, centers_q, curves_q = density_curves_over_times(values_q, bins=args.bins)
        out_q = os.path.join(args.outdir, f"Dq_{args.topology}_{'MIX' if args.mixed else args.type}_{args.rule}.png")
        plot_multi_time(
            centers_q,
            curves_q,
            list(tlist_q),
            f"D(q) — {args.topology}, "
            f"{'mixed' if args.mixed else 'type ' + args.type}, {args.rule}",
            out_q,
            xlabel="q",
            ylabel="D(q)",
        )
        print(f"[OK] Saved: {out_q}")

    # (p,q) heatmaps at chosen times
    heat_times = args.heatmap_times or [times[-1]]
    for T in heat_times:
        if T not in times:
            # choose the closest available snapshot
            T = min(times, key=lambda x: abs(x - T))
        idx = times.index(T)
        pvec = history["P"][idx]
        qvec = history["Q"][idx]
        out_h = os.path.join(
            args.outdir,
            f"heatmap_pq_t{T}_{args.topology}_{'MIX' if args.mixed else args.type}_{args.rule}.png",
        )
        plot_heatmap_pq(pvec, qvec, bins=args.bins, title=f"Heatmap (p,q) — t={T}", outpath=out_h)
        print(f"[OK] Saved: {out_h}")

    # Degree-conditioned measures
    if args.plot_pk or args.plot_qk:
        degs = np.array([G.degree[u] for u in G.nodes()], dtype=int)
        if args.plot_pk:
            values_p = {t: history["P"][i] for i, t in enumerate(times)}
            out_pk = os.path.join(
                args.outdir,
                f"p_by_degree_{args.topology}_{'MIX' if args.mixed else args.type}_{args.rule}.png",
            )
            plot_pk(
                degs,
                values_p,
                which="p",
                title=f"{args.topology}, {'mixed' if args.mixed else 'type ' + args.type}, {args.rule}",
                outpath=out_pk,
            )
            print(f"[OK] Saved: {out_pk}")
        if args.plot_qk:
            values_q = {t: history["Q"][i] for i, t in enumerate(times)}
            out_qk = os.path.join(
                args.outdir,
                f"q_by_degree_{args.topology}_{'MIX' if args.mixed else args.type}_{args.rule}.png",
            )
            plot_pk(
                degs,
                values_q,
                which="q",
                title=f"{args.topology}, {'mixed' if args.mixed else 'type ' + args.type}, {args.rule}",
                outpath=out_qk,
            )
            print(f"[OK] Saved: {out_qk}")

    # Optional history saves
    base = args.name or f"{args.topology}_{'MIX' if args.mixed else args.type}_{args.rule}"
    if args.save_history_npz:
        npz_path = os.path.join(args.outdir, f"history_{base}.npz")
        save_history_npz(history, npz_path)
        print(f"[OK] History (NPZ): {npz_path}")
    if args.save_history_csv:
        degs = np.array([G.degree[u] for u in G.nodes()], dtype=int)
        csv_path = os.path.join(args.outdir, f"history_{base}.csv")
        save_history_csv(history, degs, csv_path)
        print(f"[OK] History (CSV): {csv_path}")


if __name__ == "__main__":
    main()
