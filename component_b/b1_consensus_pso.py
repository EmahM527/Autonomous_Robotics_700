"""COMPONENT B - Task 2: Consensus + swarm intelligence.

(a) Average consensus over different communication topologies (+ random packet loss).
(b) Leader election by max-ID flooding, and re-election after the leader fails.
(c) Particle Swarm Optimisation for multi-robot task allocation (random-key encoding):
      gbest PSO (needs global broadcast) vs lbest/ring PSO (needs only neighbour comms = distributed),
      compared against exact optimum (enumeration), greedy allocation and random allocation.
    Travel costs are true shortest-path distances on the planner's inflated occupancy grid (Component A).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import itertools
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common.env import *
from component_a.a2_planning import inflate, dijkstra_field, plan, prune_path, load_grid

# ------------------------------------------------------------------ (a) consensus
def adjacency(kind, n=6, positions=None, radius=5.0):
    A = np.zeros((n, n))
    if kind == "ring":
        for i in range(n): A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    elif kind == "line":
        for i in range(n - 1): A[i, i + 1] = A[i + 1, i] = 1
    elif kind == "complete":
        A = 1 - np.eye(n)
    elif kind == "range":
        d = np.hypot(*(positions[:, None] - positions[None]).transpose(2, 0, 1)); A = ((d < radius) & (d > 0)).astype(float)
    return A


def consensus(A, x0, iters=200, eps=None, drop=0.0, rng=None):
    """x_i <- x_i + eps * sum_j a_ij (x_j - x_i); eps in (0, 2/lambda_max(L)) guarantees convergence on a connected graph."""
    n = len(x0); x = x0.astype(float).copy(); hist = [x.copy()]
    if eps is None:                                    # step minimising the spectral radius of (I - eps L): 2/(lam2+lam_n)
        ev = np.sort(np.linalg.eigvalsh(np.diag(A.sum(1)) - A)); eps = 2 / (ev[1] + ev[-1]) if ev[1] > 1e-9 else 1 / ev[-1]
    for _ in range(iters):
        M = A if drop == 0 else A * (rng.random(A.shape) > drop); M = np.triu(M, 1); M = M + M.T   # symmetric link failures
        x = x + eps * (M * (x[None, :] - x[:, None])).sum(1); hist.append(x.copy())
    return np.array(hist)


def leader_election(A, ids, alive=None):
    """Flood the maximum (score, id) each round. Returns per-round beliefs and rounds until all agree."""
    n = len(ids); alive = np.ones(n, bool) if alive is None else alive
    belief = np.where(alive, ids, -1).astype(float); hist = [belief.copy()]
    for r in range(n):
        new = belief.copy()
        for i in np.where(alive)[0]:
            nb = [j for j in np.where(A[i])[0] if alive[j]]; new[i] = max([belief[i]] + [belief[j] for j in nb])
        belief = new; hist.append(belief.copy())
        if len(set(belief[alive])) == 1: break
    return np.array(hist), r + 1

# ------------------------------------------------------------------ (c) PSO allocation
def build_problem(seed=3, n_tasks=8, n_rob=4):
    rng = np.random.default_rng(seed); grid = inflate(load_grid("fused"), ROBOT_RADIUS + 0.15)
    robots = [(0.8, 0.8), (1.4, 0.8), (0.8, 1.4), (1.4, 1.4)][:n_rob]
    reach, _ = dijkstra_field(grid, robots[0]); tasks = []       # sample only cells reachable from the robots
    while len(tasks) < n_tasks:                                  # (perception artefacts can leave free 'islands')
        p = rng.uniform(0.5, ARENA - 0.5, 2)
        if np.isfinite(reach[world_to_grid(*p)]) and all(np.hypot(*(p - q)) > 1.5 for q in tasks): tasks.append(tuple(p))
    nodes = robots + tasks; D = np.zeros((len(nodes), len(nodes)))
    for i, s in enumerate(nodes):
        g, _ = dijkstra_field(grid, s)
        for j, t in enumerate(nodes): D[i, j] = g[world_to_grid(*t)]
    assert np.isfinite(D).all(), "unreachable task"
    return grid, robots, tasks, D


def tour_table(D, n_rob, n_tasks):
    """cost[r][mask] = length of nearest-neighbour tour of robot r over the task subset `mask` (open path)."""
    T = np.zeros((n_rob, 2 ** n_tasks)); order = {}
    for r in range(n_rob):
        for m in range(2 ** n_tasks):
            left = [k for k in range(n_tasks) if m >> k & 1]; cur, L, seq = r, 0.0, []
            while left:
                k = min(left, key=lambda k: D[cur, n_rob + k]); L += D[cur, n_rob + k]; cur = n_rob + k; left.remove(k); seq.append(k)
            T[r, m] = L; order[r, m] = seq
    return T, order


def cost_of(assign, T, n_rob, lam=0.05):
    """Objective = makespan (longest robot tour) + lam * total distance."""
    masks = np.zeros(n_rob, int)
    for k, r in enumerate(assign): masks[r] |= 1 << k
    l = np.array([T[r, masks[r]] for r in range(n_rob)]); return l.max() + lam * l.sum()


def pso(T, n_rob, n_tasks, topology="gbest", swarm=30, iters=100, seed=0, w=0.72, c1=1.5, c2=1.5):
    rng = np.random.default_rng(seed); dec = lambda X: np.clip(X.astype(int), 0, n_rob - 1)
    X = rng.uniform(0, n_rob, (swarm, n_tasks)); V = rng.uniform(-1, 1, X.shape)
    f = np.array([cost_of(dec(x), T, n_rob) for x in X]); pb, pf = X.copy(), f.copy(); curve = []
    for _ in range(iters):
        if topology == "gbest": g = np.tile(pb[pf.argmin()], (swarm, 1))
        else:   # lbest ring: each particle only sees its two neighbours (local comms)
            g = np.zeros_like(X)
            for i in range(swarm):
                nb = [(i - 1) % swarm, i, (i + 1) % swarm]; g[i] = pb[nb[int(np.argmin(pf[nb]))]]
        V = w * V + c1 * rng.random(X.shape) * (pb - X) + c2 * rng.random(X.shape) * (g - X)
        X = np.clip(X + V, 0, n_rob - 1e-6); f = np.array([cost_of(dec(x), T, n_rob) for x in X])
        imp = f < pf; pb[imp], pf[imp] = X[imp], f[imp]; curve.append(pf.min())
    return dec(pb[pf.argmin()]), pf.min(), np.array(curve)


def greedy(T, D, n_rob, n_tasks):
    assign, pos, fin = [0] * n_tasks, list(range(n_rob)), np.zeros(n_rob)
    for k in range(n_tasks):                       # each task goes to the robot that would finish it earliest
        r = int(np.argmin([fin[r] + D[pos[r], n_rob + k] for r in range(n_rob)]))
        fin[r] += D[pos[r], n_rob + k]; pos[r] = n_rob + k; assign[k] = r
    return np.array(assign)


if __name__ == "__main__":
    rng = np.random.default_rng(0); n = 6
    pos = rng.uniform(0, 10, (n, 2)); x0 = rng.uniform(20, 100, n)
    print(f"initial values {np.round(x0,1)}, true average {x0.mean():.2f}")
    fig, ax = plt.subplots(2, 3, figsize=(18, 10)); rows = []
    for kind, rad in [("ring", 0), ("line", 0), ("range", 7.0), ("range", 5.0), ("complete", 0)]:
        A = adjacency(kind, n, pos, radius=rad); kind = f"range R={rad:g}m" if kind == "range" else kind; L = np.diag(A.sum(1)) - A; lam2 = np.sort(np.linalg.eigvalsh(L))[1]
        H = consensus(A, x0, 300); err = np.abs(H - x0.mean()).max(1); it = int(np.argmax(err < 1e-3)) if (err < 1e-3).any() else np.nan
        its = []                                        # average over 20 random link-failure realisations
        for sd in range(20):
            Hd = consensus(A, x0, 300, drop=0.3, rng=np.random.default_rng(sd)); errd = np.abs(Hd - x0.mean()).max(1)
            its.append(int(np.argmax(errd < 1e-3)) if (errd < 1e-3).any() else np.nan)
        itd = float(np.mean(its))
        rows.append(dict(Topology=kind, edges=int(A.sum() / 2), algebraic_connectivity=round(lam2, 3), iters_to_1e3=it, iters_30pct_loss_mean20=itd,
                         final_error=float(err[-1]), mean_preserved=bool(np.allclose(H[-1].mean(), x0.mean()))))
        ax[0, 0].semilogy(np.maximum(err, 1e-16), label=f"{kind} (λ2={lam2:.2f})"); ax[0, 1].semilogy(np.maximum(errd, 1e-16), label=kind)
        if kind == "ring": ax[0, 2].plot(H); ax[0, 2].axhline(x0.mean(), color="k", ls="--"); ax[0, 2].set(title="Ring: agent states -> average (dashed)", xlabel="iteration", ylabel="state")
    ax[0, 0].set(title="Average consensus: max error vs iteration", xlabel="iteration", ylabel="max |x_i - mean|"); ax[0, 0].legend(); ax[0, 0].grid(alpha=.3)
    ax[0, 1].set(title="Same, with 30 % random link failures per round", xlabel="iteration"); ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)
    cons = pd.DataFrame(rows).set_index("Topology"); print(cons.to_string()); cons.to_csv(OUT / "b1_consensus_table.csv")

    A = adjacency("line", n); ids = np.array([3, 17, 8, 42, 5, 11.0]); h1, r1 = leader_election(A, ids)
    alive = np.ones(n, bool); alive[3] = False; h2, r2 = leader_election(A, ids, alive)      # agent 3 (the leader) dies
    print(f"\nLeader election on a line graph: leader={h1[-1].max():.0f} after {r1} rounds (diameter {n-1});"
          f" leader dies -> line splits; each partition elects its own leader: {sorted(int(v) for v in set(h2[-1][alive]))}")
    ax[1, 0].plot(h1); ax[1, 0].set(title=f"Leader election (max-ID flooding, {r1} rounds)", xlabel="round", ylabel="believed leader ID")
    ax[1, 1].plot(h2); ax[1, 1].set(title="After leader (agent 3) fails: network partitions -> two leaders", xlabel="round")

    grid, robots, tasks, D = build_problem(); nr, nt = len(robots), len(tasks); T, order = tour_table(D, nr, nt)
    best = min(itertools.product(range(nr), repeat=nt), key=lambda a: cost_of(np.array(a), T, nr)); opt = cost_of(np.array(best), T, nr)
    rnd = np.mean([cost_of(np.random.default_rng(i).integers(0, nr, nt), T, nr) for i in range(2000)])
    gr = cost_of(greedy(T, D, nr, nt), T, nr); res, curves = {}, {}
    for topo in ["gbest", "lbest"]:
        runs = [pso(T, nr, nt, topo, seed=s) for s in range(30)]; res[topo] = np.array([r[1] for r in runs]); curves[topo] = np.array([r[2] for r in runs])
    tab = pd.DataFrame({"Method": ["Exact optimum (enumeration)", "PSO gbest (30 runs)", "PSO lbest ring (30 runs)", "Greedy earliest-finish", "Random (mean of 2000)"],
                        "Cost": [opt, res["gbest"].mean(), res["lbest"].mean(), gr, rnd],
                        "Std": [0, res["gbest"].std(), res["lbest"].std(), 0, 0],
                        "Runs_hitting_optimum_%": [100, 100 * (res["gbest"] < opt + 1e-6).mean(), 100 * (res["lbest"] < opt + 1e-6).mean(), 100 * (gr < opt + 1e-6), 0]}).set_index("Method")
    tab["Gap_to_optimum_%"] = ((tab.Cost / opt - 1) * 100); print("\nTask-allocation results (cost = makespan + 0.05*total, metres):\n", tab.round(3).to_string()); tab.round(4).to_csv(OUT / "b1_pso_results.csv")
    for topo, c in [("gbest", "tab:blue"), ("lbest", "tab:orange")]:
        m, s = curves[topo].mean(0), curves[topo].std(0); ax[1, 2].plot(m, c=c, label=f"PSO {topo}"); ax[1, 2].fill_between(range(len(m)), m - s, m + s, color=c, alpha=.2)
    ax[1, 2].axhline(opt, color="k", ls="--", label="optimum"); ax[1, 2].axhline(gr, color="g", ls=":", label="greedy"); ax[1, 2].set(title="PSO convergence (mean ± std, 30 runs)", xlabel="iteration", ylabel="cost"); ax[1, 2].legend(); ax[1, 2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "b1_consensus_pso.png", dpi=125); plt.close()

    fig, a = plt.subplots(figsize=(7.5, 7.5)); draw_map(a, title=f"PSO task allocation (cost {opt:.2f}) - one colour per robot"); cols = plt.cm.tab10.colors
    masks = np.zeros(nr, int)
    for k, r in enumerate(best): masks[r] |= 1 << k
    for r in range(nr):
        cur = robots[r]
        for k in order[r, masks[r]]:
            p = plan(grid, cur, tasks[k], "octile")["path"]; a.plot(*p.T, c=cols[r], lw=2); cur = tasks[k]
        a.plot(*robots[r], "s", c=cols[r], ms=10)
    for k, t in enumerate(tasks): a.plot(*t, "*", c=cols[best[k]], ms=15, mec="k"); a.annotate(f"T{k}", t, xytext=(5, 5), textcoords="offset points")
    plt.tight_layout(); plt.savefig(OUT / "b1_task_allocation_map.png", dpi=125); plt.close()