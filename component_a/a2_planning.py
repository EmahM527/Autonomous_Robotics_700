"""COMPONENT A - Task 2: Motion planning on an occupancy grid.

* Occupancy grid from the fused perception map, inflated by the robot radius (+ margin).
* Dijkstra and A* (Euclidean, Octile, Manhattan heuristics) on an 8-connected grid.
* Comparison: nodes expanded, path length, run-time.
* Helpers reused later: path pruning (line-of-sight), resampling, full Dijkstra field.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import heapq, math, time
import numpy as np, cv2, pandas as pd
import matplotlib.pyplot as plt
from common.env import *

SQ2 = math.sqrt(2)
MOVES = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
         (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2)]


def inflate(occ, radius=ROBOT_RADIUS + 0.05):
    """Configuration-space obstacle growth: robot becomes a point."""
    r = int(math.ceil(radius / RES))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    g = cv2.dilate(occ.astype(np.uint8), k).astype(bool)
    g[:r, :] = g[-r:, :] = True; g[:, :r] = g[:, -r:] = True      # arena walls
    return g


HEURISTICS = {
    "euclidean": lambda dr, dc: RES * math.hypot(dr, dc),
    "manhattan": lambda dr, dc: RES * (abs(dr) + abs(dc)),        # NOT admissible on 8-connected grid
    "octile": lambda dr, dc: RES * (abs(dr) + abs(dc) + (SQ2 - 2) * min(abs(dr), abs(dc))),
}


def search(grid, start, goal, heuristic=None):
    """Generic best-first search. heuristic=None -> Dijkstra. goal=None -> full Dijkstra field.
    start/goal are (row, col). Cost is in metres."""
    h = HEURISTICS[heuristic] if heuristic else (lambda dr, dc: 0.0)
    H, W = grid.shape
    g = np.full((H, W), np.inf); g[start] = 0
    parent = np.full((H, W, 2), -1, np.int32)
    closed = np.zeros((H, W), bool)
    pq = [(h(goal[0] - start[0], goal[1] - start[1]) if goal else 0.0, start)]
    expanded = 0
    while pq:
        f, (r, c) = heapq.heappop(pq)
        if closed[r, c]:
            continue
        closed[r, c] = True; expanded += 1
        if goal and (r, c) == goal:
            break
        for dr, dc, w in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or grid[nr, nc] or closed[nr, nc]:
                continue
            if dr and dc and (grid[r + dr, c] or grid[r, c + dc]):   # forbid corner cutting
                continue
            ng = g[r, c] + w * RES
            if ng < g[nr, nc]:
                g[nr, nc] = ng; parent[nr, nc] = (r, c)
                hh = h(goal[0] - nr, goal[1] - nc) if goal else 0.0
                heapq.heappush(pq, (ng + hh, (nr, nc)))
    return g, parent, closed, expanded


def reconstruct(parent, start, goal):
    path, cur = [], goal
    while cur != start:
        path.append(cur)
        cur = tuple(parent[cur])
        if cur[0] < 0:
            return None
    path.append(start)
    return path[::-1]


def plan(grid, start_xy, goal_xy, heuristic=None, repeats=1):
    s, t = world_to_grid(*start_xy), world_to_grid(*goal_xy)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        g, parent, closed, exp = search(grid, s, t, heuristic)
        times.append(time.perf_counter() - t0)
    cells = reconstruct(parent, s, t)
    if cells is None:
        return dict(path=None, length=np.inf, expanded=exp, time_ms=1e3 * np.median(times), explored=closed)
    xy = np.array([grid_to_world(r, c) for r, c in cells])
    return dict(path=xy, length=float(g[t]), expanded=exp, time_ms=1e3 * float(np.median(times)), explored=closed)


def dijkstra_field(grid, src_xy):
    """Shortest-path distance (m) from src to every free cell."""
    g, parent, _, _ = search(grid, world_to_grid(*src_xy), None, None)
    return g, parent


def _los(grid, a, b):
    n = int(np.hypot(*(np.array(b) - np.array(a))) / RES * 2) + 2
    for t in np.linspace(0, 1, n):
        r, c = world_to_grid(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        if grid[r, c]:
            return False
    return True


def prune_path(grid, xy):
    """Greedy line-of-sight shortcutting -> sparse waypoints (removes grid 'staircase')."""
    out, i = [xy[0]], 0
    while i < len(xy) - 1:
        j = len(xy) - 1
        while j > i + 1 and not _los(grid, xy[i], xy[j]):
            j -= 1
        out.append(xy[j]); i = j
    return np.array(out)


def resample(wp, step=0.05):
    seg = np.hypot(*np.diff(wp, axis=0).T); s = np.r_[0, np.cumsum(seg)]
    u = np.arange(0, s[-1], step); u = np.r_[u, s[-1]]
    return np.c_[np.interp(u, s, wp[:, 0]), np.interp(u, s, wp[:, 1])]


def compare(grid):
    rows, results = [], {}
    for name, hname in [("Dijkstra", None), ("A* Euclidean", "euclidean"),
                        ("A* Octile", "octile"), ("A* Manhattan", "manhattan")]:
        r = plan(grid, START, GOAL, hname, repeats=10); results[name] = r
        rows.append(dict(Algorithm=name, Nodes_expanded=r["expanded"], Path_length_m=round(r["length"], 3),
                         Runtime_ms=round(r["time_ms"], 1)))
    df = pd.DataFrame(rows).set_index("Algorithm")
    opt = df.loc["Dijkstra", "Path_length_m"]
    df["Excess_vs_optimal_%"] = ((df.Path_length_m / opt - 1) * 100).round(2)
    df["Nodes_saved_%"] = ((1 - df.Nodes_expanded / df.loc["Dijkstra", "Nodes_expanded"]) * 100).round(1)
    return df, results


def plot_comparison(grid, results, df):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.2))
    for a, (name, r) in zip(ax, results.items()):
        img = np.where(grid, 0.0, np.where(r["explored"], 0.72, 1.0))
        a.imshow(img, origin="lower", extent=(0, ARENA, 0, ARENA), cmap="gray", vmin=0, vmax=1)
        a.plot(*r["path"].T, "r-", lw=2)
        a.plot(*START, "go", ms=9); a.plot(*GOAL, "b*", ms=13)
        a.set_title(f"{name}\n{r['expanded']} nodes, {r['length']:.2f} m, {r['time_ms']:.0f} ms")
        a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
    fig.suptitle("Black = inflated obstacles, grey = explored nodes, red = path, green = start, blue star = goal", y=1.02)
    plt.tight_layout(); plt.savefig(OUT / "a2_planning_comparison.png", dpi=130, bbox_inches="tight"); plt.close()


def load_grid(kind="fused"):
    f = OUT / f"occ_{kind}.npy"
    if not f.exists():
        from component_a.a1_perception import run_perception
        run_perception()
    return np.load(f)


if __name__ == "__main__":
    grid = inflate(load_grid("fused"))
    df, res = compare(grid)
    print(df.to_string()); df.to_csv(OUT / "a2_planning_comparison.csv")
    plot_comparison(grid, res, df)
    wp = prune_path(grid, res["A* Octile"]["path"])
    np.savetxt(OUT / "waypoints.csv", wp, delimiter=",", header="x_m,y_m", comments="")
    print(f"\nPruned waypoints ({len(wp)}):\n", np.round(wp, 2))