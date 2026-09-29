"""Shared arena definition + geometry helpers (single source of truth).

The same map is used by every Python script in this project, and later by
the generated Webots worlds. Units: metres. Arena is 10 x 10 m, origin
bottom-left.
"""
import os
import pathlib
import numpy as np
import matplotlib

if not os.environ.get("ASR_SHOW"):
    matplotlib.use("Agg")          # headless-safe; set ASR_SHOW=1 to open windows

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

ARENA = 10.0
RES = 0.1                          # occupancy-grid resolution (m/cell)
N = int(round(ARENA / RES))        # 100 x 100 cells
ROBOT_RADIUS = 0.15
START = (1.0, 1.0)
GOAL = (9.0, 9.0)

# axis-aligned box obstacles: (x_min, y_min, x_max, y_max)
OBSTACLES = [
    (2.0, 2.0, 3.5, 5.0),
    (4.5, 3.0, 5.5, 8.0),
    (6.5, 1.0, 8.0, 2.5),
    (6.5, 5.0, 9.0, 6.0),
    (1.0, 7.0, 3.5, 8.0),
    (7.0, 7.5, 8.0, 9.0),
]
LANDMARKS = [(1.0, 4.0), (5.0, 1.0), (9.0, 3.5)]   # green discs
LANDMARK_R = 0.25


def world_to_grid(x, y):
    """(x, y) metres -> (row, col). Row index follows +y."""
    return int(np.clip(y / RES, 0, N - 1)), int(np.clip(x / RES, 0, N - 1))


def grid_to_world(r, c):
    return (c + 0.5) * RES, (r + 0.5) * RES


def ground_truth_grid():
    """True occupancy (bool NxN): cell is occupied if its centre lies in an obstacle."""
    ys, xs = np.mgrid[0:N, 0:N]
    cx, cy = (xs + 0.5) * RES, (ys + 0.5) * RES
    occ = np.zeros((N, N), bool)
    for x0, y0, x1, y1 in OBSTACLES:
        occ |= (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
    return occ


def clearance(x, y):
    """Distance from point(s) to the nearest obstacle / arena wall (vectorised)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = np.minimum.reduce([x, y, ARENA - x, ARENA - y])
    for x0, y0, x1, y1 in OBSTACLES:
        dx = np.maximum.reduce([x0 - x, np.zeros_like(x), x - x1])
        dy = np.maximum.reduce([y0 - y, np.zeros_like(y), y - y1])
        d = np.minimum(d, np.hypot(dx, dy))
    return d


def raycast(ox, oy, angles, max_range):
    """Ray/box intersection (slab method). ox, oy, angles broadcast together.
    Returns range to the first obstacle or arena wall, capped at max_range."""
    ox, oy, angles = np.broadcast_arrays(np.asarray(ox, float), np.asarray(oy, float),
                                         np.asarray(angles, float))
    dx, dy = np.cos(angles), np.sin(angles)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dx > 0, (ARENA - ox) / dx, np.where(dx < 0, -ox / dx, np.inf))
        ty = np.where(dy > 0, (ARENA - oy) / dy, np.where(dy < 0, -oy / dy, np.inf))
        r = np.minimum(np.minimum(tx, ty), max_range)
        for x0, y0, x1, y1 in OBSTACLES:
            t1x, t2x = (x0 - ox) / dx, (x1 - ox) / dx
            t1y, t2y = (y0 - oy) / dy, (y1 - oy) / dy
            tmin = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
            tmax = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
            hit = (tmax >= tmin) & (tmin > 0)
            r = np.where(hit, np.minimum(r, tmin), r)
    return r


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def draw_map(ax, occ=None, title=None):
    """Draw arena on a Matplotlib axis (occ overrides true obstacles if given)."""
    import matplotlib.patches as mp
    if occ is None:
        for x0, y0, x1, y1 in OBSTACLES:
            ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="0.25", ec="k"))
    else:
        ax.imshow(occ, origin="lower", extent=(0, ARENA, 0, ARENA), cmap="Greys", alpha=0.85)
    ax.set_xlim(0, ARENA); ax.set_ylim(0, ARENA); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    if title:
        ax.set_title(title)