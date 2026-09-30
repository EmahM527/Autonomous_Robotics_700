"""Shared multi-agent simulation library for Component B.

* 16-ray range sensing + feature vector (used by the expert, the neural network and the Webots controller)
* rule-based EXPERT policy (teacher for imitation learning)
* NumpyPolicy: pure-numpy MLP forward pass (no sklearn needed at run-time / inside Webots)
* Swarm: coverage task with two coordination architectures
    - 'distributed' : local comms (range-limited), belief-map merging, target-claim deconfliction
    - 'centralised' : one coordinator with global knowledge assigns targets (single point of failure)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json
import numpy as np
from common.env import *

N_RAYS, MAX_R, STEP, TURN = 16, 2.0, 0.12, 0.35       # 16 rays (22.5 deg apart), 2 m range, 0.12 m/step, 0.35 rad turn
CELL = 0.5                                            # coverage-map resolution
NC = int(ARENA / CELL)
SENSE_RADIUS = 0.75                                   # a cell counts as covered when robot is this close
ACTIONS = np.array([-1, 0, 1])                        # -1 = turn right, 0 = straight, +1 = turn left


def sense(x, y, th):
    """Normalised ranges of 16 rays, index 0 = front, counter-clockwise. Works on arrays."""
    x, y, th = np.asarray(x, float), np.asarray(y, float), np.asarray(th, float)
    ang = th[..., None] + np.arange(N_RAYS) * 2 * np.pi / N_RAYS
    return raycast(x[..., None], y[..., None], ang, MAX_R) / MAX_R


def features(rng8, bearing_err):
    return np.concatenate([rng8, np.sin(bearing_err)[..., None], np.cos(bearing_err)[..., None]], axis=-1)


def expert_action(f):
    """Rule-based reactive teacher. f: (..., N_RAYS+2). Returns index into ACTIONS (0=right,1=straight,2=left).
    1) obstacle ahead -> turn toward the more open side; 2) otherwise steer toward the target,
    but never turn *into* an obstacle that is close on that side.
    Thresholds (0.8 m front, 0.8 m side) were tuned on episode seeds 0-149; evaluation uses seeds 1000+."""
    r = f[..., :N_RAYS] * MAX_R
    bearing = np.arctan2(f[..., N_RAYS], f[..., N_RAYS + 1])
    front = np.minimum.reduce([r[..., 0], r[..., 1], r[..., -1]])
    left_open, right_open = r[..., 1:6].mean(-1), r[..., -5:].mean(-1)
    avoid = np.where(left_open > right_open, 2, 0)
    left_tight = np.minimum.reduce([r[..., 1], r[..., 2], r[..., 3]]) < 0.8
    right_tight = np.minimum.reduce([r[..., -1], r[..., -2], r[..., -3]]) < 0.8
    seek = np.where((bearing > 0.2) & ~left_tight, 2, np.where((bearing < -0.2) & ~right_tight, 0, 1))
    return np.where(front < 0.8, avoid, seek)


class NumpyPolicy:
    def __init__(self, path=None, weights=None):
        d = weights if weights is not None else json.load(open(path))
        self.W = [np.array(w) for w in d["W"]]; self.b = [np.array(b) for b in d["b"]]

    def __call__(self, f):
        a = np.atleast_2d(f)
        for W, b in zip(self.W[:-1], self.b[:-1]):
            a = np.maximum(a @ W + b, 0)
        return np.argmax(a @ self.W[-1] + self.b[-1], axis=-1)


def free_cells():
    """Coarse cells whose centre is at least 0.35 m from obstacles/walls."""
    cx = (np.arange(NC) + 0.5) * CELL
    X, Y = np.meshgrid(cx, cx)
    return clearance(X, Y) > 0.35, X, Y


class Swarm:
    def __init__(self, n=5, mode="distributed", policy=None, seed=0, comm_range=3.0,
                 fail_agent=None, fail_coordinator=None, fail_at=300):
        self.rng = np.random.default_rng(seed)
        self.n, self.mode, self.policy, self.rc = n, mode, policy, comm_range
        self.fail_agent, self.fail_coord, self.fail_at = fail_agent, fail_coordinator, fail_at
        self.valid, self.X, self.Y = free_cells()
        # start cluster in the lower-left corner
        self.x = 1.0 + 0.5 * self.rng.random(n); self.y = 0.8 + 0.5 * self.rng.random(n)
        self.th = self.rng.uniform(-np.pi, np.pi, n)
        self.alive = np.ones(n, bool)
        self.belief = np.zeros((n, NC, NC), bool)             # per-agent visited map
        self.truth = np.zeros((NC, NC), bool)                 # ground-truth coverage (for scoring)
        self.target = [None] * n; self.best = np.full(n, np.inf); self.stall = np.zeros(n, int)
        self.black = [dict() for _ in range(n)]               # cell -> step until which it is ignored
        self.t = 0; self.collisions = 0; self.near = 0; self.msgs = 0
        self.traj = [[] for _ in range(n)]; self.cov_hist = []

    # ---- helpers
    def _mark(self):
        for i in np.where(self.alive)[0]:
            m = np.hypot(self.X - self.x[i], self.Y - self.y[i]) < SENSE_RADIUS
            self.belief[i] |= m; self.truth |= m

    def _pick(self, i, visited, claimed):
        """Nearest unvisited, unclaimed, non-blacklisted free cell for agent i."""
        ok = self.valid & ~visited
        for (r, c), until in list(self.black[i].items()):
            if self.t < until: ok[r, c] = False
            else: del self.black[i][(r, c)]
        for (cx, cy) in claimed:
            ok &= np.hypot(self.X - cx, self.Y - cy) > 1.5
        if not ok.any():
            return None
        d = np.where(ok, np.hypot(self.X - self.x[i], self.Y - self.y[i]), np.inf)
        r, c = np.unravel_index(np.argmin(d), d.shape)
        self.best[i] = np.inf; self.stall[i] = 0
        return (int(r), int(c))

    def _cell_xy(self, rc):
        return (rc[1] + 0.5) * CELL, (rc[0] + 0.5) * CELL

    # ---- coordination layers
    def _coordinate_distributed(self):
        alive = np.where(self.alive)[0]
        A = np.zeros((self.n, self.n), bool)
        for i in alive:
            for j in alive:
                if i != j and np.hypot(self.x[i] - self.x[j], self.y[i] - self.y[j]) < self.rc: A[i, j] = True
        self.msgs += int(A.sum())
        old = self.belief.copy()
        for i in alive:
            for j in np.where(A[i])[0]: self.belief[i] |= old[j]            # merge visited maps (max-consensus)
        for i in alive:
            tgt = self.target[i]
            if tgt is not None and (self.belief[i][tgt] or self._stalled(i, tgt)):
                if not self.belief[i][tgt]: self.black[i][tgt] = self.t + 400
                tgt = self.target[i] = None
            if tgt is None:
                claimed = [self._cell_xy(self.target[j]) for j in np.where(A[i])[0] if self.target[j] is not None]
                self.target[i] = self._pick(i, self.belief[i], claimed)

    def _coordinate_central(self):
        alive = np.where(self.alive)[0]
        self.msgs += 2 * len(alive)                                         # pose uplink + command downlink
        coord_up = not (self.fail_coord is not None and self.t >= self.fail_coord)
        glob = self.truth.copy()
        need = []
        for i in alive:
            tgt = self.target[i]
            if tgt is not None and (glob[tgt] or self._stalled(i, tgt)):
                if not glob[tgt]: self.black[i][tgt] = self.t + 400
                tgt = self.target[i] = None
            if tgt is None: need.append(i)
        if not coord_up:
            return                                                          # no new orders: robots idle when done
        claimed = [self._cell_xy(self.target[j]) for j in alive if self.target[j] is not None]
        for i in need:                                                      # greedy assignment, global knowledge
            self.target[i] = self._pick(i, glob, claimed)
            if self.target[i] is not None: claimed.append(self._cell_xy(self.target[i]))

    def _stalled(self, i, tgt):
        d = np.hypot(*(np.array(self._cell_xy(tgt)) - [self.x[i], self.y[i]]))
        if d < self.best[i] - 0.05: self.best[i], self.stall[i] = d, 0
        else: self.stall[i] += 1
        return self.stall[i] > 90

    # ---- one simulation step
    def step(self):
        if self.fail_agent is not None and self.t == self.fail_at:
            self.alive[self.fail_agent] = False; self.target[self.fail_agent] = None
        self._mark()
        (self._coordinate_distributed if self.mode == "distributed" else self._coordinate_central)()
        idx = np.where(self.alive)[0]
        have = [i for i in idx if self.target[i] is not None]
        if have:
            h = np.array(have)
            tx = np.array([self._cell_xy(self.target[i])[0] for i in h]); ty = np.array([self._cell_xy(self.target[i])[1] for i in h])
            bearing = wrap(np.arctan2(ty - self.y[h], tx - self.x[h]) - self.th[h])
            f = features(sense(self.x[h], self.y[h], self.th[h]), bearing)
            a = ACTIONS[(self.policy or expert_action)(f)]
            self.th[h] = wrap(self.th[h] + a * TURN)
            nx, ny = self.x[h] + STEP * np.cos(self.th[h]), self.y[h] + STEP * np.sin(self.th[h])
            bad = clearance(nx, ny) < ROBOT_RADIUS
            self.collisions += int(bad.sum())
            self.x[h] = np.where(bad, self.x[h], nx); self.y[h] = np.where(bad, self.y[h], ny)
        for i in idx:
            self.traj[i].append((self.x[i], self.y[i]))
        for a in idx:
            for b in idx:
                if a < b and np.hypot(self.x[a] - self.x[b], self.y[a] - self.y[b]) < 0.3: self.near += 1
        self.cov_hist.append(self.truth[self.valid].mean()); self.t += 1

    def run(self, steps=1500, stop_at=0.995):
        for _ in range(steps):
            self.step()
            if self.cov_hist[-1] >= stop_at: break
        return self