"""Webots controller - Component B: one agent of the distributed coverage swarm.

Same algorithm as component_b/swarm_sim.py (distributed mode):
  * 16-ray LiDAR features -> neural-network steering (weights in nn_policy.json, pure numpy)
  * belief map of visited 0.5 m cells, merged with neighbours' maps received over Emitter/Receiver (max-consensus)
  * nearest-unvisited-cell target selection with target claims to avoid duplicated work
Each agent logs its trajectory to outputs/webots_swarm_<id>.csv.
"""
import csv, json, math, os, pathlib
import numpy as np
from controller import Robot

SCALE, ARENA, CELL, NC = 0.2, 10.0, 0.5, 20
N_RAYS, MAX_R, SENSE_R, COMM_R = 16, 2.0, 0.75, 3.0            # map metres
DECISION_T = 0.30                                                # s per NN decision
STEP, TURN = 0.12, 0.35                                          # map m / rad per decision (as in the Python sim)
WHEEL_R, AXLE, MAX_WHEEL = 0.0205, 0.052, 6.28
IDX_SIGN = -1        # Webots lidar: index grows clockwise (left->right). Set to +1 if the robot turns the wrong way.
HERE = pathlib.Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / "outputs").exists()), HERE.parents[3])
nn = json.load(open(HERE / "nn_policy.json")); Ws = [np.array(w) for w in nn["W"]]; Bs = [np.array(b) for b in nn["b"]]


def policy(f):
    a = np.atleast_2d(f)
    for W, b in zip(Ws[:-1], Bs[:-1]): a = np.maximum(a @ W + b, 0)
    return int(np.argmax(a @ Ws[-1] + Bs[-1]))              # 0 = right, 1 = straight, 2 = left


wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
robot = Robot(); dt = int(robot.getBasicTimeStep()); me = int(robot.getName().split("_")[-1])
lm, rm = robot.getDevice("left wheel motor"), robot.getDevice("right wheel motor")
for m in (lm, rm): m.setPosition(float("inf")); m.setVelocity(0.0)
gps, imu, lidar = robot.getDevice("gps"), robot.getDevice("inertial unit"), robot.getDevice("lidar")
emitter, receiver = robot.getDevice("emitter"), robot.getDevice("receiver")
for d in (gps, imu, lidar, receiver): d.enable(dt)

cx = (np.arange(NC) + 0.5) * CELL; CX, CY = np.meshgrid(cx, cx)


def free_mask(clear=0.35):
    """A-priori map from Component A (fused occupancy) -> coarse cells at least `clear` m from obstacles/walls."""
    occ = np.load(ROOT / "outputs" / "occ_fused.npy"); res = ARENA / occ.shape[0]; ok = np.ones((NC, NC), bool)
    rows, cols = np.nonzero(occ)
    for r in range(NC):
        for c in range(NC):
            x, y = cx[c], cx[r]
            if min(x, y, ARENA - x, ARENA - y) < clear or (rows.size and np.min(np.hypot((cols + .5) * res - x, (rows + .5) * res - y)) < clear): ok[r, c] = False
    return ok


VALID = free_mask()
belief = np.zeros((NC, NC), bool); target = None; claims = {}; log = []; next_decision = 0.0; action = 1
best, stall, black = math.inf, 0, {}


def pose():
    gx, gy, _ = gps.getValues(); return gx / SCALE + ARENA / 2, gy / SCALE + ARENA / 2, imu.getRollPitchYaw()[2]


def sense16(th_unused=0.0):
    rng = lidar.getRangeImage(); n = len(rng); out = []
    for k in range(N_RAYS):                                   # k-th ray, CCW from the front (index n/2)
        i = int(round(n / 2 + IDX_SIGN * (k * 2 * math.pi / N_RAYS) / (2 * math.pi) * n)) % n
        r = rng[i] / SCALE
        out.append(min(MAX_R, r if r == r and r != float("inf") else MAX_R) / MAX_R)
    return np.array(out)


def pick(x, y, t):
    ok = VALID & ~belief
    for c, until in list(black.items()):
        if t < until: ok[c] = False
        else: del black[c]
    for (tx, ty) in claims.values(): ok &= np.hypot(CX - tx, CY - ty) > 1.5
    if not ok.any(): return None
    d = np.where(ok, np.hypot(CX - x, CY - y), np.inf); r, c = np.unravel_index(np.argmin(d), d.shape); return (int(r), int(c))


while robot.step(dt) != -1:
    t = robot.getTime(); x, y, th = pose()
    belief |= np.hypot(CX - x, CY - y) < SENSE_R
    while receiver.getQueueLength() > 0:                       # neighbours' pose, target and visited cells
        msg = json.loads(receiver.getString()); receiver.nextPacket()
        if msg["id"] != me and math.hypot(msg["x"] - x, msg["y"] - y) < COMM_R:
            belief |= np.unpackbits(np.frombuffer(bytes.fromhex(msg["visited"]), np.uint8))[:NC * NC].reshape(NC, NC).astype(bool)
            claims[msg["id"]] = tuple(msg["tgt"]) if msg["tgt"] else None
            if claims[msg["id"]] is None: del claims[msg["id"]]
    if t >= next_decision:
        next_decision = t + DECISION_T
        if target is not None:
            d = math.hypot(CX[target] - x, CY[target] - y)
            stall = 0 if d < best - 0.05 else stall + 1; best = min(best, d)
            if belief[target] or stall > 90 * 1:
                if not belief[target]: black[target] = t + 400 * DECISION_T
                target = None
        if target is None:
            target = pick(x, y, t); best, stall = math.inf, 0
        if target is None:
            action = None                                       # nothing left to cover
        else:
            bearing = wrap(math.atan2(CY[target] - y, CX[target] - x) - th)
            action = policy(np.r_[sense16(), math.sin(bearing), math.cos(bearing)])
        emitter.send(json.dumps(dict(id=me, x=x, y=y, tgt=[float(CX[target]), float(CY[target])] if target else None,
                                     visited=np.packbits(belief.ravel()).tobytes().hex())).encode())
    if action is None:
        lm.setVelocity(0.0); rm.setVelocity(0.0)
    else:
        v = STEP * SCALE / DECISION_T; w = (action - 1) * TURN / DECISION_T
        lm.setVelocity(float(np.clip((v - w * AXLE / 2) / WHEEL_R, -MAX_WHEEL, MAX_WHEEL)))
        rm.setVelocity(float(np.clip((v + w * AXLE / 2) / WHEEL_R, -MAX_WHEEL, MAX_WHEEL)))
    log.append((t, x, y, th, float(belief[VALID].mean())))

LOG = pathlib.Path(os.environ.get("ASR_LOG", ROOT / "outputs")); LOG.mkdir(exist_ok=True)
with open(LOG / f"webots_swarm_{me}.csv", "w", newline="") as f:
    cw = csv.writer(f); cw.writerow(["t", "x_m", "y_m", "theta", "own_belief_coverage"]); cw.writerows(log)