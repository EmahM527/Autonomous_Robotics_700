"""Webots controller - Component A: waypoint navigation with PID heading control + LiDAR safety layer.

Waypoints come from the Python pipeline (outputs/waypoints.csv: fused map -> A* -> pruned) in MAP metres.
Webots world is scaled by SCALE (see webots/generate_worlds.py), so positions/speeds are converted here.
Logs the true (GPS/IMU) trajectory to outputs/webots_trajectory.csv for the planned-vs-actual plot.
"""
import csv, json, math, os, pathlib
import numpy as np
from controller import Robot

SCALE, ARENA = 0.2, 10.0
WHEEL_R, AXLE, MAX_WHEEL = 0.0205, 0.052, 6.28      # e-puck geometry (m, m, rad/s)
V_MAP, LOOKAHEAD, W_MAX = 0.5, 0.5, 3.0              # map-metres/s, map-metres, rad/s
HERE = pathlib.Path(__file__).resolve()
ROOT = next(p for p in HERE.parents if (p / "outputs").exists()) if any((p / "outputs").exists() for p in HERE.parents) else HERE.parents[3]
OUT = ROOT / "outputs"

gains = json.load(open(OUT / "pid_gains.json")); KP, KI, KD = gains["kp"], gains["ki"], gains["kd"]
wp = np.loadtxt(OUT / "waypoints.csv", delimiter=",", skiprows=1)


def resample(w, step=0.05):
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(w, axis=0).T))]; u = np.r_[np.arange(0, s[-1], step), s[-1]]
    return np.c_[np.interp(u, s, w[:, 0]), np.interp(u, s, w[:, 1])]


path = resample(wp)
robot = Robot(); dt = int(robot.getBasicTimeStep()); T = dt / 1000.0
lm, rm = robot.getDevice("left wheel motor"), robot.getDevice("right wheel motor")
for m in (lm, rm): m.setPosition(float("inf")); m.setVelocity(0.0)
gps, imu, lidar = robot.getDevice("gps"), robot.getDevice("inertial unit"), robot.getDevice("lidar")
for d in (gps, imu, lidar): d.enable(dt)

wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
integral, prev_th, idx, log = 0.0, None, 0, []
while robot.step(dt) != -1:
    gx, gy, _ = gps.getValues(); th = imu.getRollPitchYaw()[2]
    x, y = gx / SCALE + ARENA / 2, gy / SCALE + ARENA / 2                 # -> map metres
    idx += int(np.argmin(np.hypot(*(path[idx:idx + 60] - [x, y]).T)))
    j = idx
    while j < len(path) - 1 and math.hypot(*(path[j] - [x, y])) < LOOKAHEAD: j += 1
    err = wrap(math.atan2(path[j][1] - y, path[j][0] - x) - th)
    if abs(err) < 0.3: integral = float(np.clip(integral + err * T, -1, 1))          # integral separation
    d = 0.0 if prev_th is None else wrap(th - prev_th) / T; prev_th = th
    w = float(np.clip(KP * err + KI * integral - KD * d, -W_MAX, W_MAX))            # same PID as a3_control.py
    dgoal = math.hypot(*(path[-1] - [x, y]))
    v = V_MAP * max(0.0, math.cos(err)) ** 2 * min(1.0, 1.2 * dgoal + 0.05)
    rng = lidar.getRangeImage(); n = len(rng); k = int(n * 30 / 360)                 # front cone (index n/2 = front)
    front = min(r for r in rng[n // 2 - k: n // 2 + k + 1] if r == r) / SCALE if rng else 9.9   # -> map metres
    v *= float(np.clip((front - 0.17) / 0.35, 0.2, 1.0))                             # safety layer (as in a4)
    vw = v * SCALE                                                                    # world m/s
    lm.setVelocity(float(np.clip((vw - w * AXLE / 2) / WHEEL_R, -MAX_WHEEL, MAX_WHEEL)))
    rm.setVelocity(float(np.clip((vw + w * AXLE / 2) / WHEEL_R, -MAX_WHEEL, MAX_WHEEL)))
    log.append((robot.getTime(), x, y, th, err, v, front))
    if dgoal < 0.1:
        lm.setVelocity(0.0); rm.setVelocity(0.0)
        print(f"[a_navigator] goal reached in {robot.getTime():.1f} s"); break

LOG = pathlib.Path(os.environ.get("ASR_LOG", OUT)); LOG.mkdir(exist_ok=True)
with open(LOG / "webots_trajectory.csv", "w", newline="") as f:
    cw = csv.writer(f); cw.writerow(["t", "x_m", "y_m", "theta", "heading_err", "v_map", "front_range_m"]); cw.writerows(log)