"""COMPONENT A - Task 4: Closed-loop integration and evaluation.

perception (a1)  ->  inflated grid  ->  A* (a2)  ->  LOS pruning + resampling
                 ->  pure-pursuit target -> PID heading control (a3) -> unicycle (with lag/delay)
                 ->  noisy pose feedback + online front-LiDAR safety layer (speed scaling)

Each perception variant (LiDAR only / camera only / fused / ground truth) is planned on and then
*executed and scored against the true world*, exposing the perception <-> planning <-> control trade-offs.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common.env import *
from component_a.a1_perception import run_perception
from component_a.a2_planning import inflate, plan, prune_path, resample
from component_a.a3_control import Unicycle, PID, W_MAX

DT = 0.05
LOOKAHEAD = 0.5
MARGIN = 0.15          # planning safety margin (m) added to robot radius; sized from the sweep below
POS_NOISE, HDG_NOISE, W_BIAS_RUN = 0.01, 0.01, 0.1       # odometry-like pose noise, drift disturbance


def follow_path(path, gains, vmax=0.6, seed=0, safety=True, t_max=150.0):
    rng = np.random.default_rng(seed)
    robot = Unicycle(path[0, 0], path[0, 1], np.arctan2(*(path[5] - path[0])[::-1]) + 0.6, dt=DT, w_bias=W_BIAS_RUN)
    pid = PID(*gains); goal = path[-1]; idx = 0
    log = dict(t=[], x=[], y=[], th=[], cte=[], herr=[], v=[], clr=[], slow=0)
    for k in range(int(t_max / DT)):
        est = np.array([robot.x, robot.y]) + rng.normal(0, POS_NOISE, 2)       # noisy pose estimate
        th_est = robot.th + rng.normal(0, HDG_NOISE)
        d = np.hypot(*(path[idx:idx + 60] - est).T); idx += int(np.argmin(d))  # closest path point (monotone)
        j = idx
        while j < len(path) - 1 and np.hypot(*(path[j] - est)) < LOOKAHEAD:
            j += 1
        tgt = path[j]
        herr = float(wrap(np.arctan2(tgt[1] - est[1], tgt[0] - est[0]) - th_est))
        w = pid.step(herr, -herr, DT)                                          # meas := -herr keeps D on measurement
        dgoal = np.hypot(*(goal - est))
        v = vmax * max(0.0, np.cos(herr)) ** 2 * min(1.0, 1.2 * dgoal + 0.05)
        if safety:                                                             # online front-LiDAR layer
            a = robot.th + np.linspace(-0.5, 0.5, 9)
            dmin = raycast(robot.x, robot.y, a, 1.0).min()
            sc = np.clip((dmin - 0.17) / 0.35, 0.2, 1.0); log["slow"] += sc < 1.0; v *= sc
        robot.step(v, w)
        cte = np.hypot(*(path - np.array([robot.x, robot.y])).T).min()
        for key, val in zip(["t", "x", "y", "th", "cte", "herr", "v", "clr"],
                            [k * DT, robot.x, robot.y, robot.th, cte, herr, robot.v, float(clearance(robot.x, robot.y))]):
            log[key].append(val)
        if dgoal < 0.08 and abs(robot.v) < 0.1:
            break
    return {k: (np.array(v) if isinstance(v, list) else v) for k, v in log.items()}


def evaluate(name, occ, gains, vmax=0.6, seed=0, margin=MARGIN):
    grid = inflate(occ, ROBOT_RADIUS + margin)
    t0 = time.perf_counter(); r = plan(grid, START, GOAL, "octile"); tp = 1e3 * (time.perf_counter() - t0)
    if r["path"] is None:
        return dict(Map=name, Result="NO PATH"), None, None
    wp = prune_path(grid, r["path"]); dense = resample(wp, 0.05)
    lg = follow_path(dense, gains, vmax, seed)
    plen = float(np.hypot(*np.diff(np.c_[lg["x"], lg["y"]], axis=0).T).sum())
    reached = np.hypot(lg["x"][-1] - GOAL[0], lg["y"][-1] - GOAL[1]) < 0.15
    row = dict(Map=name, Planned_len_m=round(r["length"], 2), Pruned_len_m=round(float(np.hypot(*np.diff(wp, axis=0).T).sum()), 2),
               Actual_len_m=round(plen, 2), Time_to_goal_s=round(lg["t"][-1], 1),
               RMS_CTE_m=round(float(np.sqrt((lg["cte"] ** 2).mean())), 3), Max_CTE_m=round(float(lg["cte"].max()), 3),
               Min_true_clearance_m=round(float(lg["clr"].min()), 3),
               Collision=bool((lg["clr"] < ROBOT_RADIUS).any()), Reached=bool(reached),
               Plan_ms=round(tp, 1), Safety_slowdowns=int(lg["slow"]))
    return row, dense, lg


if __name__ == "__main__":
    gains = tuple(json.load(open(OUT / "pid_gains.json")).values())
    per = run_perception(save=True)
    maps = {"Ground truth": per["gt"], "LiDAR only": per["maps"]["LiDAR only"],
            "Camera only": per["maps"]["Camera only"], "Fused": per["maps"]["Fused (LiDAR+camera)"]}
    rows, runs = [], {}
    for name, occ in maps.items():
        row, dense, lg = evaluate(name, occ, gains); rows.append(row); runs[name] = (dense, lg)
    df = pd.DataFrame(rows).set_index("Map"); print(df.T.to_string()); df.to_csv(OUT / "a4_performance_table.csv")

    dense, lg = runs["Fused"]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    draw_map(ax[0], title="Fused pipeline: planned vs actual trajectory")
    ax[0].plot(*dense.T, "b--", lw=2, label="planned (A*, pruned)"); ax[0].plot(lg["x"], lg["y"], "r-", lw=1.6, label="actual (PID)")
    ax[0].plot(*START, "go", ms=9, label="start"); ax[0].plot(*GOAL, "k*", ms=14, label="goal"); ax[0].legend(loc="upper left", fontsize=8)
    ax[1].plot(lg["t"], lg["cte"], label="cross-track error (m)"); ax[1].plot(lg["t"], np.degrees(lg["herr"]) / 100, label="heading error (deg/100)")
    ax[1].set(xlabel="time (s)", title="Tracking error over time"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[2].plot(lg["t"], lg["v"], label="speed (m/s)"); ax[2].plot(lg["t"], lg["clr"], label="true clearance (m)")
    ax[2].axhline(ROBOT_RADIUS, color="r", ls="--", lw=.8, label="robot radius"); ax[2].set(xlabel="time (s)", title="Speed and clearance")
    ax[2].legend(); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "a4_trajectory_fused.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 4, figsize=(20, 5.2))
    for a, (name, (d, l)) in zip(ax, runs.items()):
        draw_map(a, title=f"{name}: {'COLLISION' if df.loc[name,'Collision'] else 'collision-free'}, RMS CTE {df.loc[name,'RMS_CTE_m']} m")
        if d is not None:
            a.plot(*d.T, "b--"); a.plot(l["x"], l["y"], "r-")
    plt.tight_layout(); plt.savefig(OUT / "a4_perception_variants.png", dpi=120); plt.close()

    sweep = []
    for vm in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        rs = [evaluate("Fused", maps["Fused"], gains, vm, seed=s)[0] for s in range(5)]
        sweep.append(dict(vmax=vm, time_s=np.mean([r["Time_to_goal_s"] for r in rs]), rms_cte=np.mean([r["RMS_CTE_m"] for r in rs]),
                          max_cte=np.mean([r["Max_CTE_m"] for r in rs]), min_clr=np.mean([r["Min_true_clearance_m"] for r in rs]),
                          collisions=sum(r["Collision"] for r in rs)))
    sw = pd.DataFrame(sweep).round(3); print("\nSpeed sweep (mean of 5 seeds):\n", sw.to_string(index=False)); sw.to_csv(OUT / "a4_speed_tradeoff.csv", index=False)
    fig, a = plt.subplots(figsize=(7, 4.5)); a.plot(sw.vmax, sw.time_s, "bo-"); a.set(xlabel="max speed (m/s)", ylabel="time to goal (s)")
    b = a.twinx(); b.plot(sw.vmax, sw.rms_cte, "rs-"); b.set_ylabel("RMS cross-track error (m)", color="r"); a.set_title("Responsiveness vs tracking accuracy"); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "a4_speed_tradeoff.png", dpi=130); plt.close()

    ms = []
    for mg in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        rs = [evaluate("Fused", maps["Fused"], gains, 0.6, seed=s, margin=mg)[0] for s in range(8)]
        ms.append(dict(margin_m=mg, planned_len_m=np.mean([r["Planned_len_m"] for r in rs]), actual_len_m=np.mean([r["Actual_len_m"] for r in rs]),
                       min_clearance_m=np.min([r["Min_true_clearance_m"] for r in rs]), collision_runs=f"{sum(r['Collision'] for r in rs)}/8"))
    mt = pd.DataFrame(ms).round(3); print("\nPlanning-margin sweep (8 noise seeds, vmax 0.6):\n", mt.to_string(index=False)); mt.to_csv(OUT / "a4_margin_tradeoff.csv", index=False)
    grid_f = inflate(maps["Fused"], ROBOT_RADIUS + MARGIN)
    np.savetxt(OUT / "waypoints.csv", prune_path(grid_f, plan(grid_f, START, GOAL, "octile")["path"]), delimiter=",", header="x_m,y_m", comments="")