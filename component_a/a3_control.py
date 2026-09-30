"""COMPONENT A - Task 3: Kinematic/dynamic model and PID heading control.

Kinematic model (unicycle / differential drive, state [x y th]):
    x' = v cos(th),   y' = v sin(th),   th' = w
    wheel speeds: v_R = v + w*L/2, v_L = v - w*L/2  (L = axle length)
Dynamic (actuator) model used for tuning - first-order lag + transport delay + saturation:
    v_act' = (v_cmd - v_act)/tau_v ,  w_act' = (w_cmd - w_act)/tau_w ,  cmd delayed by T_d
    => heading plant  G(s) = e^{-T_d s} / ( s (tau_w s + 1) )   (integrator + lag + delay)
Controller: PID on heading error with derivative-on-measurement (no set-point kick),
anti-windup by integral clamping (with integral separation) and output saturation.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, itertools
from collections import deque
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common.env import *

DT = 0.02
TAU_V, TAU_W, DELAY = 0.20, 0.15, 0.10       # s
V_MAX, W_MAX = 1.2, 3.0                       # m/s, rad/s
W_BIAS = 0.25                                 # rad/s constant heading drift (wheel-radius mismatch / slip)


class Unicycle:
    def __init__(self, x=0., y=0., th=0., dt=DT, w_bias=0.0):
        self.x, self.y, self.th, self.v, self.w, self.dt = x, y, th, 0., 0., dt
        self.w_bias = w_bias
        self.q = deque([(0., 0.)] * max(1, int(round(DELAY / dt))))     # transport delay buffer

    def step(self, v_cmd, w_cmd):
        self.q.append((np.clip(v_cmd, -V_MAX, V_MAX), np.clip(w_cmd, -W_MAX, W_MAX)))
        vc, wc = self.q.popleft()
        self.v += self.dt * (vc - self.v) / TAU_V
        self.w += self.dt * (wc - self.w) / TAU_W
        self.x += self.dt * self.v * np.cos(self.th)
        self.y += self.dt * self.v * np.sin(self.th)
        self.th += self.dt * (self.w + self.w_bias)


class PID:
    def __init__(self, kp, ki, kd, out_lim=W_MAX, i_lim=1.0, i_zone=0.3):
        self.kp, self.ki, self.kd, self.out_lim, self.i_lim, self.i_zone = kp, ki, kd, out_lim, i_lim, i_zone
        self.I, self.prev_meas = 0.0, None

    def step(self, err, meas, dt):
        """err = reference - measurement (angle-wrapped by caller); meas = measured output."""
        if abs(err) < self.i_zone:                       # integral separation: no windup during large slews
            self.I = np.clip(self.I + err * dt, -self.i_lim, self.i_lim)
        d = 0.0 if self.prev_meas is None else (meas - self.prev_meas) / dt
        self.prev_meas = meas
        return float(np.clip(self.kp * err + self.ki * self.I - self.kd * d, -self.out_lim, self.out_lim))


def step_response(gains, ref=1.0, T=8.0, dt=DT, bias=W_BIAS):
    """Heading step (rotate in place) with constant drift disturbance. Returns t, theta, metrics."""
    m, pid = Unicycle(dt=dt, w_bias=bias), PID(*gains)
    n = int(T / dt); t = np.arange(n) * dt; th = np.zeros(n)
    for i in range(n):
        m.step(0.0, pid.step(ref - m.th, m.th, dt))
        th[i] = m.th
    return t, th, step_metrics(t, th, ref)


def step_metrics(t, th, ref):
    if not np.all(np.isfinite(th)) or np.max(np.abs(th)) > 8 * abs(ref):
        return dict(rise_s=np.nan, overshoot_pct=np.nan, settle_s=np.inf, sse=np.nan, stable=False, itae=np.inf)
    tail = th[-int(2.0 / (t[1] - t[0])):]; e = tail - tail.mean()          # sustained oscillation in last 2 s?
    converged = not (np.ptp(tail) > 0.05 and np.sum(e[:-1] * e[1:] < 0) >= 2)
    i10, i90 = np.argmax(th >= 0.1 * ref), np.argmax(th >= 0.9 * ref)
    rise = t[i90] - t[i10] if th.max() >= 0.9 * ref else np.nan
    os_ = max(0.0, (th.max() - ref) / ref * 100)
    out = np.where(np.abs(th - ref) > 0.02 * ref)[0]
    settle = 0.0 if len(out) == 0 else (t[out[-1]] if out[-1] < len(t) - 1 else np.inf)
    sse = ref - th[-int(0.5 / (t[1] - t[0])):].mean()
    itae = np.trapezoid(t * np.abs(ref - th), t)
    return dict(rise_s=rise, overshoot_pct=os_, settle_s=settle, sse=sse, stable=converged, itae=itae)


def tune(ref=1.0):
    """Grid search minimising ITAE subject to overshoot <= 8 %, |sse| < 0.005 rad, and settling within horizon."""
    best, table = None, []
    for kp, ki, kd in itertools.product(np.arange(1, 10.5, 0.5), [0, 1.0, 2.0, 3.0, 4.0, 6.0], np.arange(0, 1.61, 0.1)):
        _, _, m = step_response((kp, ki, kd), ref)
        table.append((kp, ki, kd, m["itae"], m["overshoot_pct"], m["settle_s"]))
        if m["stable"] and np.isfinite(m["settle_s"]) and m["overshoot_pct"] <= 8 and abs(m["sse"]) < 0.005 and (best is None or m["itae"] < best[1]):
            best = ((float(kp), float(ki), float(kd)), m["itae"])
    return best[0], pd.DataFrame(table, columns=["kp", "ki", "kd", "itae", "overshoot", "settle"])


if __name__ == "__main__":
    tuned, grid = tune()
    print("ITAE-optimal gains (Kp, Ki, Kd):", tuned)
    json.dump(dict(kp=tuned[0], ki=tuned[1], kd=tuned[2]), open(OUT / "pid_gains.json", "w"))
    sets = {"P only, low gain (Kp=1.5)": (1.5, 0, 0),
            "P only, high gain (Kp=8)": (8, 0, 0),
            "PD (Kp=%.1f, Kd=%.1f)" % (tuned[0], tuned[2]): (tuned[0], 0, tuned[2]),
            "PID tuned (%.1f, %.2f, %.1f)" % tuned: tuned,
            "PID, excessive Ki (Ki=20)": (tuned[0], 20.0, tuned[2]),
            "P only, Kp=30 (unstable)": (30, 0, 0)}
    fig, ax = plt.subplots(1, 2, figsize=(15, 5)); rows = []
    for name, gset in sets.items():
        t, th, m = step_response(gset)
        ax[0].plot(t, np.clip(th, -3, 4), label=name); rows.append(dict(Controller=name, **{k: v for k, v in m.items() if k != "itae"}))
    ax[0].axhline(1.0, color="k", ls="--", lw=.8); ax[0].axhspan(.98, 1.02, color="g", alpha=.1, label="2 % band")
    ax[0].set(xlabel="time (s)", ylabel="heading (rad)", title="Heading step response (1 rad) - effect of gains", ylim=(-1, 2.2))
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    kps = np.arange(0.5, 40, 0.5); ov, amp, unstable = [], [], []
    for kp in kps:
        t, th, m = step_response((kp, 0, 0)); ov.append(m["overshoot_pct"])
        amp.append(np.ptp(th[-100:]) / 2); unstable.append(not m["stable"])   # tail = last 2 s
    ax2 = ax[1]; ax2.plot(kps, ov, "r-", label="overshoot"); ax2.set(xlabel="Kp (P-only)", ylabel="peak overshoot (%)")
    ax3 = ax2.twinx(); ax3.plot(kps, amp, "b-"); ax3.set_ylabel("steady oscillation amplitude in last 2 s (rad)", color="b")
    ax2.set_title("P-only gain sweep (dashed = onset of sustained oscillation)"); ax2.grid(alpha=.3)
    bad = np.array([k for k in kps if not step_response((k, 0, 0))[2]["stable"]])
    if len(bad): ax2.axvline(bad[0], color="k", ls="--")
    plt.tight_layout(); plt.savefig(OUT / "a3_pid_tuning.png", dpi=130); plt.close()
    df = pd.DataFrame(rows).set_index("Controller"); print(df.round(3).to_string()); df.round(4).to_csv(OUT / "a3_pid_metrics.csv")
    print("\nP-only stability boundary near Kp =", bad[0] if len(bad) else "not reached")