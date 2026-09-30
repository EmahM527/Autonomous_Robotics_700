"""COMPONENT B - Tasks 1 & 3: cooperative area-coverage swarm; centralised vs distributed decision-making.

Team of N robots (default 5) covers the arena. Every robot's low-level steering is the trained neural network.
Coordination architectures:
   centralised : one coordinator with global knowledge assigns targets (2N messages/step)
   distributed : range-limited comms (3 m), belief-map merging, target claims (no coordinator)
Scenarios (10 random seeds each): nominal | one robot fails at step 50 | coordinator fails at step 50 | reduced comm range | team-size scaling.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common.env import *
from component_b.swarm_sim import *

SEEDS, MAXSTEPS = range(10), 700


def run(n, mode, seed, policy, **kw):
    return Swarm(n, mode, policy, seed, **kw).run(MAXSTEPS, stop_at=0.99)


def summarise(swarms):
    def first(h, q): return next((i for i, c in enumerate(h) if c >= q), np.nan)
    return dict(final_cov_pct=100 * np.mean([s.cov_hist[-1] for s in swarms]),
                steps_to_90=np.nanmean([first(s.cov_hist, .90) for s in swarms]) if not all(np.isnan([first(s.cov_hist, .90) for s in swarms])) else np.nan,
                steps_to_99=np.nanmean([first(s.cov_hist, .99) for s in swarms]) if not all(np.isnan([first(s.cov_hist, .99) for s in swarms])) else np.nan,
                runs_reaching_99=f"{sum(s.cov_hist[-1] >= .99 for s in swarms)}/{len(swarms)}",
                wall_collisions_mean=np.mean([s.collisions for s in swarms]), wall_collisions_median=np.median([s.collisions for s in swarms]),
                robot_near_misses=np.mean([s.near for s in swarms]), msgs_per_step=np.mean([s.msgs / max(s.t, 1) for s in swarms]))


def curve(swarms, L=MAXSTEPS):
    a = np.array([np.r_[s.cov_hist, np.full(L - len(s.cov_hist), s.cov_hist[-1])] for s in swarms]) * 100
    return a.mean(0), a.std(0)


if __name__ == "__main__":
    nn = NumpyPolicy(OUT / "nn_policy.json")
    scen = {"Nominal": {}, "1 robot fails @50": dict(fail_agent=0, fail_at=50), "Coordinator fails @50": dict(fail_coordinator=50),
            "Distributed, short comm range (1.5 m)": dict(comm_range=1.5)}
    rows, C = [], {}
    for sname, kw in scen.items():
        for mode in ["centralised", "distributed"]:
            if sname.startswith("Distributed") and mode == "centralised": continue
            kw2 = {} if (sname.startswith("Coordinator") and mode == "distributed") else kw
            sw = [run(5, mode, s, nn, **kw2) for s in SEEDS]; C[(sname, mode)] = curve(sw)
            rows.append(dict(Scenario=sname + (" (n/a: no coordinator)" if sname.startswith("Coordinator") and mode == "distributed" else ""), Architecture=mode, **summarise(sw)))
    tab = pd.DataFrame(rows).set_index(["Scenario", "Architecture"]).round(2); print(tab.to_string()); tab.to_csv(OUT / "b3_central_vs_distributed.csv")

    pr = []
    for name, pol in [("Neural network", nn), ("Expert rules", None)]:
        sw = [run(5, "distributed", s, pol) for s in SEEDS]; pr.append(dict(Policy=name, **summarise(sw)))
    pol_tab = pd.DataFrame(pr).set_index("Policy").round(2); print("\nSteering policy inside the swarm (distributed, nominal):\n", pol_tab.to_string()); pol_tab.to_csv(OUT / "b3_policy_in_swarm.csv")

    sc = []
    for n in [1, 2, 3, 5, 8]:
        for mode in ["centralised", "distributed"]:
            sw = [run(n, mode, s, nn) for s in SEEDS]; sc.append(dict(robots=n, mode=mode, **summarise(sw)))
    sc = pd.DataFrame(sc).round(2); sc.to_csv(OUT / "b3_scaling.csv", index=False); print("\nScaling:\n", sc[["robots", "mode", "final_cov_pct", "steps_to_90", "steps_to_99", "msgs_per_step"]].to_string(index=False))

    fig, ax = plt.subplots(2, 3, figsize=(19, 10.5))
    for a, sname in zip([ax[0, 0], ax[0, 1], ax[0, 2]], ["Nominal", "1 robot fails @50", "Coordinator fails @50"]):
        for mode, c in [("centralised", "tab:red"), ("distributed", "tab:blue")]:
            m, s = C[(sname, mode)]; a.plot(m, c=c, label=mode); a.fill_between(range(len(m)), m - s, np.minimum(m + s, 100), color=c, alpha=.2)
        a.set(title=sname, xlabel="step", ylabel="area covered (%)", ylim=(0, 102)); a.axhline(99, color="k", ls=":", lw=.8); a.legend(loc="lower right"); a.grid(alpha=.3)
    for mode, c in [("centralised", "tab:red"), ("distributed", "tab:blue")]:
        d = sc[sc["mode"] == mode]; ax[1, 0].plot(d.robots, d.steps_to_90, "o-", c=c, label=mode)
    ax[1, 0].set(title="Scaling: steps to 90 % coverage", xlabel="number of robots", ylabel="steps"); ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)
    for k, (mode, ttl) in enumerate([("distributed", "Distributed"), ("centralised", "Centralised")]):
        s = run(5, mode, 0, nn); a = ax[1, 1 + k]; a.imshow(np.where(s.valid, s.truth, np.nan), origin="lower", extent=(0, ARENA, 0, ARENA), cmap="Greens", alpha=.35, vmin=0, vmax=1)
        draw_map(a, title=f"{ttl}: trajectories of 5 NN-driven robots (cov {100*s.cov_hist[-1]:.0f} % in {s.t} steps)")
        for i, tr in enumerate(s.traj): tr = np.array(tr); a.plot(*tr.T, lw=1.2, c=plt.cm.tab10(i)); a.plot(*tr[0], "o", c=plt.cm.tab10(i))
    plt.tight_layout(); plt.savefig(OUT / "b3_central_vs_distributed.png", dpi=125); plt.close()