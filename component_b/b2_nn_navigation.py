"""COMPONENT B - Task 3: Neural-network navigation / decision-making.

Imitation learning: an MLP maps [16 range sectors, sin/cos(bearing to target)] -> {turn right, straight, turn left}.
Labels come from a rule-based expert. Evaluated (i) offline (accuracy, per-class F1, confusion matrix,
learning curves) and (ii) in CLOSED LOOP (success rate, collisions, average reward) vs expert and random policy.
Exports weights to outputs/nn_policy.json (pure-numpy inference; reused by the swarm and the Webots controller).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json
import numpy as np, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from common.env import *
from component_b.swarm_sim import *

LABELS = ["turn right", "straight", "turn left"]


def make_dataset(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.uniform(0.3, ARENA - 0.3, (n * 2, 2)); P = P[clearance(P[:, 0], P[:, 1]) > ROBOT_RADIUS + 0.02][:n]
    th = rng.uniform(-np.pi, np.pi, len(P))
    # mixture: half uniform bearings (recovery from any heading), half small bearings (typical cruising)
    bearing = np.where(rng.random(len(P)) < 0.5, rng.uniform(-np.pi, np.pi, len(P)), rng.normal(0, 0.35, len(P)))
    F = features(sense(P[:, 0], P[:, 1], th), bearing)
    return F.astype(np.float32), expert_action(F)


def episode(policy, seed, max_steps=500):
    """Single robot: random start, random goal >4 m away. Outcome: success / collision (terminal) / timeout.
    Reward: +10 goal, -5 collision, -0.01 per step."""
    rng = np.random.default_rng(seed)
    while True:
        s, g = rng.uniform(0.5, ARENA - 0.5, (2, 2))
        if clearance(*s) > 0.4 and clearance(*g) > 0.4 and np.hypot(*(s - g)) > 4: break
    x, y, th = s[0], s[1], rng.uniform(-np.pi, np.pi)
    for k in range(max_steps):
        if np.hypot(g[0] - x, g[1] - y) < 0.5:
            return dict(outcome="success", steps=k, reward=10 - 0.01 * k)
        f = features(sense(x, y, th), wrap(np.arctan2(g[1] - y, g[0] - x) - th))
        a = ACTIONS[int(np.ravel(policy(f))[0])] if policy else rng.choice(ACTIONS)
        th = wrap(th + a * TURN); nx, ny = x + STEP * np.cos(th), y + STEP * np.sin(th)
        if clearance(nx, ny) < ROBOT_RADIUS:
            return dict(outcome="collision", steps=k, reward=-5 - 0.01 * k)
        x, y = nx, ny
    return dict(outcome="timeout", steps=max_steps, reward=-0.01 * max_steps)


if __name__ == "__main__":
    X, y = make_dataset()
    np.savez_compressed(OUT / "nav_dataset.npz", X=X, y=y)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=0, stratify=ytmp)
    print("class balance:", dict(zip(LABELS, np.bincount(y) / len(y))), " train/val/test:", len(Xtr), len(Xva), len(Xte))

    clf = MLPClassifier((32, 32), activation="relu", solver="adam", learning_rate_init=3e-3, batch_size=256, random_state=0)
    hist = []
    for ep in range(60):                                       # epoch-wise training to record learning curves
        clf.partial_fit(Xtr, ytr, classes=[0, 1, 2])
        hist.append((clf.loss_, clf.score(Xtr, ytr), clf.score(Xva, yva)))
    hist = np.array(hist)
    pred = clf.predict(Xte); acc = (pred == yte).mean()
    print(f"\nTEST accuracy: {acc:.4f}\n", classification_report(yte, pred, target_names=LABELS, digits=3))
    cm = confusion_matrix(yte, pred)

    json.dump(dict(W=[w.round(5).tolist() for w in clf.coefs_], b=[b.round(5).tolist() for b in clf.intercepts_],
                   input="16 range sectors (0=front, CCW, /2m) + sin(bearing) + cos(bearing)", actions=LABELS), open(OUT / "nn_policy.json", "w"))
    nn = NumpyPolicy(OUT / "nn_policy.json")
    assert (nn(Xte[:500]) == pred[:500]).all(), "numpy forward pass must match sklearn"

    rows = []
    for name, pol in [("Expert (teacher)", expert_action), ("Neural network", nn), ("Random", None)]:
        eps = pd.DataFrame([episode(pol, s) for s in range(1000, 1200)])
        rows.append(dict(Policy=name, Success_rate=(eps.outcome == "success").mean(), Collision_rate=(eps.outcome == "collision").mean(),
                         Timeout_rate=(eps.outcome == "timeout").mean(), Avg_steps_when_success=eps[eps.outcome == "success"].steps.mean(),
                         Avg_reward=eps.reward.mean(), Reward_std=eps.reward.std()))
    cl = pd.DataFrame(rows).set_index("Policy").round(3); print("\nClosed-loop (200 episodes, seeds 1000-1199):\n", cl.to_string())
    cl.to_csv(OUT / "b2_closed_loop_results.csv")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, ax=ax[0]); ax[0].set(xlabel="predicted", ylabel="expert label", title=f"Confusion matrix (test, acc={acc:.3f})")
    sns.heatmap(cm / cm.sum(1, keepdims=True), annot=True, fmt=".2f", cmap="Greens", xticklabels=LABELS, yticklabels=LABELS, ax=ax[1]); ax[1].set(xlabel="predicted", ylabel="expert label", title="Row-normalised (per-class recall)")
    ax[2].plot(hist[:, 1], label="train acc"); ax[2].plot(hist[:, 2], label="validation acc"); ax[2].set(xlabel="epoch", ylabel="accuracy", title="Learning curves"); ax[2].grid(alpha=.3)
    b = ax[2].twinx(); b.plot(hist[:, 0], "r--", label="training loss"); b.set_ylabel("loss"); ax[2].legend(loc="center right")
    plt.tight_layout(); plt.savefig(OUT / "b2_nn_evaluation.png", dpi=130); plt.close()
    pd.DataFrame(classification_report(yte, pred, target_names=LABELS, output_dict=True)).T.round(3).to_csv(OUT / "b2_classification_report.csv")