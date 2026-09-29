"""COMPONENT A - Task 1: Sensor simulation, perception and sensor fusion.

Pipeline
  1. LiDAR simulation (360 rays, Gaussian range noise + dropouts) from a survey of poses
     -> log-odds occupancy grid (inverse sensor model).
  2. Overhead-camera simulation (noisy render with sensor noise, blur, mis-registration
     and a glare patch) -> OpenCV HSV segmentation + morphology + contours
     -> obstacle mask and landmark detections.
  3. Probabilistic (log-odds) fusion of LiDAR + camera into one occupancy grid.
     The camera's log-likelihood ratios are *calibrated* on a separate render, not guessed.
  4. Quantitative evaluation (precision / recall / F1 / IoU) vs ground truth.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, cv2, pandas as pd
import matplotlib.pyplot as plt
from common.env import *

PX_PER_M = 50                       # camera image: 500 x 500 px for the 10 m arena
L_HIT, L_FREE, L_CLIP = 1.5, -0.35, 6.0
PENETRATION = 0.05                 # m: put the hit slightly *inside* the surface (>2 sigma of range noise)

# ----------------------------------------------------------------- LiDAR ---
def simulate_lidar(pose, rng, n_rays=360, max_range=8.0, sigma=0.02, p_drop=0.01):
    """One 360-degree scan. Returns (angles_world, ranges, valid_mask)."""
    x, y = pose
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    r = raycast(x, y, ang, max_range) + rng.normal(0, sigma, n_rays)
    valid = rng.random(n_rays) > p_drop
    return ang, np.clip(r, 0.05, max_range), valid


def survey_poses():
    pts = [(1, 1), (1, 6), (1, 9), (4, 1), (4, 6), (4, 9), (6, 4), (6, 7), (6, 9.3),
           (9.3, 4), (9.3, 7), (8, 3.5), (5, 1), (9.3, 9.3), (3, 6), (9.3, 1)]
    assert all(clearance(*p) > ROBOT_RADIUS for p in pts), "survey pose inside obstacle"
    return pts


def lidar_to_logodds(scans, max_range=8.0):
    L = np.zeros((N, N))
    for (x, y), (ang, r, valid) in scans:
        for a, rr, v in zip(ang[valid], r[valid], np.ones(valid.sum())):
            d = np.arange(0.0, rr - RES, RES / 2)                # free space along the ray
            fx, fy = x + d * np.cos(a), y + d * np.sin(a)
            rows = np.clip((fy / RES).astype(int), 0, N - 1)
            cols = np.clip((fx / RES).astype(int), 0, N - 1)
            cells = np.unique(rows * N + cols)                   # count each cell once per ray
            L[cells // N, cells % N] += L_FREE
            ex, ey = x + (rr + PENETRATION) * np.cos(a), y + (rr + PENETRATION) * np.sin(a)
            interior = 0.02 < ex < ARENA - 0.02 and 0.02 < ey < ARENA - 0.02
            if rr < max_range - 0.05 and interior:               # real obstacle hit (not wall/max range)
                L[int(ey / RES), int(ex / RES)] += L_HIT
    return np.clip(L, -L_CLIP, L_CLIP)

# ---------------------------------------------------------------- camera ---
def render_camera(rng, glare=True):
    """Synthetic top-down RGB camera image (BGR uint8, row 0 = y = ARENA)."""
    S = int(ARENA * PX_PER_M)
    img = np.full((S, S, 3), (190, 195, 195), np.uint8)
    P = lambda x, y: (int(round(x * PX_PER_M)), int(round((ARENA - y) * PX_PER_M)))
    for x0, y0, x1, y1 in OBSTACLES:
        # soft shadow (must NOT be detected -> justifies using hue rather than brightness)
        cv2.rectangle(img, P(x0 + .12, y1 - .12), P(x1 + .12, y0 - .12), (120, 125, 125), -1)
        cv2.rectangle(img, P(x0, y1), P(x1, y0), (40, 90, 190), -1)          # orange-brown box
    for lx, ly in LANDMARKS:
        cv2.circle(img, P(lx, ly), int(LANDMARK_R * PX_PER_M), (60, 190, 40), -1)  # green disc
    M = np.float32([[1, 0, rng.uniform(-3, 3)], [0, 1, rng.uniform(-3, 3)]])        # mis-registration
    img = cv2.warpAffine(img, M, (S, S), borderValue=(190, 195, 195))
    img = cv2.GaussianBlur(img, (5, 5), 1.2)
    if glare:                                                                        # specular glare patch
        cx, cy = P(*rng.uniform(1.5, 8.5, 2))
        ov = img.copy()
        cv2.ellipse(ov, (cx, cy), (int(rng.uniform(25, 45)), int(rng.uniform(20, 35))), 0, 0, 360, (245, 245, 245), -1)
        img = cv2.addWeighted(ov, 0.8, img, 0.2, 0)
    img = img.astype(np.float32) + rng.normal(0, 10, img.shape)
    salt = rng.random(img.shape[:2]) < 0.002
    img[salt] = 255
    return np.clip(img, 0, 255).astype(np.uint8)


def detect_camera(img):
    """OpenCV detection. Returns obstacle grid mask (world orientation), landmark centroids, debug overlay."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)                     # suppress salt noise before thresholding
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    obst = cv2.inRange(hsv, (5, 110, 90), (25, 255, 255))       # orange hue, high saturation
    lmk = cv2.inRange(hsv, (40, 90, 60), (85, 255, 255))        # green hue
    k3, k7 = np.ones((3, 3), np.uint8), np.ones((7, 7), np.uint8)
    obst = cv2.morphologyEx(cv2.morphologyEx(obst, cv2.MORPH_OPEN, k3), cv2.MORPH_CLOSE, k7)
    lmk = cv2.morphologyEx(lmk, cv2.MORPH_OPEN, k3)
    overlay = img.copy()
    cnts, _ = cv2.findContours(obst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(obst)
    for c in cnts:
        if cv2.contourArea(c) > 300:                             # reject blobs smaller than ~0.12 m^2
            cv2.drawContours(keep, [c], -1, 255, -1)
            cv2.drawContours(overlay, [c], -1, (255, 0, 255), 2)
    landmarks = []
    cnts, _ = cv2.findContours(lmk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) > 200:
            m = cv2.moments(c)
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            landmarks.append((cx / PX_PER_M, ARENA - cy / PX_PER_M))
            cv2.circle(overlay, (int(cx), int(cy)), 5, (255, 255, 0), -1)
    grid = cv2.resize(keep[::-1], (N, N), interpolation=cv2.INTER_AREA) > 127   # flip -> row = +y
    return grid, landmarks, overlay

# ------------------------------------------------------- fusion / metrics ---
def calibrate_camera(rng, n_images=4):
    """Estimate p(detect|occupied) and p(detect|free) on calibration renders -> log-likelihood ratios."""
    gt = ground_truth_grid(); tp = fn = fp = tn = 0
    for _ in range(n_images):
        det, _, _ = detect_camera(render_camera(rng))
        tp += (det & gt).sum(); fn += (~det & gt).sum(); fp += (det & ~gt).sum(); tn += (~det & ~gt).sum()
    tpr, fpr = tp / (tp + fn), max(fp / (fp + tn), 1e-4)
    return dict(l_det=float(np.log(tpr / fpr)), l_miss=float(np.log((1 - tpr) / (1 - fpr))), tpr=tpr, fpr=fpr)


def fuse(L_lidar, cam_mask, cal):
    L_cam = np.where(cam_mask, cal["l_det"], cal["l_miss"])
    return L_lidar + L_cam                                       # independent-evidence Bayesian update


def metrics(pred, gt):
    tp, fp, fn = (pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=p, recall=r, F1=2 * p * r / max(p + r, 1e-9), IoU=tp / max(tp + fp + fn, 1))


def run_perception(seed=7, save=True):
    rng = np.random.default_rng(seed)
    gt = ground_truth_grid()
    scans = [(p, simulate_lidar(p, rng)) for p in survey_poses()]
    L_lidar = lidar_to_logodds(scans)
    cal = calibrate_camera(np.random.default_rng(seed + 100))
    img = render_camera(rng)
    cam_mask, lmk, overlay = detect_camera(img)
    L_fused = fuse(L_lidar, cam_mask, cal)
    maps = {"LiDAR only": L_lidar > 0, "Camera only": cam_mask, "Fused (LiDAR+camera)": L_fused > 0}
    res = pd.DataFrame({k: metrics(v, gt) for k, v in maps.items()}).T.round(3)
    lm_err = [min(np.hypot(a - x, b - y) for a, b in lmk) if lmk else np.nan for x, y in LANDMARKS]
    out = dict(gt=gt, scans=scans, L_lidar=L_lidar, cam_mask=cam_mask, img=img, overlay=overlay,
               landmarks=lmk, landmark_err=lm_err, cal=cal, maps=maps, table=res, L_fused=L_fused)
    if save:
        np.save(OUT / "occ_lidar.npy", maps["LiDAR only"]); np.save(OUT / "occ_camera.npy", cam_mask)
        np.save(OUT / "occ_fused.npy", maps["Fused (LiDAR+camera)"]); np.save(OUT / "occ_truth.npy", gt)
        np.savez_compressed(OUT / "lidar_scans.npz", poses=np.array([p for p, _ in scans]),
                            ranges=np.array([s[1] for _, s in scans]), valid=np.array([s[2] for _, s in scans]))
        cv2.imwrite(str(OUT / "camera_image.png"), img); res.to_csv(OUT / "a1_perception_metrics.csv")
    return out


def plot_perception(o):
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
    draw_map(ax[0, 0], title="Ground truth + LiDAR survey poses")
    ax[0, 0].plot(*np.array([p for p, _ in o["scans"]]).T, "r^", ms=6, label="scan poses")
    for lx, ly in LANDMARKS: ax[0, 0].add_patch(plt.Circle((lx, ly), LANDMARK_R, color="g", alpha=.6))
    ax[0, 0].legend(loc="upper right", fontsize=8)
    (x, y), (ang, r, v) = o["scans"][6]
    axp = ax[0, 1]; axp.remove(); axp = fig.add_subplot(2, 3, 2, projection="polar")
    axp.scatter(ang[v], r[v], s=3); axp.set_title(f"One raw LiDAR scan from ({x:.0f},{y:.0f}) m", pad=15)
    ax[0, 2].imshow(cv2.cvtColor(o["overlay"], cv2.COLOR_BGR2RGB)); ax[0, 2].axis("off")
    ax[0, 2].set_title("Camera: HSV contours (magenta) + landmarks (cyan)")
    for a, k, t in [(ax[1, 0], "LiDAR only", "LiDAR occupancy"), (ax[1, 1], "Camera only", "Camera occupancy"),
                    (ax[1, 2], "Fused (LiDAR+camera)", "Fused occupancy")]:
        a.imshow(o["maps"][k], origin="lower", extent=(0, ARENA, 0, ARENA), cmap="Greys")
        m = o["table"].loc[k]
        a.set_xlim(0, ARENA); a.set_ylim(0, ARENA); a.set_aspect("equal"); a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
        a.set_title(f"{t}\nF1={m.F1:.2f}  IoU={m.IoU:.2f}  (red dashes = truth)")
        for x0, y0, x1, y1 in OBSTACLES:
            a.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="r", lw=1, ls="--"))
    plt.tight_layout(); plt.savefig(OUT / "a1_perception.png", dpi=130); plt.close()


def multi_seed_eval(n=10):
    """Repeat the whole pipeline with different noise/glare seeds -> mean +- std."""
    df = pd.concat([run_perception(seed=s, save=False)["table"] for s in range(100, 100 + n)])
    g = df.groupby(level=0).agg(["mean", "std"]).round(3)
    g.to_csv(OUT / "a1_perception_multiseed.csv")
    return g


if __name__ == "__main__":
    o = run_perception()
    plot_perception(o)
    print("Camera calibration:", {k: round(v, 3) for k, v in o["cal"].items()})
    print("\nOccupancy-map quality vs ground truth:\n", o["table"])
    print("\nLandmark localisation error (m):", np.round(o["landmark_err"], 3))
    print(f"\nRobustness over 10 random seeds (mean/std):\n", multi_seed_eval(10))