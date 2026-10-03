"""Run every Python/Matplotlib part of the project end to end, in dependency order.

Does NOT run Webots (that's done separately - see README.txt). Takes about 2-3 minutes.
Usage:  python run_all.py
"""
import runpy
import sys
import time
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SCRIPTS = [
    "component_a/a1_perception.py",
    "component_a/a2_planning.py",
    "component_a/a3_control.py",
    "component_a/a4_integration.py",
    "component_b/b2_nn_navigation.py",
    "component_b/b1_consensus_pso.py",
    "component_b/b3_central_vs_distributed.py",
]

if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    t0 = time.time()
    for rel in SCRIPTS:
        print(f"\n{'=' * 70}\nRunning {rel}\n{'=' * 70}")
        t1 = time.time()
        runpy.run_path(str(ROOT / rel), run_name="__main__")
        print(f"--- {rel} finished in {time.time() - t1:.1f} s ---")
    print(f"\nAll scripts finished in {time.time() - t0:.1f} s. Outputs are in {ROOT / 'outputs'}")