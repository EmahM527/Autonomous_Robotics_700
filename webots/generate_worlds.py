"""Generate Webots world files from common/env.py so Python simulations and Webots share ONE map.

Webots scale: the 10 x 10 m arena is scaled by SCALE = 0.2 (-> 2 x 2 m) because the e-puck is only 7 cm wide.
Map (x, y) metres  ->  Webots (X, Y) = ((x - 5) * SCALE, (y - 5) * SCALE)   (ENU, z up).
Target: Webots R2025a (matches the version installed for this project - see README).
Run:  python webots/generate_worlds.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.env import *

SCALE = 0.2
BASE = "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects"

_HEADER = """#VRML_SIM R2025a utf8

EXTERNPROTO "@BASE@/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "@BASE@/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "@BASE@/objects/floors/protos/RectangleArena.proto"
EXTERNPROTO "@BASE@/robots/gctronic/e-puck/protos/E-puck.proto"

WorldInfo {
  info ["ASR700 Assignment - @TITLE@"]
  title "@TITLE@"
  basicTimeStep 16
  coordinateSystem "ENU"
}
Viewpoint {
  orientation -0.28 0.28 0.92 1.65
  position 0 -2.6 2.6
}
TexturedBackground {
}
TexturedBackgroundLight {
}
RectangleArena {
  floorSize @SIZE@ @SIZE@
  wallHeight 0.1
}
"""


def header(title):
    return _HEADER.replace("@BASE@", BASE).replace("@TITLE@", title).replace("@SIZE@", f"{ARENA * SCALE:g}")


def W(x, y):
    return (x - ARENA / 2) * SCALE, (y - ARENA / 2) * SCALE


def obstacles():
    out = []
    for i, (x0, y0, x1, y1) in enumerate(OBSTACLES):
        cx, cy = W((x0 + x1) / 2, (y0 + y1) / 2); sx, sy = (x1 - x0) * SCALE, (y1 - y0) * SCALE
        out.append(f'''Solid {{
  translation {cx:.4f} {cy:.4f} 0.05
  children [ Shape {{ appearance PBRAppearance {{ baseColor 0.85 0.4 0.1 roughness 1 metalness 0 }} geometry Box {{ size {sx:.4f} {sy:.4f} 0.1 }} }} ]
  name "obstacle_{i}"
  boundingObject Box {{ size {sx:.4f} {sy:.4f} 0.1 }}
}}''')
    for i, (lx, ly) in enumerate(LANDMARKS):
        X, Y = W(lx, ly)
        out.append(f'''Solid {{
  translation {X:.4f} {Y:.4f} 0.001
  rotation 1 0 0 1.5708
  children [ Shape {{ appearance PBRAppearance {{ baseColor 0.25 0.75 0.15 roughness 1 metalness 0 }} geometry Cylinder {{ height 0.002 radius {LANDMARK_R*SCALE:.4f} }} }} ]
  name "landmark_{i}"
}}''')
    return "\n".join(out)


SENSORS = f'''turretSlot [
    GPS {{ }}
    InertialUnit {{ }}
    Lidar {{
      translation 0 0 0.02
      horizontalResolution 360
      fieldOfView 6.2832
      numberOfLayers 1
      minRange 0.01
      maxRange {2.0 * SCALE:g}
      noise 0.005
    }}
  ]'''


def epuck(name, ctrl, x, y, heading=0.0):
    X, Y = W(x, y)
    return f'''E-puck {{
  translation {X:.4f} {Y:.4f} 0
  rotation 0 0 1 {heading:.4f}
  name "{name}"
  controller "{ctrl}"
  {SENSORS}
}}'''


if __name__ == "__main__":
    d = pathlib.Path(__file__).parent / "worlds"; d.mkdir(exist_ok=True)
    a = header("Component A - autonomous navigation") + obstacles() + "\n" + epuck("epuck_nav", "a_navigator", *START, 0.6) + "\n"
    (d / "component_a.wbt").write_text(a)
    starts = [(0.8, 0.8), (1.4, 0.8), (0.8, 1.4), (1.4, 1.4), (1.1, 1.1)]
    b = header("Component B - cooperative coverage swarm") + obstacles() + "\n"
    b += "\n".join(epuck(f"agent_{i}", "swarm_agent", x, y, 0.9 * i) for i, (x, y) in enumerate(starts)) + "\n"
    (d / "component_b.wbt").write_text(b)
    print("wrote", d / "component_a.wbt", "and", d / "component_b.wbt")