ASR700 ASSIGNMENT - INTELLIGENT AUTONOMOUS ROBOTIC SYSTEMS PROJECT
====================================================================
Component A: Perception, Motion Planning and Control for a Mobile Robot
Component B: Multi-Agent Swarm with Learning-Based Decision-Making

--------------------------------------------------------------------
1. WHAT THIS PACKAGE CONTAINS
--------------------------------------------------------------------
common/env.py                Shared 10x10 m arena definition and geometry
                              helpers (obstacles, raycasting, occupancy grid).
                              Used by BOTH the Python pipelines and the
                              generated Webots worlds, so every part of the
                              project uses exactly one map.

component_a/
  a1_perception.py           LiDAR + camera simulation, OpenCV detection,
                              calibrated log-odds sensor fusion, evaluation.
  a2_planning.py              Dijkstra and A* (Euclidean/Octile/Manhattan)
                              on the fused occupancy grid, comparison.
  a3_control.py               Unicycle kinematic model + first-order actuator
                              lag/delay, PID heading controller with
                              anti-windup, ITAE gain tuning, step responses.
  a4_integration.py           Closed-loop path following (pure pursuit +
                              PID), planned-vs-actual trajectory, speed and
                              planning-margin trade-off sweeps.

component_b/
  swarm_sim.py                Shared swarm simulation library: 16-ray
                              sensing, rule-based expert policy, numpy MLP
                              inference, centralised/distributed coverage
                              coordination.
  b1_consensus_pso.py         Average consensus (multiple topologies + link
                              loss), leader election + re-election, PSO
                              (gbest vs lbest) task allocation vs exact
                              optimum/greedy/random.
  b2_nn_navigation.py          MLP imitation-learning navigation policy:
                              dataset generation, training, confusion
                              matrix, learning curves, closed-loop
                              evaluation vs expert and random policies.
  b3_central_vs_distributed.py Cooperative area-coverage swarm; centralised
                              vs distributed coordination under nominal
                              conditions, robot failure, coordinator
                              failure, reduced comms, and team-size scaling.

webots/
  generate_worlds.py          Generates component_a.wbt / component_b.wbt
                              FROM common/env.py (single source of truth).
  worlds/                     Generated .wbt files (already built).
  controllers/a_navigator/    Webots controller: waypoint following with
                              the SAME PID gains and pure-pursuit logic as
                              component_a/a4_integration.py.
  controllers/swarm_agent/    Webots controller: distributed coverage using
                              the SAME neural-network policy (nn_policy.json)
                              as component_b.
  mock/                       An offline EMULATOR of the Webots `controller`
                              API (controller.py) and a runner (run_mock.py),
                              used to smoke-test both controllers WITHOUT
                              Webots installed. This is not a substitute for
                              running the real simulation in Webots - see
                              section 4.

run_all.py                    Runs every Python/Matplotlib script in
                              dependency order. Does not touch Webots.
outputs/                      All generated figures, CSVs and .npy/.npz/.json
                              data (created by run_all.py).
report/                       Put the 3-9 page PDF report here.
requirements.txt               Python dependencies.

--------------------------------------------------------------------
2. SOFTWARE VERSIONS USED TO BUILD AND TEST THIS PACKAGE
--------------------------------------------------------------------
Python            3.12.3
numpy             2.4.4
matplotlib        3.10.8
opencv-python     4.13.0
scikit-learn      1.8.0
seaborn           0.13.2
pandas            (see requirements.txt, >=2.1)
scipy             (see requirements.txt, >=1.11)
Webots            Worlds/controllers target R2023b syntax (EXTERNPROTO
                  URLs point at the R2023b branch). NOT tested against a
                  real Webots install in this environment - see section 4.

--------------------------------------------------------------------
3. HOW TO RUN THE PYTHON / MATPLOTLIB PART (Components A and B)
--------------------------------------------------------------------
  1. Create a virtual environment (recommended) and install dependencies:
       python -m venv venv
       venv\Scripts\activate          (Windows)   OR   source venv/bin/activate (Linux/Mac)
       pip install -r requirements.txt

  2. From the project root, run everything:
       python run_all.py

     This runs, in order:
       a1_perception -> a2_planning -> a3_control -> a4_integration
       -> b2_nn_navigation -> b1_consensus_pso -> b3_central_vs_distributed
     (a4 needs a1's occupancy maps; a2/a4 need each other's helpers; b1's
     PSO task allocator plans on a2's inflated grid; b3 needs b2's trained
     network weights (outputs/nn_policy.json) - hence this order.)

     Expected runtime: approx. 80 seconds on a normal laptop (measured:
     79.8 s single-threaded, no GPU required).

  3. All figures (.png), tables (.csv) and intermediate data (.npy/.npz/
     .json) are written to outputs/. Each script can also be run on its
     own, e.g.:
       python component_a/a1_perception.py
       python component_b/b1_consensus_pso.py
     as long as outputs/ already has any files it depends on (run_all.py
     guarantees this; running scripts out of order may not).

  4. Headless plotting: matplotlib is set to the "Agg" backend by default
     (common/env.py) so this runs on a machine with no display (e.g. CI,
     SSH). Set the environment variable ASR_SHOW=1 before running a script
     if you want interactive plot windows instead.

--------------------------------------------------------------------
4. HOW TO RUN THE WEBOTS PART
--------------------------------------------------------------------
IMPORTANT / HONEST LIMITATION: this package was built in an environment
without Webots installed. The world files and controllers were written
against the documented Webots R2023b API and syntax, and were smoke-tested
against a hand-written Python EMULATOR of that API (webots/mock/), NOT
against real Webots. Before you submit, you must:
  (a) open both worlds in your own Webots installation,
  (b) confirm they load without errors (missing PROTO, device-name
      mismatches, etc. are the most likely issues), and
  (c) fix anything that doesn't match your Webots version.
Do not present the emulator run as evidence the real simulation works.

  Step 1 - generate/regenerate the world files (already done, re-run only
  if you change common/env.py):
       python webots/generate_worlds.py

  Step 2 - open in Webots:
       Webots > File > Open World... > webots/worlds/component_a.wbt
       Webots > File > Open World... > webots/worlds/component_b.wbt
       (component_b.wbt places 5 e-pucks; open the .wbt directly, do not
       try to merge it into component_a.wbt.)

  Step 3 - run. Component A: the single e-puck should navigate from
  (1,1) to (9,9) map-metres (scaled x0.2 in the Webots world, so roughly
  corner-to-corner of the 2x2 m arena), following outputs/waypoints.csv
  and logging its true GPS/IMU trajectory to outputs/webots_trajectory.csv
  for comparison against the planned path.
  Component B: the 5 e-pucks should spread out and cover the arena,
  logging outputs/webots_swarm_<id>.csv each.

  Step 4 (optional) - offline smoke test without Webots:
       python webots/mock/run_mock.py a_navigator
       python webots/mock/run_mock.py swarm_agent
     These run the SAME controller scripts against webots/mock/controller.py
     (an emulator, not real physics) and print a one-line summary. Useful
     for catching Python errors before opening Webots, not for validating
     robot behaviour.

  Known unverified assumptions in the controllers (check these first if
  something looks wrong in Webots):
    - The e-puck device names used ("left wheel motor", "right wheel
      motor", "gps", "inertial unit", "lidar", "emitter", "receiver")
      match the standard Webots E-puck PROTO's turretSlot device names
      for the Webots version you install; check the PROTO in the
      Webots project directory (or webots/webots_projects/robots/gctronic/e-puck/protos/E-puck.proto)
      if a device lookup fails.
    - Lidar ray-index convention: webots/controllers/swarm_agent/swarm_agent.py
      assumes index 0 (and index n/2) is the front-facing ray and the
      index increases clockwise (IDX_SIGN = -1 near the top of the file).
      If the swarm robots consistently turn the wrong way, flip
      IDX_SIGN to +1.
    - Emitter/Receiver: both components in a single world share the
      default radio channel, so no explicit setChannel() call was needed
      for 5 agents at short range; if you add more robots or merge
      worlds, check for channel collisions.

--------------------------------------------------------------------
5. DEPENDENCIES (see also requirements.txt)
--------------------------------------------------------------------
numpy, matplotlib, scipy, scikit-learn, opencv-python, seaborn, pandas.
No GPU, no internet access, and no ROS/Gazebo are required to run
run_all.py. Webots (any recent version, tested syntax against R2023b) is
required only for the webots/ part.

--------------------------------------------------------------------
6. HARDWARE REQUIREMENTS
--------------------------------------------------------------------
Any laptop/desktop capable of running Python 3.10+ and Webots. No GPU
needed (the neural network is a small MLP trained with scikit-learn on
CPU, <10 s). Webots itself recommends a discrete or recent integrated GPU
for smooth 3-D rendering, but the simulation content here (a 2x2 m arena,
up to 5 e-pucks) is light.

--------------------------------------------------------------------
7. KNOWN LIMITATIONS / HONEST NOTES
--------------------------------------------------------------------
- Webots was not available in the environment used to build this
  package; see section 4.
- The A4 "speed vs tracking-error" trade-off sweep uses a simulator with
  no wheel slip or terrain effects, so the improvement in tracking error
  at higher speeds is a property of this simulation, not a general
  robotics result - say so if you discuss it in the report.
- Centralised-vs-distributed collision counts are right-skewed (a small
  number of runs account for most collisions); both mean and median are
  reported in outputs/b3_central_vs_distributed.csv for that reason.
- The neural-network closed-loop success rate (91%) is lower than the
  expert it was trained to imitate (95%) - a textbook example of
  covariate shift in imitation learning, worth discussing in the report
  rather than treated as a bug.

Author:
Name: [Cainos Emah Mtsweni]
Student Number: [402307830]

Module:
Autonomous Systems and Robotics 700