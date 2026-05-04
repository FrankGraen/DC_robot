# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo layout

All real code lives in `daran/`. The repository root only holds `environment.yml`. Run scripts and notebooks from inside `daran/` so relative asset paths (`assets/Dr_arm_6_desk/...`, `font/`, `captured_images/`) resolve correctly.

## Environment

Python 3.10 conda env named `robot`, defined by `environment.yml` at the repo root.

```bash
conda env create -f environment.yml   # first-time setup
conda activate robot
```

Key libraries the code depends on: `roboticstoolbox-python`, `spatialmath-python`, `modern-robotics`, `opencv-python` / `opencv-contrib-python`, `pyserial`, `pybullet`, `pyrobosim`, `trimesh`. Some notebooks also import `mujoco`. The `requirements.txt` inside `daran/` is UTF-16 with broken encoding — use `environment.yml` instead.

## Hardware connection

The real arm uses a serial-to-CAN bridge. The serial port string is platform-specific and is passed to `arm_robot(...)` via the `com=` kwarg (and is hard-coded as a default in many sample scripts):

- Windows: `COM4` / `COM7`
- Jetson / Ubuntu: `/dev/ttyUSB0`
- Raspberry Pi: `/dev/ttyAMA0`
- macOS: `/dev/cu.usbserial-110`

Default baud is 115200. Treat any script that constructs `arm_robot(...)` as a **real-hardware** entry point — it opens the serial port at import time of the object. There is no mock/dry-run mode in `arm_robot.py` itself; for sim-only work use the `dh_*` notebooks (see below).

## Class hierarchy

The robot stack is a single linear inheritance chain. Understanding this is necessary to know where a given method lives:

```
DrEmpower_can         daran/DrEmpower_can.py     low-level serial/CAN protocol, per-joint set_angle/get_angle, PID, properties
   └── arm            daran/arm_six_axis.py      6-axis kinematics: hand-derived analytical FK/IK, Jacobian, static torques
        └── gripper   daran/gripper.py           parallel gripper as joint id_num=7 (width<->angle conversion)
             └── arm_robot  daran/arm_robot.py   user-facing API: set_arm_pose, set_arm_joints, trajectory exec, model<->servo angle mapping
```

`arm_robot` is what tests/notebooks instantiate. Joint-limit clamping (`MAX_list` / `MIN_list`) and the model-angle ↔ servo-angle conversion (`P1_list`, `P2_list`) live on `arm_robot`. Only joints 1–6 are arm joints; joint 7 is the gripper.

## Two parallel kinematics implementations

This is the single most important thing to know before touching kinematics:

1. **Hand-derived analytical formulas** in `arm_six_axis.py` (`forward_kinematics_pose`, `inverse_kinematics`, `forward_kinematics_jacobi`). Uses link parameters `L = [l1=150, l2=150, l3=68, d3=54.94, d4=33]` mm. This is what `arm_robot` uses at runtime.
2. **roboticstoolbox `DHRobot`** built from `RevoluteDH` in the `dh_*` notebooks (`dh_roboticstoolbox_control.ipynb`, `dh_cartesian_line_motion.ipynb`). Uses `ikine_LM` for inverse kinematics and `ctraj` for Cartesian paths. Units inside the toolbox are **meters**, but the user-facing inputs in those notebooks are in **mm** and converted at the boundary.

The DH parameter table is documented in `daran/report.md`. The two implementations are kept consistent — `daran/validate_dh_params.py` is the regression check that compares the standard-DH transform, the analytical position formula, and the URDF zero pose. Run it after editing any kinematics code:

```bash
cd daran && python validate_dh_params.py
```

## Tutorial notebook sequence

The numbered notebooks in `daran/` are an ordered course, not interchangeable demos:

1. `1、运动控制实操.ipynb` — joint motion basics
2. `2-1`, `2-2`, `2、正逆运动学*` — kinematics theory + practice
3. `3、`, `4、轨迹规划*` — trajectory planning
4. `6、相机标定`, `7、空间点在相机中的投影`, `8、视觉定位`, `9、视觉抓取` — vision pipeline ending in a pick-and-place

The newer `dh_roboticstoolbox_control.ipynb` and `dh_cartesian_line_motion.ipynb` are standalone reimplementations using the roboticstoolbox DH model and are independent of the numbered series.

## "Simulate then real" pattern

The `dh_*` notebooks (and any new notebook that drives the arm) gate real motion behind a flag:

```python
MOVE_REAL_ROBOT = False   # set to True only after the simulation/animation looks correct
```

Two more guards live alongside it:

- `check_model_limits(ROBOT_TARGET_DEG)` — verifies every joint is within `MODEL_MAX_DEG` / `MODEL_MIN_DEG` before sending.
- `INPUT_MODE = 'joint' | 'xyz'` — `'xyz'` runs `ikine_LM` first; the result must be sanity-checked before flipping `MOVE_REAL_ROBOT`.

When adding new motion code, preserve this guard pattern. Do not silently send commands to the real robot from a script that the user is editing.

## Pitfalls specific to this codebase

- Several `.py` files contain `sys.path.append('d:/project/robot_show/robot/daran')` (e.g. `color_take.py`, `cv_sorting.py`). That hard-coded Windows path is dead on this Linux host — it's a no-op rather than an error. Don't propagate it; rely on running from `daran/`.
- `cv_sorting.py` imports `kinematic`, `transform`, and `SerialServoRunning` which **do not exist** in the repo. The file is partial / non-runnable as-is. `cv_sorting.ipynb` is the working version.
- Joint-angle limits are duplicated in multiple places (class defaults in `arm_robot.py`, per-script `max_list_temp` / `min_list_temp` literals). When changing limits, search for both.
- `arm_six_axis.py` uses `umatrix.py` (a vendored MicroPython linalg module). Don't try to swap it for NumPy without checking call sites — most newer code already uses NumPy directly.
- The MuJoCo XMLs under `assets/Dr_arm_6_desk/mujoco/` have produced "Nan/Inf in QACC" warnings (`MUJOCO_LOG.TXT`); the URDF under `assets/Dr_arm_6_desk/urdf/` is the more reliable model.
