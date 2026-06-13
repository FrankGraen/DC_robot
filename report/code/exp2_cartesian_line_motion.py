from roboticstoolbox import DHRobot, RevoluteDH
import roboticstoolbox as rtb
import numpy as np

QMIN_DEG = [-160, -40, -160, -160, -180, -180]
QMAX_DEG = [160, 180, 160, 160, 180, 180]

LINK_PARAMS = (
    dict(alpha=np.pi / 2),
    dict(a=0.15),
    dict(a=0.15),
    dict(d=-0.05494, alpha=np.pi / 2, offset=np.pi / 2),
    dict(d=0.068, alpha=-np.pi / 2),
    dict(d=0.033),
)


def make_dfarm() -> DHRobot:
    links = [
        RevoluteDH(qlim=np.deg2rad([QMIN_DEG[i], QMAX_DEG[i]]), **kw)
        for i, kw in enumerate(LINK_PARAMS)
    ]
    return DHRobot(links, name="DFarm_StdDH")


dfarm = make_dfarm()
q_start_deg = np.array([0, 90, 0, 0, 0, 0], dtype=float)
path_end_mm = np.array([100, 200, 200], dtype=float)
waypoint_count = 40

T_start = dfarm.fkine(np.deg2rad(q_start_deg))
T_end = T_start.copy()
T_end.t = path_end_mm / 1000.0
cartesian_path = rtb.tools.trajectory.ctraj(T_start, T_end, waypoint_count)

joint_path_deg = []
q_seed = np.deg2rad(q_start_deg)
for idx, T_step in enumerate(cartesian_path):
    ik = dfarm.ikine_LM(T_step, q0=q_seed, joint_limits=True, ilimit=5000)
    if not ik.success:
        raise RuntimeError(f"IK failed at waypoint {idx}: {ik.reason}")
    q_seed = ik.q
    joint_path_deg.append(np.rad2deg(ik.q))
