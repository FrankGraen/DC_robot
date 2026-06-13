"""task2:A 点抓取 → B 点放置(示教点回放)。

参考点已用示教方式记录在 `ref_pose.json` 中,共 4 条,顺序固定:
    [0] A_safe   — A 上方安全位
    [1] A_grasp  — A 抓取位(下降到刚好接触物体)
    [2] B_safe   — B 上方安全位
    [3] B_place  — B 放置位

夹爪开合宽度:打开 6 mm,夹紧 1.8 mm。

执行序列(任务起始与终止时机械臂均自动回到竖直 park 位姿):
    park → A_safe → (打开夹爪) → A_grasp → (夹紧)
         → A_safe → B_safe → B_place → (打开夹爪)
         → B_safe → park

脚本沿用 task1 的"先仿真后实机"范式:
  * `MOVE_REAL_ROBOT = False` 时,仅生成关节插值轨迹并保存 gif 动画;
  * 把 `MOVE_REAL_ROBOT` 改成 True 才会通过串口下发到一体化关节。

所有安全检查 (关节限位 / 桌面 z 下限 / safe 高度 / 下降幅度) 在仿真阶段就会运行,
任一项不满足就抛 ValueError 终止脚本,而不会触达实机。
"""
from __future__ import annotations
import os
# 默认走非交互式后端,避免 conda 环境与系统 Qt 版本冲突导致 "xcb plugin / IOT" 段错误。
# 想交互式预览动画的话可以在外部 export MPLBACKEND=Qt5Agg 覆盖。
os.environ.setdefault('MPLBACKEND', 'Agg')

import json
import sys
import time
import warnings

# 屏蔽 Agg 后端关于 "cannot show the figure" 的提示;我们只保存 gif,不需要 plt.show。
warnings.filterwarnings('ignore', message='Matplotlib is currently using agg')

import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3  # noqa: F401  (供调试时手动构造位姿)

# ---- 把存放 arm_robot.py 的目录加入 sys.path ----
_search_root = os.path.abspath(os.path.dirname(__file__))
for _ in range(5):
    if os.path.isfile(os.path.join(_search_root, 'arm_robot.py')):
        if _search_root not in sys.path:
            sys.path.insert(0, _search_root)
        break
    _search_root = os.path.dirname(_search_root)
else:
    raise ImportError('未在最近 5 层父目录中定位到 arm_robot.py;请在 daran/ 或其子目录下运行本脚本')

np.set_printoptions(precision=3, suppress=True)

# ============================================================
# 1) 参数
# ============================================================
# --- 仿真/实机开关 ---
MOVE_REAL_ROBOT = True         # 先 False 跑仿真,确认 gif 无误后再人工改 True 由用户运行

# --- 参考点 ---
REF_POSE_PATH = os.path.join(os.path.dirname(__file__), 'ref_pose.json')
# 任务的起止位姿:竖直 park,所有连杆 z>=0,作为搬运前后的固定安全位。
PARK_POSE_DEG = [0, 90, 0, 0, 0, 0]

# --- 夹爪 ---
# 注意:`gripper.grasp()` 内部的角度公式有 bug(系数错 10×、符号反),已在 task1/arm_gui_control.py
# 中实测过。这里直接绕开 `grasp()`,调底层 `set_angle_adaptive(id_num=7, ...)`,公式按实测校准。
#   angle_deg     = width_mm * GRIPPER_DEG_PER_MM        (≈ -5.73 °/mm,负角张开)
#   rpm           = GRIPPER_SPEED_MM_S/(d/2)/(2π)*60
#   torque_Nm     = force_N * (d/2) / 1000
# 等待靠 `detect_wideth_grasp()` 实测回读宽度收敛,放弃不可靠的 `position_done`(自适应模式力限
# 提前停转时控制器不会把它置 1)。
GRIPPER_ID             = 7
GRIPPER_GEAR_D_MM      = 10
GRIPPER_DEG_PER_MM     = -180.0 / (np.pi * GRIPPER_GEAR_D_MM)   # ≈ -5.7296
GRIPPER_OPEN_WIDTH_MM  = 6.0     # 打开,送物之前 / 释放物体
GRIPPER_CLOSE_WIDTH_MM = 1.8     # 夹紧物体
GRIPPER_SPEED_MM_S     = 10.0    # 与 arm_gui 默认值对齐
GRIPPER_OPEN_FORCE_N   = 20.0    # 张开需克服静摩擦
# 闭爪 = 接触前需要克服静摩擦,接触后由自适应力限决定夹紧力。
# 历次实测:
#   10N (τ=0.05 Nm) — 电机不动 (静摩擦太大)
#   20N (τ=0.10 Nm) — 能闭合到 2.29 mm,但搬运中夹不住物块
#   40N (τ=0.20 Nm) — 当前值,与 arm_gui_control.py 默认 50N 同量级,夹紧力 ≈ 40 N
# 若物块仍打滑可继续提到 50–60N;若反过来压坏物体,把这里调小。
GRIPPER_CLOSE_FORCE_N  = 40.0
GRIPPER_WIDTH_TOL_MM   = 1.0     # 实测宽度与目标在 ±tol 内视为到位
GRIPPER_SETTLE_S       = 0.5     # 到位后再静置 (让物理震荡衰减)

# --- 关节运动 ---
JOINT_SPEED_RPM      = 2.0
SEG_WAYPOINTS        = 40        # 仿真时每个分段插值的中间点数

# --- 安全限位 ---
QMIN_DEG = [-160, -40, -160, -160, -180, -180]
QMAX_DEG = [ 160, 180,  160,  160,  180,  180]
DESK_Z_MIN_M        = 0.0        # 所有 DH 中间坐标系原点的 z 下限(m)
SAFE_Z_MIN_MM       = 130.0      # A_safe / B_safe 末端帧 z 必须 ≥ 该值
GRASP_DROP_MAX_MM   = 120.0      # safe → grasp 之间 EE z 下降幅度上限(防示教漂移)
EE_Z_MIN_MM         = -10.0      # 整条轨迹上 F6 z 的下限(给 A_grasp / B_place 留余量)

# --- 串口 ---
CAN_BRIDGE_PORT = '/dev/serial/by-id/usb-Dr-Tech_DR-USB_CAN_9A856B82094B-if00'
SERIAL_BAUD     = 115200

# --- 仿真输出 ---
ANIMATION_PATH = os.path.join(os.path.dirname(__file__), 'pick_and_place_anim.gif')
PLOT_BOX       = [-0.20, 0.40, -0.30, 0.30, 0.00, 0.42]
# 仅在 MOVE_REAL_ROBOT=False 时生成 gif;实机模式下沿用上次的 gif,避免每次 30s+ 的渲染。

# --- 实机等待:替代库自带的紧轮询 pose_done / grasp_done ---
ARM_WAIT_TIMEOUT_S       = 25.0  # 单段关节运动等待上限 — 超时 = 真故障,必须 raise 中止
GRIPPER_OPEN_TIMEOUT_S   = 5.0   # 开爪(无负载)等待上限 — 超时 = 真故障,必须 raise 中止
GRIPPER_CLOSE_WAIT_S     = 2.5   # 闭爪(夹物)等待上限 — 自适应力限可能让 position_done 永远不置 1,
                                 # 故超时被视为"已经卡到物体",仅警告并继续
WAIT_POLL_PERIOD_S       = 0.08  # 控制器属性查询周期(过小会拉低 CAN 带宽)
WAIT_PROGRESS_PERIOD_S   = 0.5   # 终端进度刷新间隔

# ============================================================
# 2) DH 模型(与 task1 完全一致)
# ============================================================
LINK_PARAMS = (
    dict(alpha=np.pi / 2),
    dict(a=0.15),
    dict(a=0.15),
    dict(d=-0.05494, alpha=np.pi / 2, offset=np.pi / 2),
    dict(d=0.068,    alpha=-np.pi / 2),
    dict(d=0.033),
)


def make_dfarm() -> DHRobot:
    links = [
        RevoluteDH(qlim=np.deg2rad([QMIN_DEG[i], QMAX_DEG[i]]), **kw)
        for i, kw in enumerate(LINK_PARAMS)
    ]
    return DHRobot(links, name='DFarm_StdDH')


# ============================================================
# 3) 安全检查
# ============================================================
def verify_within_limits(q_deg, label: str = '') -> None:
    arr = np.asarray(q_deg, dtype=float)
    bad = [(i, arr[i]) for i in range(6)
           if arr[i] < QMIN_DEG[i] or arr[i] > QMAX_DEG[i]]
    if bad:
        detail = '; '.join(
            f'J{i + 1}: {v:+.2f}° not in [{QMIN_DEG[i]:+.1f}, {QMAX_DEG[i]:+.1f}]'
            for i, v in bad
        )
        raise ValueError(f'[{label}] 关节模型角越界 -> {detail}')


def verify_above_desk(dfarm: DHRobot, q_rad, label: str = '',
                      z_min: float = DESK_Z_MIN_M) -> None:
    Ts = dfarm.fkine_all(q_rad)
    below = [(i, float(Ts[i].t[2]))
             for i in range(1, len(Ts))
             if float(Ts[i].t[2]) < z_min - 1e-9]
    if below:
        detail = '; '.join(f'F{i}: z={z * 1000:+.1f}mm' for i, z in below)
        raise ValueError(
            f'[{label}] 关节坐标系低于桌面安全线 (< {z_min * 1000:.0f} mm) -> ' + detail
        )


def verify_ee_z_floor(dfarm: DHRobot, joint_path_deg, label: str = '',
                      z_min_mm: float = EE_Z_MIN_MM) -> None:
    z_mm = np.asarray([dfarm.fkine(np.deg2rad(q)).t[2] for q in joint_path_deg]) * 1000.0
    if z_mm.min() < z_min_mm - 1e-6:
        k = int(np.argmin(z_mm))
        raise ValueError(
            f'[{label}] 整条轨迹末端 z 最低 {z_mm.min():.1f} mm < 阈值 {z_min_mm:.1f} mm '
            f'(发生在第 {k}/{len(z_mm) - 1} 个采样点);可能需要重新示教或抬高 safe 点'
        )


def fk_ee_xyz_mm(dfarm: DHRobot, q_deg) -> np.ndarray:
    return np.asarray(dfarm.fkine(np.deg2rad(q_deg)).t) * 1000.0


# ============================================================
# 4) 载入参考点 + 整体合理性检查
# ============================================================
def load_waypoints(path: str = REF_POSE_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if len(data) < 4:
        raise ValueError(
            f'ref_pose.json 需要 4 条记录 (A_safe/A_grasp/B_safe/B_place);'
            f' 当前只有 {len(data)} 条'
        )
    names = ['A_safe', 'A_grasp', 'B_safe', 'B_place']
    out = []
    for i, name in enumerate(names):
        q = np.asarray(data[i]['joints_deg'], dtype=float)
        if q.size != 6:
            raise ValueError(f'第 {i + 1} 条 joints_deg 长度 != 6')
        verify_within_limits(q, label=name)
        out.append((name, q))
    return out


def safety_check_reference_set(dfarm: DHRobot, waypoints) -> None:
    """对 4 个示教点本身做静态体检 — 在生成任何轨迹之前先跑。"""
    names = [w[0] for w in waypoints]
    qs    = [w[1] for w in waypoints]
    ee    = [fk_ee_xyz_mm(dfarm, q) for q in qs]

    print('=== 参考点 FK 校验 ===')
    for n, q, p in zip(names, qs, ee):
        print(f'  {n:8s}  q={np.round(q, 2).tolist()}'
              f'  EE(mm)={np.round(p, 1).tolist()}')

    # 桌面保护:所有 DH 坐标系原点 z >= DESK_Z_MIN_M
    for n, q in zip(names, qs):
        verify_above_desk(dfarm, np.deg2rad(q), label=n)

    # safe 点必须够高
    if ee[0][2] < SAFE_Z_MIN_MM:
        raise ValueError(f'A_safe EE z={ee[0][2]:.1f} mm < {SAFE_Z_MIN_MM} mm,请重新示教或抬高')
    if ee[2][2] < SAFE_Z_MIN_MM:
        raise ValueError(f'B_safe EE z={ee[2][2]:.1f} mm < {SAFE_Z_MIN_MM} mm,请重新示教或抬高')

    # safe 必须高于对应的抓取/放置点
    if ee[0][2] <= ee[1][2]:
        raise ValueError(
            f'A_safe EE z={ee[0][2]:.1f} 应严格大于 A_grasp EE z={ee[1][2]:.1f};示教错位'
        )
    if ee[2][2] <= ee[3][2]:
        raise ValueError(
            f'B_safe EE z={ee[2][2]:.1f} 应严格大于 B_place EE z={ee[3][2]:.1f};示教错位'
        )

    # safe → grasp 下降幅度上限,防止"safe 拍得太高,grasp 太低"导致大下沉
    dz_a = ee[0][2] - ee[1][2]
    dz_b = ee[2][2] - ee[3][2]
    if dz_a > GRASP_DROP_MAX_MM:
        raise ValueError(
            f'A_safe→A_grasp Δz={dz_a:.1f} mm > {GRASP_DROP_MAX_MM} mm,标点幅度异常'
        )
    if dz_b > GRASP_DROP_MAX_MM:
        raise ValueError(
            f'B_safe→B_place Δz={dz_b:.1f} mm > {GRASP_DROP_MAX_MM} mm,标点幅度异常'
        )

    print(f'  Δz(A_safe→A_grasp) = {dz_a:.1f} mm')
    print(f'  Δz(B_safe→B_place) = {dz_b:.1f} mm')
    print('=== 参考点校验通过 ===\n')


# ============================================================
# 5) 计划与轨迹构造
# ============================================================
# kind ∈ {'move', 'gripper'}
def build_plan(waypoints):
    A_safe  = waypoints[0][1]
    A_grasp = waypoints[1][1]
    B_safe  = waypoints[2][1]
    B_place = waypoints[3][1]
    park = np.asarray(PARK_POSE_DEG, dtype=float)
    return [
        ('Park→A_safe',     'move',          ('A_safe',   A_safe)),
        ('Open gripper',    'gripper_open',  GRIPPER_OPEN_WIDTH_MM),
        ('A_safe→A_grasp',  'move',          ('A_grasp',  A_grasp)),
        ('Close gripper',   'gripper_close', GRIPPER_CLOSE_WIDTH_MM),
        ('A_grasp→A_safe',  'move',          ('A_safe',   A_safe)),
        ('A_safe→B_safe',   'move',          ('B_safe',   B_safe)),
        ('B_safe→B_place',  'move',          ('B_place',  B_place)),
        ('Open gripper',    'gripper_open',  GRIPPER_OPEN_WIDTH_MM),
        ('B_place→B_safe',  'move',          ('B_safe',   B_safe)),
        ('B_safe→Park',     'move',          ('Park',     park)),
    ]


def jtraj_seg(q_start_deg, q_end_deg, n: int = SEG_WAYPOINTS) -> np.ndarray:
    """关节空间五次多项式插值;返回 (n, 6) 的 deg。"""
    tr = rtb.jtraj(np.deg2rad(q_start_deg), np.deg2rad(q_end_deg), n)
    return np.rad2deg(tr.q)


def build_trajectories(dfarm: DHRobot, plan, start_q_deg):
    """对所有 'move' 分段生成关节插值轨迹并做安全检查;
    'gripper' 占位为持续 3 帧的静止段,只用于动画。"""
    cur_q = np.asarray(start_q_deg, dtype=float)
    segs = []
    for label, kind, payload in plan:
        if kind == 'move':
            seg_name, target = payload
            target = np.asarray(target, dtype=float)
            path = jtraj_seg(cur_q, target)

            # 每个采样点都做关节限位 + 桌面 + EE z 下限检查
            arr = np.asarray(path, dtype=float)
            verify_within_limits(arr.min(axis=0), label=f'{seg_name}/min')
            verify_within_limits(arr.max(axis=0), label=f'{seg_name}/max')
            for k, q in enumerate(np.deg2rad(arr)):
                verify_above_desk(dfarm, q, label=f'{seg_name}/k={k}')
            verify_ee_z_floor(dfarm, arr, label=seg_name)

            segs.append((label, kind, path, target))
            cur_q = target.copy()
        elif kind in ('gripper_open', 'gripper_close'):
            # 用静止帧表示夹爪事件,便于在动画上看到停顿
            hold = np.tile(cur_q, (3, 1))
            segs.append((label, kind, hold, payload))
        else:
            raise ValueError(f'未知 kind = {kind}')
    return segs


# ============================================================
# 6) 仿真(PyPlot gif)
# ============================================================
def simulate(dfarm: DHRobot, waypoints, segs) -> None:
    from roboticstoolbox.backends.PyPlot import PyPlot

    env = PyPlot()
    env.launch(name='Pick & Place', limits=PLOT_BOX)
    env.add(dfarm)
    ax = env.ax

    # 标注 A_grasp / B_place 位置
    A_grasp_xyz = fk_ee_xyz_mm(dfarm, waypoints[1][1]) / 1000.0
    B_place_xyz = fk_ee_xyz_mm(dfarm, waypoints[3][1]) / 1000.0
    A_safe_xyz  = fk_ee_xyz_mm(dfarm, waypoints[0][1]) / 1000.0
    B_safe_xyz  = fk_ee_xyz_mm(dfarm, waypoints[2][1]) / 1000.0

    ax.scatter(*A_grasp_xyz, color='limegreen',  s=120, marker='o',
               label='A_grasp', depthshade=False)
    ax.scatter(*B_place_xyz, color='magenta',    s=120, marker='X',
               label='B_place', depthshade=False)
    ax.scatter(*A_safe_xyz,  color='limegreen',  s=60,  marker='^',
               alpha=0.6, depthshade=False, label='A_safe')
    ax.scatter(*B_safe_xyz,  color='magenta',    s=60,  marker='^',
               alpha=0.6, depthshade=False, label='B_safe')
    ax.legend(loc='upper left', fontsize=8)

    frames = []
    for label, kind, path, _ in segs:
        for q_deg in path:
            dfarm.q = np.deg2rad(q_deg)
            env.step(0.05)
            frames.append(env.getframe())

    frames[0].save(
        ANIMATION_PATH,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=80,
        loop=0,
    )
    print(f'仿真动画已保存: {ANIMATION_PATH} (帧数 {len(frames)})')


# ============================================================
# 7) 实机执行
# ============================================================
def query_joint_state(arm):
    raw = arm.read_joints()
    if raw is False:
        return False, False
    return raw, arm.servo_to_model(raw)


def _wait_positions_done(arm, id_list, timeout_s: float, label: str) -> bool:
    """替代库自带的 positions_done / position_done:逐 ID 轮询 dr.controller.position_done,
    带超时与终端进度回显。返回 True 表示全部上报完成,False 表示超时强制退出。

    arm.pose_done() / arm.grasp_done() 是无超时的紧轮询,在夹爪自适应控制力矩限位提前停止
    时控制器可能永远不会把 position_done 置 1,导致永久阻塞 — 因此实机阶段一律走本函数。
    """
    t0 = time.time()
    pending = set(id_list)
    last_print = 0.0
    while pending:
        elapsed = time.time() - t0
        if elapsed >= timeout_s:
            print(f'\r  [{label}] TIMEOUT {elapsed:5.1f}s,未完成关节 ID={sorted(pending)};'
                  '继续后续步骤(请检查物理状态)。' + ' ' * 8, flush=True)
            return False
        if elapsed - last_print >= WAIT_PROGRESS_PERIOD_S:
            print(f'\r  [{label}] waiting {elapsed:5.1f}s '
                  f'(pending ID={sorted(pending)})', end='', flush=True)
            last_print = elapsed
        done_now = []
        for jid in list(pending):
            try:
                v = arm.read_property(id_num=jid, property='dr.controller.position_done')
            except Exception:
                v = None
            if v == 1:
                done_now.append(jid)
        for jid in done_now:
            pending.discard(jid)
        if pending:
            time.sleep(WAIT_POLL_PERIOD_S)
    elapsed = time.time() - t0
    print(f'\r  [{label}] done in {elapsed:5.2f}s' + ' ' * 40, flush=True)
    return True


def _move_arm_with_progress(arm, target_deg, label: str) -> None:
    """关节移动 + 等待全部到位。任何关节超时都视为真故障并 raise,避免后续动作在异常位姿下危险下发。"""
    target_list = list(target_deg)
    if arm.set_arm_joints(angle_list=target_list, speed=JOINT_SPEED_RPM) is False:
        raise RuntimeError(f'{label}: set_arm_joints 返回 False')
    if not _wait_positions_done(arm, arm.ID_list, ARM_WAIT_TIMEOUT_S, label=label):
        raise RuntimeError(
            f'{label}: 关节未在 {ARM_WAIT_TIMEOUT_S}s 内到位,中止任务以防干涉。'
            ' 请检查电机状态/限位/是否撞到障碍。'
        )


def _send_gripper(arm, width_mm: float, force_n: float) -> None:
    """绕开 `arm.grasp()` 的公式 bug,直接对 ID=7 调用 `set_angle_adaptive`。
    速度/力矩换算与 task1/arm_gui_control.py 完全一致。"""
    width_mm = float(width_mm)
    angle_deg = width_mm * GRIPPER_DEG_PER_MM
    r = GRIPPER_GEAR_D_MM / 2.0
    rpm = GRIPPER_SPEED_MM_S / r / (2 * np.pi) * 60
    torque_nm = float(force_n) * r / 1000.0
    print(f'  [grip-cmd] width={width_mm:.2f} mm → angle={angle_deg:+.2f}°, '
          f'rpm={rpm:.1f}, τ={torque_nm:.3f} Nm')
    arm.set_angle_adaptive(id_num=GRIPPER_ID, angle=angle_deg,
                           speed=rpm, torque=torque_nm)


def _wait_gripper_width(arm, target_width_mm: float, timeout_s: float, label: str):
    """轮询 detect_wideth_grasp,直到实测宽度落入 ±GRIPPER_WIDTH_TOL_MM,或超时。
    返回 (done: bool, last_width_mm: float | None)。"""
    t0 = time.time()
    last_print = 0.0
    last_w = None
    while True:
        elapsed = time.time() - t0
        try:
            w_raw = arm.detect_wideth_grasp()
            w_val = abs(float(w_raw)) if w_raw is not None else None
        except Exception:
            w_val = None
        if w_val is not None:
            last_w = w_val
            if abs(w_val - target_width_mm) <= GRIPPER_WIDTH_TOL_MM:
                print(f'\r  [{label}] done in {elapsed:5.2f}s, width≈{w_val:.2f} mm '
                      f'(target {target_width_mm:.2f} ±{GRIPPER_WIDTH_TOL_MM:.1f})'
                      + ' ' * 10, flush=True)
                return True, w_val
        if elapsed >= timeout_s:
            wstr = f'{last_w:.2f} mm' if last_w is not None else '?'
            print(f'\r  [{label}] TIMEOUT {elapsed:5.1f}s; 实测 width≈{wstr}, '
                  f'目标 {target_width_mm:.2f} mm' + ' ' * 10, flush=True)
            return False, last_w
        if elapsed - last_print >= WAIT_PROGRESS_PERIOD_S:
            wstr = f'{w_val:.2f}' if w_val is not None else '?'
            print(f'\r  [{label}] waiting {elapsed:5.1f}s; width={wstr} mm '
                  f'→ {target_width_mm:.2f} mm', end='', flush=True)
            last_print = elapsed
        time.sleep(WAIT_POLL_PERIOD_S)


def _grasp_with_progress(arm, width_mm: float, label: str, *, is_open: bool) -> None:
    """夹爪动作 + 等待到位(轮询实测宽度) + 沉降时间。

    is_open=True  释放/张爪 — 无外力阻挡,实测宽度必须收敛到目标,否则视为故障并 raise。
                  这一类动作出现在"下降抓取"/"放置后撤离"的关键前置位,**绝不可静默通过**。
    is_open=False 闭合夹物 — 自适应力矩限位可能在到达目标角前停转,实测宽度可能停在
                  > target 处。超时是预期行为,只警告并继续。
    """
    force = GRIPPER_OPEN_FORCE_N if is_open else GRIPPER_CLOSE_FORCE_N
    _send_gripper(arm, width_mm, force)
    timeout = GRIPPER_OPEN_TIMEOUT_S if is_open else GRIPPER_CLOSE_WAIT_S
    done, w = _wait_gripper_width(arm, width_mm, timeout, label=label)
    if is_open and not done:
        wstr = f'{w:.2f}' if w is not None else '?'
        raise RuntimeError(
            f'{label}: 张爪超时 ({timeout}s) — 实测 width≈{wstr} mm,'
            f'目标 {width_mm:.2f} mm。'
            ' 拒绝继续后续下降/接触动作,以免干涉。'
            ' 请检查夹爪电源/线缆,或先用 task1/arm_gui_control.py 验证夹爪可控。'
        )
    if not done:
        wstr = f'{w:.2f}' if w is not None else '?'
        print(f'  [{label}] 夹物力限触发 (预期),实测 width≈{wstr} mm(目标 {width_mm:.2f} mm)')
    time.sleep(GRIPPER_SETTLE_S)


def execute_on_hardware(plan, dfarm: DHRobot) -> None:
    import arm_robot as robot

    arm = robot.arm_robot(
        L_p             = 0,
        L_p_mass_center = 0,
        MAX_list_temp   = QMAX_DEG,
        MIN_list_temp   = QMIN_DEG,
        G_p             = 0,
        com             = CAN_BRIDGE_PORT,
        uart_baudrate   = SERIAL_BAUD,
    )

    raw0, model0 = query_joint_state(arm)
    if model0 is False:
        raise RuntimeError('读取当前关节角失败,中止运动。')
    print(f'初始 model = {np.round(model0, 2).tolist()}')

    # 起始安全位:无论上一次任务结束在哪里,先回到竖直 park,夹爪打开释放物体。
    # plan 的第 0 步是 Park→A_safe,这一步只是把任意起点收敛到 plan 假定的起点。
    print(f'[init] move to park 姿态(任务起始安全位)'
          f' speed={JOINT_SPEED_RPM} rpm,预计 {ARM_WAIT_TIMEOUT_S}s 内完成')
    _move_arm_with_progress(arm, list(PARK_POSE_DEG), label='init/park')
    time.sleep(0.4)

    print(f'[init] gripper open ({GRIPPER_OPEN_WIDTH_MM} mm)')
    _grasp_with_progress(arm, GRIPPER_OPEN_WIDTH_MM, label='init/open', is_open=True)

    # 按 plan 顺序执行(plan 末尾自带 B_safe→Park,任务终止时回到竖直 park)
    n_steps = len(plan)
    for idx, (label, kind, payload) in enumerate(plan):
        tag = f'{idx + 1:>2d}/{n_steps}'
        if kind == 'move':
            seg_name, target = payload
            target = np.asarray(target, dtype=float)
            verify_within_limits(target, label=seg_name)   # 二次保险
            print(f'[move {tag}] {label:18s} -> {np.round(target, 2).tolist()}')
            _move_arm_with_progress(arm, target.tolist(), label=seg_name)
            time.sleep(0.2)
        elif kind in ('gripper_open', 'gripper_close'):
            width = float(payload)
            is_open = (kind == 'gripper_open')
            print(f'[grip {tag}] {label:18s} width = {width} mm  '
                  f'({"释放/张爪,超时=故障" if is_open else "夹物,允许力限提前停"})')
            _grasp_with_progress(arm, width, label=label, is_open=is_open)
        else:
            raise ValueError(f'未知 kind = {kind}')

    raw_after, model_after = query_joint_state(arm)
    if model_after is not False:
        print(f'结束 model = {np.round(model_after, 2).tolist()}')
        T_done = dfarm.fkine(np.deg2rad(model_after))
        print(f'FK 末端 mm = {np.round(np.asarray(T_done.t) * 1000, 1).tolist()}')


# ============================================================
# 8) 主流程
# ============================================================
def main() -> None:
    dfarm = make_dfarm()
    waypoints = load_waypoints()
    safety_check_reference_set(dfarm, waypoints)

    plan = build_plan(waypoints)
    print('=== 计划 ===')
    for i, (label, kind, payload) in enumerate(plan):
        if kind == 'move':
            seg_name, target = payload
            print(f'  {i:>2d}. [move      ] {label:18s} target={np.round(target, 2).tolist()}')
        elif kind == 'gripper_open':
            print(f'  {i:>2d}. [grip-open ] {label:18s} width={payload} mm  '
                  '(无负载,超时=故障)')
        elif kind == 'gripper_close':
            print(f'  {i:>2d}. [grip-close] {label:18s} width={payload} mm  '
                  '(夹物,允许力限提前停)')
        else:
            print(f'  {i:>2d}. [{kind}] {label}')
    print()

    # 安全扫描:构造插值轨迹并对每一段做关节限位 + 桌面 + EE z 三重检查
    segs = build_trajectories(dfarm, plan, PARK_POSE_DEG)
    n_total = sum(s[2].shape[0] for s in segs)
    print(f'轨迹分段 {len(segs)},采样点合计 {n_total}\n')

    if not MOVE_REAL_ROBOT:
        # 仿真模式:渲染 gif 供人工核查
        simulate(dfarm, waypoints, segs)
        print('\nMOVE_REAL_ROBOT = False,未发送实机命令。')
        print('确认 gif 中轨迹无碰撞、参考点正确后,把 MOVE_REAL_ROBOT 改成 True 再运行。')
        return

    # 实机模式:跳过 gif 渲染(避免每次 30s+ 等待);沿用上次的 ANIMATION_PATH
    print(f'MOVE_REAL_ROBOT = True,跳过仿真 gif 重新生成(沿用上次的 {ANIMATION_PATH})。')
    print('=== 进入实机执行 ===')
    execute_on_hardware(plan, dfarm)
    print('\n=== 任务完成 ===')


if __name__ == '__main__':
    main()
