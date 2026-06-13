"""夹爪(ID=7)直连调试脚本

目的:
  - 确认 ID=7 总线响应、读角度可用
  - 直接用最低层 ``set_angle_adaptive(id_num=7, angle, speed, torque)``
    分别下发 0° / 负角度 / 正角度,实测哪个方向是张开
  - 给出 angle ↔ wideth 的真实系数,用于修正 ``gripper.grasp/detect_wideth_grasp``

使用:
  python debug_gripper.py --port /dev/ttyACM1 --probe          # 只读状态
  python debug_gripper.py --port /dev/ttyACM1 --target -5      # 单点测试(-5°)
  python debug_gripper.py --port /dev/ttyACM1 --sweep          # 默认扫描:0,-2,-5,-10,-20,-40,-80,0
  python debug_gripper.py --port /dev/ttyACM1 --pos-sweep      # 反向扫描(确认正角度方向)

约定:
  上位机已经告知:0°=刚好闭合,负角=张开。脚本会按这条假设走,
  并把每一步实际角度、宽度估计同时打印出来。
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time

_p = os.path.abspath(os.path.dirname(__file__))
for _ in range(5):
    if os.path.isfile(os.path.join(_p, 'arm_robot.py')):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        break
    _p = os.path.dirname(_p)
else:
    raise ImportError('未在最近 5 层父目录中定位到 arm_robot.py')

import arm_robot as robot

GRIPPER_ID = 7
GEAR_D_MM = 10  # gripper.py 里的 self.d

# ====== 安全参数 ======
# 在搞清楚物理行程之前,不允许测试目标超过这条软极限(单位:度)。
# 0=刚好闭合,负=张开,所以"最负"才是张开侧的上限。
SAFETY_NEG_LIMIT_DEG = -45.0
SAFETY_POS_LIMIT_DEG = +10.0
# 堵转判据:连续 STALL_WINDOW 个样本角度极差 < STALL_EPS_DEG 且 |Δtarget| > STALL_FAR_DEG → 视为堵转
STALL_WINDOW = 5
STALL_EPS_DEG = 0.3
STALL_FAR_DEG = 1.0
# 退出时的"安全保持力矩"——保持当前角度但用很弱的推力,不再硬顶限位
SAFE_HOLD_FORCE_N = 5


def angle_to_wideth_existing(angle_deg: float) -> float:
    """gripper.py 当前实现的 detect_wideth_grasp 公式(带负号)"""
    return -angle_deg / 180.0 * math.pi * (GEAR_D_MM / 2) * 2


def angle_to_wideth_inverse_of_grasp(angle_deg: float) -> float:
    """grasp() 实际下发角度的反函数(没有负号,系数 ~1.745 mm/deg)"""
    # grasp: angle = w * 1.8 / pi  =>  w = angle * pi / 1.8
    return angle_deg * math.pi / 1.8


def speed_mm_to_rpm(speed_mm_s: float) -> float:
    """gripper.grasp 内部的速度转换公式,保留一致以便对比"""
    return speed_mm_s / (GEAR_D_MM / 2) / (math.pi * 2) * 60


def force_n_to_torque_nm(force_n: float) -> float:
    """gripper.grasp 内部的力矩转换公式"""
    return force_n * (GEAR_D_MM / 2 / 1000)


# ====== 安全工具:hold_here / install_safety / is_stalled ======
def hold_here(arm: 'robot.arm_robot | None', force_n: float = SAFE_HOLD_FORCE_N) -> None:
    """把夹爪关节(仅 ID=7)指向"当前角度",用极小力矩保持;不会动臂关节 1-6。

    用于:Ctrl-C / 异常退出 / 检测到堵转后,停止继续硬推。
    """
    if arm is None:
        return
    try:
        cur = arm.get_angle(id_num=GRIPPER_ID)
        if not isinstance(cur, (int, float)):
            print(f'[safe] get_angle 失败({cur!r}),无法 hold_here', flush=True)
            return
        arm.set_angle_adaptive(
            id_num=GRIPPER_ID, angle=cur, speed=0,
            torque=force_n_to_torque_nm(force_n),
        )
        print(f'[safe] hold at {cur:+.2f}° with ~{force_n} N (≈ {force_n_to_torque_nm(force_n):.3f} Nm)',
              flush=True)
    except Exception as e:
        print(f'[safe] hold_here failed: {e}', flush=True)


_safety_installed = False


def install_safety(arm: 'robot.arm_robot'):
    """注册 SIGINT 处理器 + atexit:任何退出路径都先 hold_here。"""
    global _safety_installed
    if _safety_installed:
        return
    import atexit
    import signal

    def _sig_handler(_sig, _frame):
        print('\n[signal] 收到 Ctrl-C,先释放夹爪 ...', flush=True)
        hold_here(arm)
        sys.exit(130)

    signal.signal(signal.SIGINT, _sig_handler)
    atexit.register(hold_here, arm)
    _safety_installed = True
    print('[safe] 已安装 SIGINT/atexit 保护:任何退出会先把 J7 持回当前角度', flush=True)


def is_stalled(history: list, target_deg: float) -> bool:
    """连续 STALL_WINDOW 次的角度极差很小、且离目标还很远 → 视为堵转。"""
    if len(history) < STALL_WINDOW:
        return False
    window = history[-STALL_WINDOW:]
    if any(not isinstance(v, (int, float)) for v in window):
        return False
    spread = max(window) - min(window)
    if spread >= STALL_EPS_DEG:
        return False
    return abs(window[-1] - target_deg) > STALL_FAR_DEG


def clamp_target(angle_deg: float, override: bool) -> float:
    if override:
        return angle_deg
    if angle_deg < SAFETY_NEG_LIMIT_DEG:
        print(f'  [clamp] 目标 {angle_deg:+.2f}° 超出软极限,已截断到 {SAFETY_NEG_LIMIT_DEG:+.2f}°',
              flush=True)
        return SAFETY_NEG_LIMIT_DEG
    if angle_deg > SAFETY_POS_LIMIT_DEG:
        print(f'  [clamp] 目标 {angle_deg:+.2f}° 超出软极限,已截断到 {SAFETY_POS_LIMIT_DEG:+.2f}°',
              flush=True)
        return SAFETY_POS_LIMIT_DEG
    return angle_deg


def open_arm(port: str, baud: int = 115200) -> robot.arm_robot:
    print(f'[init] 连接 {port} @ {baud} ...', flush=True)
    arm = robot.arm_robot(
        L_p=0, L_p_mass_center=0,
        MAX_list_temp=[160, 180, 160, 160, 180, 180],
        MIN_list_temp=[-160, -10, -160, -160, -180, -180],
        G_p=0, com=port, uart_baudrate=baud,
    )
    print('[init] OK', flush=True)
    return arm


def probe(arm: robot.arm_robot, repeats: int = 5, interval: float = 0.2):
    """轮询 ID=7 当前角度/速度/力矩,看总线是否通"""
    print(f'\n==== probe (ID={GRIPPER_ID}) ====', flush=True)
    for k in range(repeats):
        angle = safe_call(arm.get_angle, id_num=GRIPPER_ID)
        speed = safe_call(arm.get_speed, id_num=GRIPPER_ID)
        torque = safe_call(arm.get_torque, id_num=GRIPPER_ID)
        w_existing = angle_to_wideth_existing(angle) if isinstance(angle, (int, float)) else None
        w_inv_grasp = angle_to_wideth_inverse_of_grasp(angle) if isinstance(angle, (int, float)) else None
        print(
            f'  [{k}] angle={fmt(angle)}°  '
            f'speed={fmt(speed)} r/min  '
            f'torque={fmt(torque)} Nm  '
            f'| width(by detect)={fmt(w_existing)} mm '
            f'  width(by inv-grasp)={fmt(w_inv_grasp)} mm',
            flush=True,
        )
        time.sleep(interval)


def go(arm: robot.arm_robot, angle_deg: float,
       hold_s: float = 2.0, poll: float = 0.1,
       force_n: float = 50, speed_mm_s: float = 10,
       stall_check: bool = True, override_limit: bool = False) -> bool:
    """下发一个绝对角度,在 hold_s 内连续打印实际角度。

    返回:
      True  — 命令期间没检测到堵转(到位 / 自然停下)
      False — 检测到堵转,期间已经触发 hold_here 释放,sweep 应当中断后续目标
    """
    angle_deg = clamp_target(angle_deg, override=override_limit)
    rpm = speed_mm_to_rpm(speed_mm_s)
    torque = force_n_to_torque_nm(force_n)
    print(
        f'\n---- target angle = {angle_deg:+.2f}°  '
        f'(rpm={rpm:.2f}, torque={torque:.3f} Nm)  hold={hold_s:.1f}s '
        f'stall_check={stall_check} ----',
        flush=True,
    )
    ok = safe_call(arm.set_angle_adaptive, id_num=GRIPPER_ID,
                   angle=angle_deg, speed=rpm, torque=torque)
    print(f'  set_angle_adaptive returned: {ok!r}', flush=True)

    history: list = []
    stalled = False
    t0 = time.monotonic()
    while time.monotonic() - t0 < hold_s:
        cur = safe_call(arm.get_angle, id_num=GRIPPER_ID)
        elapsed = time.monotonic() - t0
        if isinstance(cur, (int, float)):
            history.append(cur)
            w_e = angle_to_wideth_existing(cur)
            w_g = angle_to_wideth_inverse_of_grasp(cur)
            print(
                f'    t={elapsed:4.1f}s   cur={cur:+7.2f}°   '
                f'Δ={cur - angle_deg:+6.2f}°   '
                f'w(detect)={w_e:+6.2f}mm   w(inv-grasp)={w_g:+6.2f}mm',
                flush=True,
            )
            if stall_check and is_stalled(history, angle_deg):
                stalled = True
                print(
                    f'  ⚠ STALL  连续 {STALL_WINDOW} 次极差 < {STALL_EPS_DEG}°,'
                    f'仍距目标 {abs(history[-1] - angle_deg):.2f}° > {STALL_FAR_DEG}°,'
                    f'立即释放',
                    flush=True,
                )
                hold_here(arm, force_n=SAFE_HOLD_FORCE_N)
                break
        else:
            print(f'    t={elapsed:4.1f}s   cur={cur!r}', flush=True)
        time.sleep(poll)
    return not stalled


def sweep(arm: robot.arm_robot, targets, hold_s: float = 2.0,
          force_n: float = 50, speed_mm_s: float = 10,
          stall_check: bool = True, override_limit: bool = False):
    for t in targets:
        ok = go(arm, angle_deg=t, hold_s=hold_s,
                force_n=force_n, speed_mm_s=speed_mm_s,
                stall_check=stall_check, override_limit=override_limit)
        if not ok:
            print('  [sweep] 检测到堵转,中断后续目标', flush=True)
            break


def safe_call(fn, **kwargs):
    try:
        return fn(**kwargs)
    except Exception as e:
        return f'<err {type(e).__name__}: {e}>'


def fmt(x):
    if isinstance(x, (int, float)):
        return f'{x:+.3f}'
    return str(x)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--port', required=True, help='/dev/ttyACM* 或 /dev/serial/by-id/...')
    p.add_argument('--baud', type=int, default=115200)
    p.add_argument('--probe', action='store_true', help='只读取状态')
    p.add_argument('--target', type=float, default=None,
                   help='单点目标角度(°);负值张开,0=闭合')
    p.add_argument('--sweep', action='store_true', help='张开方向扫描')
    p.add_argument('--pos-sweep', action='store_true',
                   help='正角度扫描(验证正角度是否相反方向)')
    p.add_argument('--hold', type=float, default=2.0,
                   help='每个目标的保持时间(秒),期间持续打印 cur angle')
    p.add_argument('--force', type=float, default=50,
                   help='等效力矩档位 N(对应 ~0.25 Nm)')
    p.add_argument('--speed', type=float, default=10,
                   help='等效线速度 mm/s')
    p.add_argument('--no-stall-check', action='store_true',
                   help='关闭堵转检测(默认开启;不建议关)')
    p.add_argument('--override-limit', action='store_true',
                   help=f'绕过软极限 [{SAFETY_NEG_LIMIT_DEG:+.0f}°, {SAFETY_POS_LIMIT_DEG:+.0f}°],'
                        f'仅在明确知道行程时使用')
    return p.parse_args()


def main():
    args = parse_args()
    arm = open_arm(args.port, args.baud)
    install_safety(arm)

    stall_check = not args.no_stall_check
    common = dict(
        hold_s=args.hold,
        force_n=args.force,
        speed_mm_s=args.speed,
        stall_check=stall_check,
        override_limit=args.override_limit,
    )
    try:
        probe(arm)
        if args.target is not None:
            go(arm, angle_deg=args.target, **common)
        if args.sweep:
            # 软极限 -45° 之内的保守扫描;命令外加 0 与 -2 验证小幅指令是否生效
            sweep(arm, targets=[0, -2, -5, -10, -15, -25, -40, 0], **common)
        if args.pos_sweep:
            sweep(arm, targets=[0, +2, +5, +10, 0], **common)
        if not (args.probe or args.target is not None or args.sweep or args.pos_sweep):
            print('\n[hint] 你没有指定动作,只做了 probe。'
                  '试试  --sweep  或  --target -10  --hold 3', flush=True)
    finally:
        # hold_here 已经通过 atexit 注册,这里只关串口。
        try:
            hold_here(arm)  # 即便没异常,也显式保持一次再关串口
        except Exception:
            pass
        try:
            arm.uart.close()
            print('\n[exit] 串口已关闭', flush=True)
        except Exception:
            pass


if __name__ == '__main__':
    main()
