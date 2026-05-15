"""DFarm 关节控制 GUI(PyQt5 + matplotlib 3D)。

功能:
  1. 6 个关节滑块/数显;改任意值后右侧 3D 仿真即时刷新
  2. "读取关节"按钮 -> 拉取实机当前角度,同步到滑块与 3D 视图
  3. "发送指令"按钮 -> 把当前滑块值作为目标下发;执行期内以 4 Hz 轮询读取
     真实角度,与目标进行最大偏差比对。若在宽限期内仍未收敛到阈值之内,
     自动 `arm.lock()` 制动并在状态栏报警

运行:
  conda activate DC_robot
  cd daran/task1
  python arm_gui_control.py
"""
from __future__ import annotations
import os
import sys
import time
import traceback

# ---- Qt 库版本自愈:若 LD_LIBRARY_PATH 把系统 Qt 排在 conda env 之前 -------
# (典型表现:启动报 "Cannot mix incompatible Qt library 5.15.13 with 5.15.18"
#  或 "Could not load the Qt platform plugin xcb" 后核心转储)
# 这里清掉系统 lib 路径并以同样参数 re-exec 自己,实现一次性自愈。
def _purge_system_libpath_and_reexec():
    if os.environ.get('_QT_LIBPATH_FIXED') == '1':
        return
    raw = os.environ.get('LD_LIBRARY_PATH', '')
    BAD = ('/usr/lib/x86_64-linux-gnu', '/lib/x86_64-linux-gnu')
    if not any(b in raw for b in BAD):
        return
    parts = [p for p in raw.split(':') if p and not any(b in p for b in BAD)]
    env = os.environ.copy()
    if parts:
        env['LD_LIBRARY_PATH'] = ':'.join(parts)
    else:
        env.pop('LD_LIBRARY_PATH', None)
    env['_QT_LIBPATH_FIXED'] = '1'
    print('[arm_gui] 检测到 LD_LIBRARY_PATH 含系统 Qt 路径,自动清理后重启本进程 ...',
          file=sys.stderr, flush=True)
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


_purge_system_libpath_and_reexec()

# ---- 把存放 arm_robot.py 的目录加入 sys.path ----
_p = os.path.abspath(os.path.dirname(__file__))
for _ in range(5):
    if os.path.isfile(os.path.join(_p, 'arm_robot.py')):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        break
    _p = os.path.dirname(_p)
else:
    raise ImportError('未在最近 5 层父目录中定位到 arm_robot.py')

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib import rcParams, font_manager
from roboticstoolbox import DHRobot, RevoluteDH

import arm_robot as robot

# ---- 中文字体(若存在则注册;不影响 GUI 控件,只用于 matplotlib 标题) ----
_CN_FONT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'font', 'simhei.ttf'))
if os.path.exists(_CN_FONT):
    font_manager.fontManager.addfont(_CN_FONT)
    rcParams['font.sans-serif'] = [
        font_manager.FontProperties(fname=_CN_FONT).get_name(), 'DejaVu Sans'
    ]
    rcParams['axes.unicode_minus'] = False

# ---- 与 notebook 完全一致的 DH 参数 / 限位 ----
LINK_PARAMS = (
    dict(alpha=np.pi / 2),
    dict(a=0.15),
    dict(a=0.15),
    dict(d=-0.05494, alpha=np.pi / 2, offset=np.pi / 2),
    dict(d=0.068, alpha=-np.pi / 2),
    dict(d=0.033),
)
QMIN_DEG = [-160, -10, -160, -160, -180, -180]
QMAX_DEG = [160, 180, 160, 160, 180, 180]
DESK_Z_MIN_M = 0.0

# ---- 夹爪默认参数(实测校准 — 见 debug_gripper.py 记录) ----
# `gripper.grasp()` 公式系数错 10×、符号反:它给的角度是正的、且系数 ~0.573°/mm。
# 实测 0=刚好闭合、负角=张开,正确换算是 angle = -wideth * 18 / π (≈ -5.73 °/mm)。
# `detect_wideth_grasp()` 公式正确,可直接用。
GRIPPER_ID = 7
GRIPPER_GEAR_D_MM = 10
GRIPPER_DEG_PER_MM = -180.0 / (np.pi * GRIPPER_GEAR_D_MM)  # ≈ -5.7296
GRIPPER_WIDTH_MIN_MM = 0.0
GRIPPER_WIDTH_MAX_MM = 15.0   # 实测 7 mm (≈ -40°) 仍线性,留余量到 15 mm (≈ -86°)
GRIPPER_OPEN_MM = 7.0         # 默认全开:已实测安全
GRIPPER_CLOSE_MM = 0.0        # 默认闭合:0° 自然位
GRIPPER_SPEED_DEFAULT = 10    # mm/s,内部转 rpm
GRIPPER_FORCE_DEFAULT = 50    # N,内部转 ~0.25 Nm


def make_dfarm() -> DHRobot:
    links = [
        RevoluteDH(qlim=np.deg2rad([QMIN_DEG[i], QMAX_DEG[i]]), **kw)
        for i, kw in enumerate(LINK_PARAMS)
    ]
    return DHRobot(links, name='DFarm')


# =============================================================================
# 单关节的滑块行:label + slider(0.1° 精度) + spinbox
# =============================================================================
class JointRow(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(int, float)  # (joint_idx, degrees)

    def __init__(self, idx: int, qmin: float, qmax: float, parent=None):
        super().__init__(parent)
        self.idx = idx

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.label = QtWidgets.QLabel(f'J{idx + 1}')
        self.label.setFixedWidth(30)
        self.label.setStyleSheet('font-weight: bold;')

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # 用 ×10 缩放支持 0.1° 精度
        self.slider.setMinimum(int(qmin * 10))
        self.slider.setMaximum(int(qmax * 10))
        self.slider.setValue(0)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(int((qmax - qmin) * 10 / 8))

        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(qmin, qmax)
        self.spin.setDecimals(1)
        self.spin.setSingleStep(0.5)
        self.spin.setSuffix(' °')
        self.spin.setValue(0.0)
        self.spin.setFixedWidth(85)

        self.range_label = QtWidgets.QLabel(f'[{qmin:+.0f}, {qmax:+.0f}]')
        self.range_label.setFixedWidth(80)
        self.range_label.setStyleSheet('color: #888; font-size: 9pt;')

        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)

        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        layout.addWidget(self.range_label)

    def _slider_changed(self, v):
        deg = v / 10.0
        self.spin.blockSignals(True)
        self.spin.setValue(deg)
        self.spin.blockSignals(False)
        self.valueChanged.emit(self.idx, deg)

    def _spin_changed(self, deg):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(deg * 10)))
        self.slider.blockSignals(False)
        self.valueChanged.emit(self.idx, deg)

    def setValue(self, deg: float):
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(int(round(deg * 10)))
        self.spin.setValue(deg)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)

    def value(self) -> float:
        return self.spin.value()


# =============================================================================
# 主窗口
# =============================================================================
class ArmControlGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('DFarm 关节控制 GUI')
        self.dfarm = make_dfarm()
        self.arm: robot.arm_robot | None = None

        self.exec_timer = QtCore.QTimer()
        self.exec_timer.timeout.connect(self._on_exec_tick)
        self.exec_start_time: float | None = None
        self.target_q_deg: np.ndarray | None = None

        # 示教模式状态
        self.teach_active: bool = False
        self.teach_timer = QtCore.QTimer()
        self.teach_timer.timeout.connect(self._on_teach_tick)
        self.teach_last_joints_deg: list | None = None
        self.teach_last_gripper_deg: float | None = None
        self.teach_records: list[dict] = []

        self._build_ui()
        self._update_plot()

    # --- 布局 ---------------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # === 左栏 ===
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 0)

        # ---- 关节滑块 ----
        sliders_group = QtWidgets.QGroupBox('关节模型角(度)')
        sg_layout = QtWidgets.QVBoxLayout(sliders_group)
        self.rows: list[JointRow] = []
        for i in range(6):
            row = JointRow(i, QMIN_DEG[i], QMAX_DEG[i])
            row.valueChanged.connect(self._joint_changed)
            sg_layout.addWidget(row)
            self.rows.append(row)
        left.addWidget(sliders_group)

        # ---- 预置姿态 ----
        preset_box = QtWidgets.QHBoxLayout()
        btn_zero = QtWidgets.QPushButton('置零位 [0×6]')
        btn_zero.clicked.connect(lambda: self._set_q([0] * 6))
        btn_park = QtWidgets.QPushButton('竖直 park [0,90,0,0,0,0]')
        btn_park.clicked.connect(lambda: self._set_q([0, 90, 0, 0, 0, 0]))
        preset_box.addWidget(btn_zero)
        preset_box.addWidget(btn_park)
        left.addLayout(preset_box)

        # ---- 硬件 ----
        conn_group = QtWidgets.QGroupBox('硬件连接')
        cg = QtWidgets.QFormLayout(conn_group)
        self.port_edit = QtWidgets.QLineEdit('/dev/ttyACM0')
        self.baud_edit = QtWidgets.QLineEdit('115200')
        self.btn_connect = QtWidgets.QPushButton('连接')
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_read = QtWidgets.QPushButton('读取实机关节')
        self.btn_read.clicked.connect(self._on_read)
        self.btn_read.setEnabled(False)
        self.btn_send = QtWidgets.QPushButton('发送指令(运行到目标)')
        self.btn_send.setStyleSheet('font-weight: bold;')
        self.btn_send.clicked.connect(self._on_send)
        self.btn_send.setEnabled(False)
        self.btn_abort = QtWidgets.QPushButton('紧急制动 / 锁定')
        self.btn_abort.setStyleSheet(
            'QPushButton { background-color: #d9534f; color: white; font-weight: bold; }'
            'QPushButton:disabled { background-color: #d0d0d0; color: #888; font-weight: normal; }'
        )
        self.btn_abort.clicked.connect(self._on_abort)
        self.btn_abort.setEnabled(False)
        cg.addRow('串口:', self.port_edit)
        cg.addRow('波特率:', self.baud_edit)
        cg.addRow(self.btn_connect)
        cg.addRow(self.btn_read)
        cg.addRow(self.btn_send)
        cg.addRow(self.btn_abort)
        left.addWidget(conn_group)

        # ---- 示教模式 -----------------------------------------------------
        teach_group = QtWidgets.QGroupBox('示教模式(手动拖臂记录)')
        tg_layout = QtWidgets.QVBoxLayout(teach_group)

        teach_btn_row = QtWidgets.QHBoxLayout()
        self.btn_teach_toggle = QtWidgets.QPushButton('进入示教')
        self.btn_teach_toggle.setStyleSheet(
            'QPushButton { background-color: #ffc107; font-weight: bold; }'
            'QPushButton:disabled { background-color: #d0d0d0; color: #888; font-weight: normal; }'
        )
        self.btn_teach_toggle.clicked.connect(self._on_teach_toggle)
        self.btn_teach_record = QtWidgets.QPushButton('记录姿态 (Ctrl+R)')
        self.btn_teach_record.clicked.connect(self._on_teach_record)
        teach_btn_row.addWidget(self.btn_teach_toggle)
        teach_btn_row.addWidget(self.btn_teach_record)
        tg_layout.addLayout(teach_btn_row)

        # 快捷键 Ctrl+R 记录
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+R'), self,
                            activated=self._on_teach_record)

        self.teach_list = QtWidgets.QListWidget()
        self.teach_list.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked
                                        | QtWidgets.QAbstractItemView.EditKeyPressed)
        self.teach_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.teach_list.itemChanged.connect(self._on_teach_item_renamed)
        self.teach_list.setMaximumHeight(120)
        tg_layout.addWidget(self.teach_list)

        teach_btn_row2 = QtWidgets.QHBoxLayout()
        self.btn_teach_apply = QtWidgets.QPushButton('应用到滑块')
        self.btn_teach_apply.clicked.connect(self._on_teach_apply)
        self.btn_teach_delete = QtWidgets.QPushButton('删除选中')
        self.btn_teach_delete.clicked.connect(self._on_teach_delete)
        self.btn_teach_clear = QtWidgets.QPushButton('清空')
        self.btn_teach_clear.clicked.connect(self._on_teach_clear)
        self.btn_teach_export = QtWidgets.QPushButton('导出 JSON')
        self.btn_teach_export.clicked.connect(self._on_teach_export)
        teach_btn_row2.addWidget(self.btn_teach_apply)
        teach_btn_row2.addWidget(self.btn_teach_delete)
        teach_btn_row2.addWidget(self.btn_teach_clear)
        teach_btn_row2.addWidget(self.btn_teach_export)
        tg_layout.addLayout(teach_btn_row2)

        # 未连接时全部禁用
        self._teach_buttons = [
            self.btn_teach_toggle, self.btn_teach_record,
            self.btn_teach_apply, self.btn_teach_delete,
            self.btn_teach_clear, self.btn_teach_export,
        ]
        for b in self._teach_buttons:
            b.setEnabled(False)
        # 列表管理类按钮在没连接时仍可用(便于在离线下整理已有 JSON)
        self.btn_teach_apply.setEnabled(True)
        self.btn_teach_delete.setEnabled(True)
        self.btn_teach_clear.setEnabled(True)
        self.btn_teach_export.setEnabled(True)

        left.addWidget(teach_group)

        # ---- 夹爪控制(参考 daran/color_take.py 的 grasp/grasp_done 用法) ----
        gripper_group = QtWidgets.QGroupBox('夹爪控制(关节 ID=7)')
        gg = QtWidgets.QFormLayout(gripper_group)

        self.grip_width_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.grip_width_slider.setMinimum(int(GRIPPER_WIDTH_MIN_MM * 10))
        self.grip_width_slider.setMaximum(int(GRIPPER_WIDTH_MAX_MM * 10))
        self.grip_width_slider.setValue(int(GRIPPER_OPEN_MM * 10))
        self.grip_width_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.grip_width_slider.setTickInterval(50)

        self.grip_width_spin = QtWidgets.QDoubleSpinBox()
        self.grip_width_spin.setRange(GRIPPER_WIDTH_MIN_MM, GRIPPER_WIDTH_MAX_MM)
        self.grip_width_spin.setDecimals(1)
        self.grip_width_spin.setSingleStep(1.0)
        self.grip_width_spin.setSuffix(' mm')
        self.grip_width_spin.setValue(GRIPPER_OPEN_MM)
        self.grip_width_spin.setFixedWidth(85)

        self.grip_width_slider.valueChanged.connect(
            lambda v: (self.grip_width_spin.blockSignals(True),
                       self.grip_width_spin.setValue(v / 10.0),
                       self.grip_width_spin.blockSignals(False))
        )
        self.grip_width_spin.valueChanged.connect(
            lambda d: (self.grip_width_slider.blockSignals(True),
                       self.grip_width_slider.setValue(int(round(d * 10))),
                       self.grip_width_slider.blockSignals(False))
        )

        grip_width_row = QtWidgets.QHBoxLayout()
        grip_width_row.addWidget(self.grip_width_slider, 1)
        grip_width_row.addWidget(self.grip_width_spin)

        self.grip_speed_spin = QtWidgets.QSpinBox()
        self.grip_speed_spin.setRange(1, 10)
        self.grip_speed_spin.setSuffix(' mm/s')
        self.grip_speed_spin.setValue(GRIPPER_SPEED_DEFAULT)

        self.grip_force_spin = QtWidgets.QSpinBox()
        self.grip_force_spin.setRange(1, 120)
        self.grip_force_spin.setSuffix(' N')
        self.grip_force_spin.setValue(GRIPPER_FORCE_DEFAULT)

        self.btn_grip_apply = QtWidgets.QPushButton('发送(按当前宽度)')
        self.btn_grip_apply.clicked.connect(self._on_grip_apply)
        self.btn_grip_open = QtWidgets.QPushButton(f'全开 ({GRIPPER_OPEN_MM:.0f} mm)')
        self.btn_grip_open.clicked.connect(self._on_grip_open)
        self.btn_grip_close = QtWidgets.QPushButton(f'夹紧 ({GRIPPER_CLOSE_MM:.0f} mm)')
        self.btn_grip_close.clicked.connect(self._on_grip_close)
        self.btn_grip_read = QtWidgets.QPushButton('回读宽度')
        self.btn_grip_read.clicked.connect(self._on_grip_read)

        self._gripper_buttons = [
            self.btn_grip_apply, self.btn_grip_open,
            self.btn_grip_close, self.btn_grip_read,
        ]
        for b in self._gripper_buttons:
            b.setEnabled(False)

        gg.addRow('开合宽度', grip_width_row)
        gg.addRow('速度', self.grip_speed_spin)
        gg.addRow('力矩', self.grip_force_spin)

        grip_btn_row = QtWidgets.QHBoxLayout()
        grip_btn_row.addWidget(self.btn_grip_open)
        grip_btn_row.addWidget(self.btn_grip_close)
        gg.addRow(grip_btn_row)

        grip_btn_row2 = QtWidgets.QHBoxLayout()
        grip_btn_row2.addWidget(self.btn_grip_apply)
        grip_btn_row2.addWidget(self.btn_grip_read)
        gg.addRow(grip_btn_row2)

        left.addWidget(gripper_group)

        # ---- 阈值 ----
        thresh_group = QtWidgets.QGroupBox('运行监控参数')
        tg = QtWidgets.QFormLayout(thresh_group)
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.2, 90.0)
        self.threshold_spin.setSingleStep(0.5)
        self.threshold_spin.setSuffix(' °')
        self.threshold_spin.setValue(3.0)
        self.grace_spin = QtWidgets.QDoubleSpinBox()
        self.grace_spin.setRange(1.0, 120.0)
        self.grace_spin.setSingleStep(1.0)
        self.grace_spin.setSuffix(' s')
        self.grace_spin.setValue(15.0)
        self.poll_spin = QtWidgets.QDoubleSpinBox()
        self.poll_spin.setRange(0.05, 2.0)
        self.poll_spin.setSingleStep(0.05)
        self.poll_spin.setSuffix(' s')
        self.poll_spin.setValue(0.25)
        tg.addRow('到位阈值', self.threshold_spin)
        tg.addRow('宽限时间', self.grace_spin)
        tg.addRow('采样周期', self.poll_spin)
        left.addWidget(thresh_group)

        # ---- 日志 ----
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setFixedHeight(160)
        self.log.setStyleSheet('font-family: monospace; font-size: 9pt;')
        left.addWidget(self.log, 0)

        # === 右栏 ===
        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 1)

        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        right.addWidget(self.canvas, 1)

        self.status_label = QtWidgets.QLabel('就绪')
        font = self.status_label.font()
        font.setPointSize(11)
        font.setBold(True)
        self.status_label.setFont(font)
        self.status_label.setStyleSheet('padding: 6px;')
        self._set_status('就绪', 'idle')
        right.addWidget(self.status_label)

    # --- 通用工具 -----------------------------------------------------------
    def _info(self, msg: str):
        ts = time.strftime('%H:%M:%S')
        self.log.appendPlainText(f'[{ts}] {msg}')

    def _set_status(self, text: str, kind: str = 'idle'):
        colors = {
            'idle': '#eeeeee',
            'busy': '#fff3cd',
            'ok': '#d4edda',
            'err': '#f8d7da',
        }
        bg = colors.get(kind, '#eeeeee')
        self.status_label.setStyleSheet(f'padding: 6px; background-color: {bg};')
        self.status_label.setText(text)

    def _current_q(self) -> np.ndarray:
        return np.array([r.value() for r in self.rows], dtype=float)

    def _set_q(self, q_deg):
        # 设置滑块时只在最后一次触发重绘,避免连续 6 次 update_plot 卡顿
        for i, row in enumerate(self.rows):
            row.blockSignals(True)
            row.setValue(float(q_deg[i]))
            row.blockSignals(False)
        self._update_plot()

    def _set_sliders_enabled(self, en: bool):
        for r in self.rows:
            r.setEnabled(en)

    # --- 桌面保护 -----------------------------------------------------------
    def _check_desk_z(self, q_deg) -> tuple[bool, str]:
        Ts = self.dfarm.fkine_all(np.deg2rad(q_deg))
        bad = []
        for i in range(1, len(Ts)):
            z = float(Ts[i].t[2])
            if z < DESK_Z_MIN_M - 1e-9:
                bad.append((i, z * 1000))
        if bad:
            return False, '; '.join(f'F{i}: z={z:+.1f}mm' for i, z in bad)
        return True, ''

    # --- 3D 渲染 ------------------------------------------------------------
    def _joint_changed(self, idx, deg):
        # 在执行模式下,即便滑块变化也不重绘(避免和监控重绘冲突)
        if self.exec_timer.isActive():
            return
        self._update_plot()

    def _draw_arm(self, ax, q_deg, color_link, color_joint, label=None, lw=4, alpha=1.0):
        Ts = self.dfarm.fkine_all(np.deg2rad(q_deg))
        pts = np.asarray([T.t for T in Ts]) * 1000.0
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-',
                color=color_link, lw=lw, alpha=alpha,
                label=label, zorder=2)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   color=color_joint, s=35, depthshade=True, zorder=3)
        return pts

    def _draw_axes(self):
        # 桌面平面 z=0
        xx, yy = np.meshgrid(np.linspace(-300, 450, 4), np.linspace(-300, 300, 4))
        self.ax.plot_surface(xx, yy, np.zeros_like(xx),
                             color='#cccccc', alpha=0.15, edgecolor='none')
        self.ax.set_xlim(-300, 450)
        self.ax.set_ylim(-300, 300)
        self.ax.set_zlim(-50, 450)
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')

    def _update_plot(self):
        q = self._current_q()
        self.ax.clear()
        self._draw_axes()
        pts = self._draw_arm(self.ax, q, '#4a90d9', '#1f5fa5')
        ee = pts[-1]
        ok_desk, _ = self._check_desk_z(q)
        title = f'TCP ({ee[0]:+6.1f}, {ee[1]:+6.1f}, {ee[2]:+6.1f}) mm'
        if not ok_desk:
            title += '   [⚠ 低于桌面]'
        self.ax.set_title(title)
        self.canvas.draw_idle()

    def _update_plot_compare(self, q_target, q_actual):
        self.ax.clear()
        self._draw_axes()
        self._draw_arm(self.ax, q_target, '#a0a0a0', '#a0a0a0',
                       label='目标(滑块)', lw=3, alpha=0.55)
        pts_a = self._draw_arm(self.ax, q_actual, '#d62728', '#7e1416',
                               label='实测(回读)', lw=4)
        max_d = float(np.max(np.abs(np.asarray(q_actual) - np.asarray(q_target))))
        self.ax.legend(loc='upper left', fontsize=9)
        ee = pts_a[-1]
        self.ax.set_title(f'EXEC | 实测 TCP ({ee[0]:+6.1f},{ee[1]:+6.1f},{ee[2]:+6.1f}) | max|Δ|={max_d:.2f}°')
        self.canvas.draw_idle()

    # --- 串口连接 -----------------------------------------------------------
    def _on_connect(self):
        if self.arm is not None:
            try:
                self.arm.uart.close()
            except Exception:
                pass
            # 若正处在示教中,先安全退出示教
            if self.teach_active:
                self._exit_teach_mode(silent=True)
            self.arm = None
            self.btn_connect.setText('连接')
            self.btn_read.setEnabled(False)
            self.btn_send.setEnabled(False)
            self.btn_abort.setEnabled(False)
            for b in self._gripper_buttons:
                b.setEnabled(False)
            self.btn_teach_toggle.setEnabled(False)
            self.btn_teach_record.setEnabled(False)
            self._info('串口已断开。')
            self._set_status('已断开', 'idle')
            return

        port = self.port_edit.text().strip()
        try:
            baud = int(self.baud_edit.text().strip())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, '波特率错误', '波特率必须是整数')
            return

        self._set_status(f'正在连接 {port} ...', 'busy')
        self._info(f'尝试连接 {port} @ {baud} ...')
        QtWidgets.QApplication.processEvents()
        try:
            self.arm = robot.arm_robot(
                L_p=0, L_p_mass_center=0,
                MAX_list_temp=QMAX_DEG, MIN_list_temp=QMIN_DEG,
                G_p=0, com=port, uart_baudrate=baud,
            )
        except Exception as e:
            self._info(f'连接失败: {e}')
            self._set_status(f'连接失败: {e}', 'err')
            QtWidgets.QMessageBox.warning(self, '连接失败',
                                          f'{type(e).__name__}: {e}')
            self.arm = None
            return

        self._info(f'已连接 {port} @ {baud}')
        self._set_status(f'已连接 {port}', 'ok')
        self.btn_connect.setText('断开')
        self.btn_read.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.btn_abort.setEnabled(True)
        for b in self._gripper_buttons:
            b.setEnabled(True)
        self.btn_teach_toggle.setEnabled(True)
        # 记录按钮仅在示教模式中可用,这里先不启用

    # --- 读取关节角 ---------------------------------------------------------
    def _on_read(self):
        if self.arm is None:
            return
        self._set_status('读取关节中 ...', 'busy')
        QtWidgets.QApplication.processEvents()
        try:
            servo = self.arm.read_joints()
        except Exception as e:
            self._info(f'read_joints 异常: {e}')
            self._set_status('读取失败', 'err')
            return
        if servo is False:
            self._info('read_joints 返回 False')
            self._set_status('读取失败', 'err')
            return
        model = self.arm.servo_to_model(servo)
        self._info(f'回读模型角(°): {[round(v, 2) for v in model]}')
        self._set_q(model)
        self._set_status('已同步实机关节角', 'ok')

    # --- 发送指令 + 监控 ---------------------------------------------------
    def _on_send(self):
        if self.arm is None:
            return
        if self.exec_timer.isActive():
            QtWidgets.QMessageBox.information(self, '运行中',
                                              '已有运动在执行,请等待或紧急制动。')
            return

        target = self._current_q()
        # 限位
        if not all(QMIN_DEG[i] <= target[i] <= QMAX_DEG[i] for i in range(6)):
            QtWidgets.QMessageBox.warning(self, '越界', f'目标关节越界: {target}')
            return
        # 桌面保护
        ok, detail = self._check_desk_z(target)
        if not ok:
            QtWidgets.QMessageBox.warning(self, '桌面保护',
                                          f'存在关节坐标系 z < 0:\n{detail}\n请调整目标。')
            return

        # 确认对话框
        reply = QtWidgets.QMessageBox.question(
            self, '确认发送',
            f'目标关节(°):\n{[round(v, 2) for v in target.tolist()]}\n\n'
            f'到位阈值 {self.threshold_spin.value():.1f}°,'
            f'宽限 {self.grace_spin.value():.0f} s。\n确认发送?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        self.target_q_deg = target.copy()
        self._info(f'发送目标(°): {[round(v, 2) for v in target.tolist()]}')
        try:
            ok = self.arm.set_arm_joints(angle_list=target.tolist(), speed=1.0)
        except Exception as e:
            self._info(f'set_arm_joints 异常: {e}')
            self._set_status('指令异常', 'err')
            return
        if ok is False:
            self._info('set_arm_joints 返回 False(目标可能越界或参数有误)')
            self._set_status('指令拒绝', 'err')
            return

        # 进入执行态
        self.exec_start_time = time.monotonic()
        self._set_sliders_enabled(False)
        self.btn_send.setEnabled(False)
        self.btn_read.setEnabled(False)
        self._set_status('执行中 — 监控偏差 ...', 'busy')
        period_ms = max(50, int(self.poll_spin.value() * 1000))
        self.exec_timer.start(period_ms)

    def _on_exec_tick(self):
        if self.arm is None or self.target_q_deg is None:
            self.exec_timer.stop()
            return
        elapsed = time.monotonic() - self.exec_start_time
        try:
            servo = self.arm.read_joints()
        except Exception as e:
            self._info(f't={elapsed:.1f}s: 读取异常 {e}')
            return
        if servo is False:
            self._info(f't={elapsed:.1f}s: 读取失败,跳过本轮')
            return

        actual = np.array(self.arm.servo_to_model(servo), dtype=float)
        delta = actual - self.target_q_deg
        max_abs = float(np.max(np.abs(delta)))
        threshold = self.threshold_spin.value()
        grace = self.grace_spin.value()

        self._update_plot_compare(self.target_q_deg, actual)
        self._set_status(f'执行中  t={elapsed:.1f}s  max|Δ|={max_abs:.2f}°  '
                          f'(阈值 {threshold:.1f}°,宽限 {grace:.0f}s)', 'busy')

        if max_abs <= threshold:
            self._info(f't={elapsed:.1f}s: 收敛,max|Δ|={max_abs:.2f}°')
            self._finish_exec(ok=True, max_abs=max_abs, elapsed=elapsed)
            return

        if elapsed > grace:
            self._info(f't={elapsed:.1f}s: 超时未到位 max|Δ|={max_abs:.2f}° > {threshold:.1f}°,紧急制动')
            self._info(f'  Δ 各轴(°): {[round(v, 2) for v in delta.tolist()]}')
            self._do_lock()
            self._finish_exec(ok=False, max_abs=max_abs, elapsed=elapsed)

    def _finish_exec(self, ok: bool, max_abs: float, elapsed: float):
        self.exec_timer.stop()
        self._set_sliders_enabled(True)
        self.btn_send.setEnabled(True)
        self.btn_read.setEnabled(True)
        if ok:
            self._set_status(f'到位  max|Δ|={max_abs:.2f}°  耗时 {elapsed:.1f}s', 'ok')
        else:
            self._set_status(f'异常终止  max|Δ|={max_abs:.2f}°  已 lock', 'err')

    def _do_lock(self):
        if self.arm is None:
            return
        try:
            self.arm.lock()
            self._info('arm.lock() 已调用,所有关节锁定在当前位置')
        except Exception as e:
            self._info(f'arm.lock 异常: {e}')

    # --- 夹爪 ---------------------------------------------------------------
    def _grip_send(self, width_mm: float):
        """统一封装的夹爪下发逻辑(非阻塞)。

        绕开 `arm.grasp()` 中的公式 bug,直接对 ID=7 调用 `set_angle_adaptive`:
            angle_deg  = -wideth_mm * 18/π   (负角 = 张开)
            rpm        = speed/(d/2)/(2π)*60
            torque_Nm  = force*(d/2)/1000
        """
        if self.arm is None:
            return
        width_mm = max(GRIPPER_WIDTH_MIN_MM, min(GRIPPER_WIDTH_MAX_MM, float(width_mm)))
        speed = int(self.grip_speed_spin.value())
        force = int(self.grip_force_spin.value())
        angle_deg = width_mm * GRIPPER_DEG_PER_MM  # 负值张开
        r = GRIPPER_GEAR_D_MM / 2.0
        rpm = speed / r / (2 * np.pi) * 60
        torque_nm = force * r / 1000.0
        self._info(
            f'夹爪: width={width_mm:.1f}mm -> angle={angle_deg:+.2f}°  '
            f'(rpm={rpm:.1f}, τ={torque_nm:.3f}Nm)'
        )
        try:
            self.arm.set_angle_adaptive(
                id_num=GRIPPER_ID, angle=angle_deg,
                speed=rpm, torque=torque_nm,
            )
        except Exception as e:
            self._info(f'set_angle_adaptive 异常: {e}')
            self._set_status(f'夹爪指令异常: {e}', 'err')
            return
        # 同步滑块/数显到下发值
        self.grip_width_spin.blockSignals(True)
        self.grip_width_slider.blockSignals(True)
        self.grip_width_spin.setValue(width_mm)
        self.grip_width_slider.setValue(int(round(width_mm * 10)))
        self.grip_width_spin.blockSignals(False)
        self.grip_width_slider.blockSignals(False)

    def _on_grip_apply(self):
        self._grip_send(self.grip_width_spin.value())

    def _on_grip_open(self):
        self._grip_send(GRIPPER_OPEN_MM)

    def _on_grip_close(self):
        self._grip_send(GRIPPER_CLOSE_MM)

    def _on_grip_read(self):
        if self.arm is None:
            return
        try:
            w = self.arm.detect_wideth_grasp()
        except Exception as e:
            self._info(f'detect_wideth_grasp 异常: {e}')
            return
        if w is None:
            self._info('detect_wideth_grasp 返回 None')
            return
        # 显示在状态栏,且把滑块同步到实测宽度(便于继续微调)
        try:
            w_val = float(w)
        except (TypeError, ValueError):
            self._info(f'detect_wideth_grasp 返回非数值: {w!r}')
            return
        self._info(f'夹爪实测宽度: {w_val:+.2f} mm')
        clamped = max(GRIPPER_WIDTH_MIN_MM, min(GRIPPER_WIDTH_MAX_MM, abs(w_val)))
        self.grip_width_spin.blockSignals(True)
        self.grip_width_slider.blockSignals(True)
        self.grip_width_spin.setValue(clamped)
        self.grip_width_slider.setValue(int(round(clamped * 10)))
        self.grip_width_spin.blockSignals(False)
        self.grip_width_slider.blockSignals(False)

    # --- 示教模式 -----------------------------------------------------------
    def _enter_teach_mode(self) -> bool:
        """安全弹窗 + arm.free() + 启动周期读取。成功则返回 True。"""
        if self.arm is None:
            return False
        if self.exec_timer.isActive():
            QtWidgets.QMessageBox.information(
                self, '运行中', '当前正在执行运动指令,请先紧急制动再进入示教。',
            )
            return False

        reply = QtWidgets.QMessageBox.warning(
            self, '即将释放所有关节',
            '点击确定后所有关节(包括夹爪)将进入待机模式,可手动拖动。\n'
            '⚠ 释放后 J2/J3 会因自重塌下,请先用手托稳大臂!\n\n'
            '退出示教时会自动锁死在当前位置。\n\n是否继续?',
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if reply != QtWidgets.QMessageBox.Ok:
            return False

        try:
            self.arm.free()
        except Exception as e:
            self._info(f'arm.free() 异常: {e}')
            QtWidgets.QMessageBox.critical(self, '释放失败', f'{type(e).__name__}: {e}')
            return False

        self.teach_active = True
        self.btn_teach_toggle.setText('退出示教 (锁定)')
        self.btn_teach_toggle.setStyleSheet(
            'QPushButton { background-color: #d9534f; color: white; font-weight: bold; }'
        )
        self.btn_teach_record.setEnabled(True)
        # 禁用与运动/制动相关的按钮,防止示教中误下发
        self.btn_send.setEnabled(False)
        self.btn_read.setEnabled(False)
        for b in self._gripper_buttons:
            b.setEnabled(False)
        self._set_sliders_enabled(False)  # 示教时滑块仅作显示,禁止人工拖
        self._set_status('示教中 — 手动拖臂,完成后点"退出示教 (锁定)"', 'busy')
        self._info('已调用 arm.free();进入示教模式。')
        # 200 ms 周期读关节角,实时灌进滑块/3D
        self.teach_timer.start(200)
        return True

    def _exit_teach_mode(self, silent: bool = False) -> None:
        """停止周期读取 + arm.lock() 锁死;UI 复位。silent 时省略弹窗。"""
        if self.teach_timer.isActive():
            self.teach_timer.stop()
        if self.arm is not None:
            try:
                self.arm.lock()
                self._info('arm.lock() 已调用,关节锁死在当前位置。')
            except Exception as e:
                self._info(f'arm.lock 异常: {e}')
                if not silent:
                    QtWidgets.QMessageBox.warning(self, '锁定失败',
                                                  f'{type(e).__name__}: {e}')
        self.teach_active = False
        self.btn_teach_toggle.setText('进入示教')
        self.btn_teach_toggle.setStyleSheet(
            'QPushButton { background-color: #ffc107; font-weight: bold; }'
            'QPushButton:disabled { background-color: #d0d0d0; color: #888; font-weight: normal; }'
        )
        self.btn_teach_record.setEnabled(False)
        if self.arm is not None:
            self.btn_send.setEnabled(True)
            self.btn_read.setEnabled(True)
            for b in self._gripper_buttons:
                b.setEnabled(True)
        self._set_sliders_enabled(True)
        if not silent:
            self._set_status('已退出示教,关节锁定', 'ok')

    def _on_teach_toggle(self):
        if self.teach_active:
            self._exit_teach_mode()
        else:
            self._enter_teach_mode()

    def _on_teach_tick(self):
        """示教周期回调:读实测关节角 + 夹爪角,刷新滑块和 3D。"""
        if self.arm is None or not self.teach_active:
            return
        try:
            servo = self.arm.read_joints()
        except Exception as e:
            self._info(f'示教读取异常: {e}')
            return
        if servo is False:
            return
        model = self.arm.servo_to_model(servo)
        self.teach_last_joints_deg = [float(v) for v in model]
        try:
            g_angle = self.arm.get_angle(id_num=GRIPPER_ID)
            self.teach_last_gripper_deg = float(g_angle) if isinstance(g_angle, (int, float)) else None
        except Exception:
            self.teach_last_gripper_deg = None
        # 灌进滑块 + 重绘(_set_q 已经做了 block 信号 + 重绘)
        self._set_q(self.teach_last_joints_deg)

    def _on_teach_record(self):
        if not self.teach_active or self.teach_last_joints_deg is None:
            QtWidgets.QMessageBox.information(
                self, '尚未读到姿态',
                '请先进入示教并等待几个周期再记录。',
            )
            return
        rec = {
            'name': f'teach_{len(self.teach_records) + 1:03d}',
            't': time.strftime('%Y-%m-%d %H:%M:%S'),
            'joints_deg': list(self.teach_last_joints_deg),
            'gripper_deg': self.teach_last_gripper_deg,
        }
        self.teach_records.append(rec)
        joints_str = ', '.join(f'{v:+6.1f}' for v in rec['joints_deg'])
        g_str = f'  J7={rec["gripper_deg"]:+6.2f}°' if rec['gripper_deg'] is not None else ''
        item = QtWidgets.QListWidgetItem(
            f"{rec['name']}  [{joints_str}]°{g_str}"
        )
        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        item.setData(QtCore.Qt.UserRole, len(self.teach_records) - 1)
        self.teach_list.addItem(item)
        self._info(f'记录 {rec["name"]}: J1..J6={joints_str}{g_str}')

    def _on_teach_item_renamed(self, item: QtWidgets.QListWidgetItem):
        """双击修改条目时,把新文本视为名字(取空格前第一段;失败就忽略)"""
        idx = item.data(QtCore.Qt.UserRole)
        if not isinstance(idx, int) or not (0 <= idx < len(self.teach_records)):
            return
        new_text = item.text().strip()
        if not new_text:
            return
        new_name = new_text.split()[0]
        self.teach_records[idx]['name'] = new_name

    def _on_teach_apply(self):
        item = self.teach_list.currentItem()
        if item is None:
            return
        idx = item.data(QtCore.Qt.UserRole)
        if not isinstance(idx, int) or not (0 <= idx < len(self.teach_records)):
            return
        rec = self.teach_records[idx]
        # 示教中不应回灌滑块(会被 teach_tick 立刻覆盖);提示用户
        if self.teach_active:
            QtWidgets.QMessageBox.information(
                self, '示教进行中', '请先退出示教模式再"应用到滑块"。',
            )
            return
        self._set_q(rec['joints_deg'])
        self._info(f'已将记录 {rec["name"]} 应用到滑块(未下发)')

    def _on_teach_delete(self):
        item = self.teach_list.currentItem()
        if item is None:
            return
        idx = item.data(QtCore.Qt.UserRole)
        if not isinstance(idx, int) or not (0 <= idx < len(self.teach_records)):
            return
        del self.teach_records[idx]
        # 删后重建列表,保持 UserRole 与索引一致
        self._rebuild_teach_list()

    def _on_teach_clear(self):
        if not self.teach_records:
            return
        reply = QtWidgets.QMessageBox.question(
            self, '清空记录',
            f'确认清空全部 {len(self.teach_records)} 条记录?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        self.teach_records.clear()
        self.teach_list.clear()

    def _rebuild_teach_list(self):
        self.teach_list.blockSignals(True)
        self.teach_list.clear()
        for i, rec in enumerate(self.teach_records):
            joints_str = ', '.join(f'{v:+6.1f}' for v in rec['joints_deg'])
            g_str = (f'  J7={rec["gripper_deg"]:+6.2f}°'
                     if rec.get('gripper_deg') is not None else '')
            item = QtWidgets.QListWidgetItem(
                f"{rec['name']}  [{joints_str}]°{g_str}"
            )
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
            item.setData(QtCore.Qt.UserRole, i)
            self.teach_list.addItem(item)
        self.teach_list.blockSignals(False)

    def _on_teach_export(self):
        if not self.teach_records:
            QtWidgets.QMessageBox.information(self, '无记录', '当前列表为空。')
            return
        default_name = time.strftime('teach_%Y%m%d_%H%M%S.json')
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, '导出示教记录', default_name, 'JSON (*.json);;All Files (*)',
        )
        if not path:
            return
        try:
            import json
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.teach_records, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, '导出失败',
                                            f'{type(e).__name__}: {e}')
            return
        self._info(f'已导出 {len(self.teach_records)} 条记录到 {path}')

    def _on_abort(self):
        if self.arm is None:
            return
        self._do_lock()
        if self.exec_timer.isActive():
            self.exec_timer.stop()
        self._set_sliders_enabled(True)
        self.btn_send.setEnabled(True)
        self.btn_read.setEnabled(True)
        self._set_status('已手动制动,关节锁定', 'err')
        self._info('用户主动制动。')

    def closeEvent(self, event):
        if self.exec_timer.isActive():
            self.exec_timer.stop()
        if self.teach_active:
            # 关窗前必须确保关节锁死,避免 free 状态下窗口消失导致臂塌
            self._exit_teach_mode(silent=True)
        if self.arm is not None:
            try:
                self.arm.uart.close()
            except Exception:
                pass
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = ArmControlGUI()
    win.resize(1400, 820)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
