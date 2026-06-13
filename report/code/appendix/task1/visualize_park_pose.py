"""对比 DFarm 的参数零位与安全竖直 park 姿态的几何关系。

输出:
  zero_vs_park_pose.png  左:q=[0,0,0,0,0,0] 水平全展(撞桌风险大)
                          右:q=[0,90,0,0,0,0] 竖直 park(安全)
  并在终端打印两组 q 下的 F1..F6 z 坐标
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from roboticstoolbox import DHRobot, RevoluteDH

# ---- 中文字体 ----
_CN = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'font', 'simhei.ttf'))
if os.path.exists(_CN):
    font_manager.fontManager.addfont(_CN)
    rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=_CN).get_name(), 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False

# ---- DH 模型 ----
LINK_PARAMS = (
    dict(alpha=np.pi / 2),
    dict(a=0.15),
    dict(a=0.15),
    dict(d=-0.05494, alpha=np.pi / 2, offset=np.pi / 2),
    dict(d=0.068, alpha=-np.pi / 2),
    dict(d=0.033),
)
dfarm = DHRobot([RevoluteDH(**kw) for kw in LINK_PARAMS], name='DFarm_StdDH')


def draw_pose(ax, q_deg, title):
    Ts = dfarm.fkine_all(np.deg2rad(q_deg))
    pts = np.asarray([T.t for T in Ts]) * 1000.0

    # 桌面平面(z = 0)
    xx, yy = np.meshgrid(np.linspace(-200, 450, 8), np.linspace(-250, 250, 8))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='#cfcfcf', alpha=0.20, edgecolor='none')

    # 连杆 + 关节
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
            '-', color='#4a90d9', lw=4, alpha=0.65, zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               color='#1f5fa5', s=70, depthshade=True, zorder=3)

    AX = 40.0
    for i, T in enumerate(Ts):
        o = T.t * 1000.0
        R = T.R
        for col, c in [(0, '#d62728'), (1, '#2ca02c'), (2, '#1f77b4')]:
            d = R[:, col] * AX
            ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2],
                      color=c, arrow_length_ratio=0.25, lw=2.0)
        ax.text(o[0] + 8, o[1] + 8, o[2] + 8, f'F{i}',
                fontsize=10, fontweight='bold', color='#222')

    # F6 z 坐标
    f6_z = pts[-1, 2]
    txt = (f'q = {q_deg} (deg)\n'
           f'F6 末端 = ({pts[-1,0]:+.1f}, {pts[-1,1]:+.1f}, {pts[-1,2]:+.1f}) mm\n'
           f'所有 frame 最低 z = {pts[:,2].min():+.1f} mm')
    ax.text2D(0.02, 0.97, txt, transform=ax.transAxes,
              fontsize=9, va='top',
              bbox=dict(boxstyle='round,pad=0.4',
                        facecolor='#f7f7f7', edgecolor='#bbb', alpha=0.95))

    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    ax.set_title(title, fontsize=12)

    # 统一比例
    ax.set_xlim(-200, 450)
    ax.set_ylim(-250, 250)
    ax.set_zlim(-50, 420)
    ax.view_init(elev=18, azim=-60)


fig = plt.figure(figsize=(16, 7))
ax1 = fig.add_subplot(121, projection='3d')
draw_pose(ax1, [0, 0, 0, 0, 0, 0], '参数零位 q = 0  ——  水平全展(贴桌面,撞桌风险)')

ax2 = fig.add_subplot(122, projection='3d')
draw_pose(ax2, [0, 90, 0, 0, 0, 0], '安全竖直 park  q = [0, 90, 0, 0, 0, 0]  ——  臂体朝上')

fig.suptitle('DFarm:参数零位 vs 安全竖直 park 姿态对比', fontsize=14, fontweight='bold')
fig.tight_layout()
out = 'zero_vs_park_pose.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print('saved', out)

for label, q_deg in [('零位', [0,0,0,0,0,0]), ('竖直 park', [0,90,0,0,0,0])]:
    Ts = dfarm.fkine_all(np.deg2rad(q_deg))
    zs = [round(float(T.t[2])*1000, 2) for T in Ts]
    print(f'{label:10s} F0..F6 z(mm) = {zs}')
