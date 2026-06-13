"""可视化 DFarm 在零位(q = [0,0,0,0,0,0])时各 DH 坐标系的位姿关系。

输出:
  - zero_pose_dh_visual.png  3D 图:连杆 + 每个 Frame i 的 (x,y,z) 三轴 + 参数标注
  - 终端打印每个 Frame 的原点坐标与零位末端 T06
"""
import os
import numpy as np
import matplotlib.pyplot as plt
# ---- 注册中文字体(daran/font/simhei.ttf)避免中文显示成方块 ----
from matplotlib import rcParams
from matplotlib import font_manager
_CN_FONT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'font', 'simhei.ttf'))
if os.path.exists(_CN_FONT):
    font_manager.fontManager.addfont(_CN_FONT)
    _name = font_manager.FontProperties(fname=_CN_FONT).get_name()
    rcParams['font.sans-serif'] = [_name, 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False

from roboticstoolbox import DHRobot, RevoluteDH

# ---- 当前标准 DH 表(与两个 notebook 保持一致) ----
LINK_PARAMS = (
    dict(alpha=np.pi / 2),
    dict(a=0.15),
    dict(a=0.15),
    dict(d=-0.05494, alpha=np.pi / 2, offset=np.pi / 2),
    dict(d=0.068, alpha=-np.pi / 2),
    dict(d=0.033),
)
DH_LABELS = [
    '$J_1$: a=0,    α=π/2,  d=0',
    '$J_2$: a=0.15, α=0,    d=0',
    '$J_3$: a=0.15, α=0,    d=0',
    '$J_4$: a=0,    α=π/2,  d=-0.05494,  θ-offset=π/2',
    '$J_5$: a=0,    α=-π/2, d=0.068',
    '$J_6$: a=0,    α=0,    d=0.033',
]

dfarm = DHRobot([RevoluteDH(**kw) for kw in LINK_PARAMS], name='DFarm_StdDH')
q_zero = np.zeros(6)

# fkine_all 返回每个连杆坐标系的 SE3,包括 base (索引 0)
fk_all = dfarm.fkine_all(q_zero)
origins_mm = np.asarray([T.t for T in fk_all]) * 1000.0

print('零位下各坐标系原点(mm):')
for i, o in enumerate(origins_mm):
    print(f'  Frame {i}: x={o[0]:+7.2f}  y={o[1]:+7.2f}  z={o[2]:+7.2f}')

print('\n零位下末端齐次变换矩阵 T06:')
np.set_printoptions(precision=4, suppress=True)
print(fk_all[-1].A)

# ---- 绘图 ----
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 连杆(原点之间用粗线相连)
ax.plot(origins_mm[:, 0], origins_mm[:, 1], origins_mm[:, 2],
        '-', color='#4a90d9', lw=4, alpha=0.55, zorder=1)
ax.scatter(origins_mm[:, 0], origins_mm[:, 1], origins_mm[:, 2],
           color='#1f5fa5', s=70, depthshade=True, zorder=3)

# 各 frame 三轴
AX_LEN = 40.0  # mm
for i, T in enumerate(fk_all):
    o = T.t * 1000.0
    R = T.R
    for col_idx, c, name in [(0, '#d62728', 'X'),
                              (1, '#2ca02c', 'Y'),
                              (2, '#1f77b4', 'Z')]:
        d = R[:, col_idx] * AX_LEN
        ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2],
                  color=c, arrow_length_ratio=0.25, lw=2.2)
    ax.text(o[0] + 8, o[1] + 8, o[2] + 8, f'F{i}',
            fontsize=11, fontweight='bold', color='#222')

# 在每段连杆中点标 a / d 长度(只挑非零的)
for i in range(6):
    p0 = origins_mm[i]
    p1 = origins_mm[i + 1]
    seg = p1 - p0
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1.0:
        continue
    mid = (p0 + p1) / 2.0
    ax.text(mid[0], mid[1], mid[2] + 5, f'{seg_len:.1f} mm',
            fontsize=9, color='#444', style='italic')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title(r'DFarm 零位(q = $\mathbf{0}$) DH 坐标系'
             '\nRed=X, Green=Y, Blue=Z, F0=base, F6=末端')

# DH 参数表(图右侧)
txt = '当前标准 DH 表\n' + '\n'.join(DH_LABELS)
ax.text2D(1.02, 0.50, txt, transform=ax.transAxes,
          fontsize=9, verticalalignment='center',
          bbox=dict(boxstyle='round,pad=0.5',
                    facecolor='#f7f7f7', edgecolor='#bbb', alpha=0.95))

# 等比缩放
mn = origins_mm.min(axis=0) - 60
mx = origins_mm.max(axis=0) + 60
ctr = (mn + mx) / 2.0
rng = (mx - mn).max() / 2.0
ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
ax.view_init(elev=24, azim=-72)

plt.tight_layout()
out = 'zero_pose_dh_visual.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f'\n已保存图像: {out}')
