# 一. DH坐标
## 1. 转换原理

- **标准 DH**：连杆变换顺序为  
  \( \text{Rot}(z, \theta_i) \cdot \text{Trans}(z, d_i) \cdot \text{Trans}(x, a_i) \cdot \text{Rot}(x, \alpha_i) \)  
  参数：\( a_i, \alpha_i, d_i, \theta_i \)（关节变量 \( \theta_i \) 可能包含偏移量）

- **改进 DH**：连杆变换顺序为  
  \( \text{Rot}(x, \alpha_{i-1}) \cdot \text{Trans}(x, a_{i-1}) \cdot \text{Rot}(z, \theta_i) \cdot \text{Trans}(z, d_i) \)  
  参数：\( \alpha_{i-1}, a_{i-1}, \theta_i, d_i \)（关节变量 \( \theta_i \) 同样可含偏移）

**转换规则**（对于第 \( i \) 个关节，\( i = 1 \ldots n \)）：
- \( \alpha_{i-1} = \alpha_i^{\text{(std)}} \)
- \( a_{i-1} = a_i^{\text{(std)}} \)
- \( \theta_i = \theta_i^{\text{(std)}} \)（包括 offset）
- \( d_i = d_i^{\text{(std)}} \)

即：将标准 DH 中属于“连杆 \( i \)”的 \( \alpha_i, a_i \) 分配给改进 DH 中的“连杆 \( i-1 \)”，而 \( \theta_i, d_i \) 保持不变（下标不变）。

## 2. 标准 DH 参数整理


| 关节 i | \( a_i \) (m) | \( \alpha_i \) (rad) | \( d_i \) (m) | \( \theta_i \) (rad) |
|--------|--------------|----------------------|---------------|----------------------|
| 1      | 0            | \( \pi/2 \)           | 0             | \( q_1 \)            |
| 2      | 0.15         | 0                    | 0             | \( q_2 \)            |
| 3      | 0.15         | 0                    | 0             | \( q_3 \)            |
| 4      | 0            | \( \pi/2 \)           | -0.05494      | \( q_4 + \pi/2 \)     |
| 5      | 0            | \( -\pi/2 \)          | 0.068         | \( q_5 \)            |
| 6      | 0            | 0                    | 0.033         | \( q_6 \)            |

## 3. 转换后的改进 DH 参数

应用上述规则，得到改进 DH 参数表（\( i \) 表示关节编号，对应变换从坐标系 \( i-1 \) 到 \( i \)）：

| 关节 i | \( \alpha_{i-1} \) (rad) | \( a_{i-1} \) (m) | \( \theta_i \) (rad) | \( d_i \) (m) |
|--------|--------------------------|-------------------|----------------------|---------------|
| 1      | \( \pi/2 \)               | 0                 | \( q_1 \)            | 0             |
| 2      | 0                        | 0.15              | \( q_2 \)            | 0             |
| 3      | 0                        | 0.15              | \( q_3 \)            | 0             |
| 4      | \( \pi/2 \)               | 0                 | \( q_4 + \pi/2 \)     | -0.05494      |
| 5      | \( -\pi/2 \)              | 0                 | \( q_5 \)            | 0.068         |
| 6      | 0                        | 0                 | \( q_6 \)            | 0.033         |

**说明**：
- \( \alpha_0 = \pi/2, a_0 = 0 \) 表示基座坐标系（坐标系 0）相对于第一个关节坐标系（坐标系 1）有一个绕 x 轴的 \( 90^\circ \) 旋转，这是从标准 DH 直接继承来的，运动学上等价。
- 关节 4 的 \( \theta_4 = q_4 + \pi/2 \) 保留了标准 DH 中的偏移量。
- 所有长度单位均为米，角度单位为弧度。

## 4. 验证（可选）

您可以使用以下 Python 代码验证两种参数化是否给出相同的正向运动学结果（以一组关节角为例）：

```python
import numpy as np
from spatialmath import SE3
from roboticstoolbox import DHRobot, RevoluteDH, RevoluteMDH

# ---------- 标准 DH 模型 ----------
std_dh = DHRobot([
    RevoluteDH(alpha=np.pi/2),                     # 关节1
    RevoluteDH(a=0.15),                            # 关节2
    RevoluteDH(a=0.15),                            # 关节3
    RevoluteDH(d=-0.05494, alpha=np.pi/2, offset=np.pi/2),  # 关节4
    RevoluteDH(d=0.068, alpha=-np.pi/2),           # 关节5
    RevoluteDH(d=0.033)                            # 关节6
], name="DFbot_std")

# ---------- 改进 DH 模型 ----------
# 注意：RevoluteMDH 参数顺序为 (a, alpha, d, offset) 或使用关键字
mdh = DHRobot([
    RevoluteMDH(a=0, alpha=np.pi/2, d=0, offset=0),           # 关节1: a0, α0, d1, θ1
    RevoluteMDH(a=0.15, alpha=0, d=0, offset=0),              # 关节2
    RevoluteMDH(a=0.15, alpha=0, d=0, offset=0),              # 关节3
    RevoluteMDH(a=0, alpha=np.pi/2, d=-0.05494, offset=np.pi/2), # 关节4
    RevoluteMDH(a=0, alpha=-np.pi/2, d=0.068, offset=0),      # 关节5
    RevoluteMDH(a=0, alpha=0, d=0.033, offset=0)              # 关节6
], name="DFbot_mdh")

# 测试一组关节角
q_test = np.deg2rad([30, 45, 60, 90, 45, 30])
T_std = std_dh.fkine(q_test)
T_mdh = mdh.fkine(q_test)

print("标准 DH 末端位姿：\n", T_std)
print("改进 DH 末端位姿：\n", T_mdh)
print("位姿差异（应接近零）：\n", (T_std - T_mdh).norm())
```

运行结果应显示两个位姿矩阵完全一致（数值误差可忽略）。

## 5. 总结

您提供的标准 DH 参数可以唯一地转换为改进 DH 参数，转换结果如上表所示。若希望基座坐标系与第一个关节坐标系对齐（即使 \( \alpha_0 = 0, a_0 = 0 \)），可以通过重新定义基座坐标系实现，但这会改变关节 1 的零位和连杆偏移，通常不必要。直接使用上述转换后的参数即可在改进 DH 框架下进行运动学计算。
# 二、像素坐标到世界坐标转换
## 1. 坐标系定义

- **世界坐标系** \( \{W\} \)：固定参考系（例如机械臂基座坐标系）
- **相机坐标系** \( \{C\} \)：原点在相机光心，Z轴沿光轴方向
- **像素坐标系** \( (u,v) \)：图像左上角为原点，单位为像素

---

## 2. 相机模型与参数

### 2.1 内参矩阵 \( K \)

通过相机标定获得：

\[
K = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
\]

其中 \( f_x, f_y \) 为焦距（像素），\( (c_x, c_y) \) 为主点坐标。

### 2.2 外参矩阵 \( [R \mid t] \)

从世界坐标系到相机坐标系的变换：

\[
P_C = R \cdot P_W + t
\]

- \( R \)：3×3 旋转矩阵
- \( t \)：3×1 平移向量

或者用齐次变换矩阵：

\[
T_{C}^{W} = \begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix}, \quad P_C = T_{C}^{W} \cdot P_W
\]

---

## 3. 单目相机下从像素坐标到世界坐标

### 3.1 已知深度 \( Z_C \) 的情况

如果已知物体在相机坐标系下的深度 \( Z_C \)（例如深度相机或激光测距获得），则：

1. **像素坐标 → 归一化平面坐标**：
   \[
   \begin{bmatrix} x_n \\ y_n \\ 1 \end{bmatrix} = K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
   \]
2. **相机坐标系下的三维坐标**：
   \[
   \begin{bmatrix} X_C \\ Y_C \\ Z_C \end{bmatrix} = \begin{bmatrix} x_n Z_C \\ y_n Z_C \\ Z_C \end{bmatrix}
   \]
3. **世界坐标系下的三维坐标**：
   \[
   P_W = R^T (P_C - t) \quad \text{或} \quad P_W = (T_{C}^{W})^{-1} \cdot P_C
   \]

import numpy as np
import cv2

def pixel_to_world(u, v, Zc, K, R_w2c, t_w2c, dist_coeffs=None):
    """
    将像素坐标 (u,v) 和相机坐标系深度 Zc 转换为世界坐标系下的三维点。
    
    参数:
        u, v      : 像素坐标（浮点数）
        Zc        : 物体在相机坐标系下的深度 (米)
        K         : 相机内参矩阵 (3x3)
        R_w2c     : 世界坐标系到相机坐标系的旋转矩阵 (3x3)
        t_w2c     : 世界坐标系到相机坐标系的平移向量 (3x1)
        dist_coeffs: 畸变系数，可选 (k1,k2,p1,p2,k3,...)，若为 None 则不做畸变校正
    
    返回:
        P_world   : 世界坐标系下的三维点 (3x1) 单位与 Zc 相同
    """
    # 1. 畸变校正（如果需要）
    if dist_coeffs is not None:
        # 将像素坐标转为归一化平面坐标并去畸变
        # 注意：cv2.undistortPoints 需要输入 (N,1,2) 形状
        pts = np.array([[u, v]], dtype=np.float32).reshape(1, 1, 2)
        pts_undist = cv2.undistortPoints(pts, K, dist_coeffs, P=K)
        u_undist = pts_undist[0, 0, 0]
        v_undist = pts_undist[0, 0, 1]
    else:
        u_undist, v_undist = u, v

    # 2. 像素坐标转归一化平面坐标 (x_n, y_n)
    # 归一化坐标 = inv(K) * [u, v, 1]^T
    uv_hom = np.array([[u_undist], [v_undist], [1.0]], dtype=np.float64)
    xy_n = np.linalg.inv(K) @ uv_hom   # (3,1)
    x_n, y_n = xy_n[0, 0], xy_n[1, 0]

    # 3. 相机坐标系下的三维点 (Xc, Yc, Zc)
    Xc = x_n * Zc
    Yc = y_n * Zc
    P_cam = np.array([[Xc], [Yc], [Zc]], dtype=np.float64)

    # 4. 转换到世界坐标系
    # P_world = R_w2c^T * (P_cam - t_w2c)
    # 注意：t_w2c 是 3x1 列向量
    P_world = R_w2c.T @ (P_cam - t_w2c.reshape(3,1))

    return P_world.flatten()   # 返回一维数组 [Xw, Yw, Zw]

# ================== 示例用法 ==================
if __name__ == "__main__":
    # 示例：内参（假设已标定）
    K = np.array([[800.0, 0.0, 320.0],
                  [0.0, 800.0, 240.0],
                  [0.0, 0.0, 1.0]])

    # 外参：世界坐标系 -> 相机坐标系（例如从手眼标定或相机外参获得）
    # 这里假设世界坐标系与机械臂基座重合
    R_w2c = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])          # 示例：相机与世界坐标系方向相同
    t_w2c = np.array([[0.5], [0.2], [1.0]])  # 示例：相机光心在世界系下的坐标

    # 畸变系数（如果没有则设为 None）
    dist_coeffs = np.array([0.1, -0.05, 0.0, 0.0, 0.0])  # 示例

    # 待转换的像素坐标和深度
    u, v = 400, 300      # 像素坐标
    Zc = 1.2             # 深度（米）

    # 计算世界坐标
    P_world = pixel_to_world(u, v, Zc, K, R_w2c, t_w2c, dist_coeffs)
    print("世界坐标 (Xw, Yw, Zw):", P_world)

    # 如果已知的是相机到基座的变换 T_cam2base，可以转换为 R_w2c, t_w2c
    # T_cam2base = [[R_c2b, t_c2b], [0,0,0,1]]
    # 那么世界（基座）到相机的变换为 T_w2c = inv(T_cam2base)
    # 使用 np.linalg.inv 计算即可。

    以下是已知物体深度 \(Z_C\) 的情况下，将像素坐标转换为世界坐标系（如机械臂基座坐标系）的完整 Python 程序。

```python
import numpy as np
import cv2

def pixel_to_world(u, v, Zc, K, R_w2c, t_w2c, dist_coeffs=None):
    """
    将像素坐标 (u,v) 和相机坐标系深度 Zc 转换为世界坐标系下的三维点。
    
    参数:
        u, v      : 像素坐标（浮点数）
        Zc        : 物体在相机坐标系下的深度 (米)
        K         : 相机内参矩阵 (3x3)
        R_w2c     : 世界坐标系到相机坐标系的旋转矩阵 (3x3)
        t_w2c     : 世界坐标系到相机坐标系的平移向量 (3x1)
        dist_coeffs: 畸变系数，可选 (k1,k2,p1,p2,k3,...)，若为 None 则不做畸变校正
    
    返回:
        P_world   : 世界坐标系下的三维点 (3x1) 单位与 Zc 相同
    """
    # 1. 畸变校正（如果需要）
    if dist_coeffs is not None:
        # 将像素坐标转为归一化平面坐标并去畸变
        # 注意：cv2.undistortPoints 需要输入 (N,1,2) 形状
        pts = np.array([[u, v]], dtype=np.float32).reshape(1, 1, 2)
        pts_undist = cv2.undistortPoints(pts, K, dist_coeffs, P=K)
        u_undist = pts_undist[0, 0, 0]
        v_undist = pts_undist[0, 0, 1]
    else:
        u_undist, v_undist = u, v

    # 2. 像素坐标转归一化平面坐标 (x_n, y_n)
    # 归一化坐标 = inv(K) * [u, v, 1]^T
    uv_hom = np.array([[u_undist], [v_undist], [1.0]], dtype=np.float64)
    xy_n = np.linalg.inv(K) @ uv_hom   # (3,1)
    x_n, y_n = xy_n[0, 0], xy_n[1, 0]

    # 3. 相机坐标系下的三维点 (Xc, Yc, Zc)
    Xc = x_n * Zc
    Yc = y_n * Zc
    P_cam = np.array([[Xc], [Yc], [Zc]], dtype=np.float64)

    # 4. 转换到世界坐标系
    # P_world = R_w2c^T * (P_cam - t_w2c)
    # 注意：t_w2c 是 3x1 列向量
    P_world = R_w2c.T @ (P_cam - t_w2c.reshape(3,1))

    return P_world.flatten()   # 返回一维数组 [Xw, Yw, Zw]

# ================== 示例用法 ==================
if __name__ == "__main__":
    # 示例：内参（假设已标定）
    K = np.array([[800.0, 0.0, 320.0],
                  [0.0, 800.0, 240.0],
                  [0.0, 0.0, 1.0]])

    # 外参：世界坐标系 -> 相机坐标系（例如从手眼标定或相机外参获得）
    # 这里假设世界坐标系与机械臂基座重合
    R_w2c = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])          # 示例：相机与世界坐标系方向相同
    t_w2c = np.array([[0.5], [0.2], [1.0]])  # 示例：相机光心在世界系下的坐标

    # 畸变系数（如果没有则设为 None）
    dist_coeffs = np.array([0.1, -0.05, 0.0, 0.0, 0.0])  # 示例

    # 待转换的像素坐标和深度
    u, v = 400, 300      # 像素坐标
    Zc = 1.2             # 深度（米）

    # 计算世界坐标
    P_world = pixel_to_world(u, v, Zc, K, R_w2c, t_w2c, dist_coeffs)
    print("世界坐标 (Xw, Yw, Zw):", P_world)

    # 如果已知的是相机到基座的变换 T_cam2base，可以转换为 R_w2c, t_w2c
    # T_cam2base = [[R_c2b, t_c2b], [0,0,0,1]]
    # 那么世界（基座）到相机的变换为 T_w2c = inv(T_cam2base)
    # 使用 np.linalg.inv 计算即可。
```

## 关键说明

1. **输入数据**：
   - `u, v`：像素坐标（整数或浮点数）。
   - `Zc`：物体在相机坐标系下的深度，单位与平移向量一致（通常为米）。
   - `K`：相机内参矩阵，3×3。
   - `R_w2c, t_w2c`：世界坐标系到相机坐标系的旋转和平移。注意：`t_w2c` 是**相机光心在世界系下的位置**（负值）。
   - `dist_coeffs`：畸变系数，可选。若已知畸变系数，请传入。

2. **畸变校正**：
   - 使用 OpenCV 的 `cv2.undistortPoints` 对像素坐标进行去畸变。
   - 如果未提供畸变系数，则跳过此步。

3. **坐标转换流程**：
   - 像素 → 归一化平面 → 乘以深度得到相机坐标 → 通过外参转换到世界坐标。
   - 外参转换公式：\( P_{world} = R_{w2c}^T (P_{cam} - t_{w2c}) \)。

4. **外参来源**：
   - 若你已有手眼标定结果 \( X = T_{base}^{cam} \)（相机在基座下的位姿），则：
     \[
     T_{world}^{cam} = X^{-1}
     \]
     其中 \( R_{w2c} = (X^{-1})_{:3,:3} \)，\( t_{w2c} = (X^{-1})_{:3,3} \)。
   - 若直接使用相机标定得到的外参，则 `R_w2c` 和 `t_w2c` 即为标定输出。

5. **注意事项**：
   - 确保深度 \( Z_c \) 的单位与平移向量单位一致（通常米）。
   - 深度值必须为正数（相机前向）。
   - 对于实际应用，建议将像素坐标先进行畸变校正，再代入计算。

该函数可直接嵌入你的机器人视觉系统中，实现已知深度下的三维坐标定位。