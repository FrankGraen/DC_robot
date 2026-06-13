
import umatrix as um
import math as cm
import DrEmpower_can as Dr

RAD_DEG = 180 / cm.pi


class arm(Dr.DrEmpower_can):
    """一个六关节机械臂类，关节皆为转动副，专注运动模型
    全局坐标系Z轴与第一个关节轴线重合且竖直向上，X轴为第一关节轴线和初始姿态时的第二关节轴线的公垂线，并指向第二关节，Y轴由右手准则确定。

    Attributes:
        L: 机械臂杆件长度组成的列表；
        pl: 机械臂末端坐标系原点在全局坐标系下的坐标：[x, y, z]；
        theta: 六个关节角组成的列表，关节角以弧度制表示，关节角度相对于初始姿态逆时针为正，顺时针为负（正方向为使末端位置向上运动的方向）；
        theta_P_Y_R: 机械臂末端姿态角组成的列表[Pitch, Yaw, Roll]
    Methods:
        __init__: 对象初始化函数；
        set_pl: 末端点全局坐标pl赋值函数；
        inverse_kinematics：根据末端件坐标系位姿求解关节角度的函数（逆解）；
        forward_kinematics_joint_z_position: 根据关节角度求解末端点坐标系原点在全局坐标系中的z轴坐标（正解）；
        forward_kinematics_joint_x_position: 根据关节角度求解末端点坐标系原点在全局坐标系中的x轴坐标（正解）；
        forward_kinematics_pose: 根据关节角度求解末端点坐标系位姿函数（正解）；
        forward_kinematics_jacobi：机械臂雅克比矩阵
        forward_kinematics_torque：根据静力学模型计算关节力矩：静力学模型
    """
    L = [150, 150, 68, 54.94, 33] # 机械臂尺寸参数列表：[l1, l2, l3, d3, d4]，详见库函数说明
    l_p_mass_center = 55 # 末端件（负载/工具）质心到 6 号关节输出面的距离
    lp = 105 # 末端件（工具）中心到 6 号关节输出面的距离
    pl = [0, 0, 0]
    theta = [0, 0, 0, 0, 0, 0] # 6个关节角度
    theta_P_Y_R = [0, 0, 0] # 3个末端姿态角
    G = [0.15, 0.35, 0.15, 0.485, 0.227]  # 重量参数，单位kg，分别为杆件2、关节3、杆件3、关节4重量(两个电机)、负载重量(一个电机+实际负载)
    pl_camera = [140, 100, -18]  # pl_camera[0]摄像头坐标系原点相对5号关节轴线在6号关节轴线方向上移动的距离;
                              # pl_camera[1]摄像头坐标系原点相对6号关节轴线在5号关节轴线方向上移动的距离。
                              # pl_camera[2], hcamera摄像头坐标系原点相对6号关节轴线在6号关节坐标系初始状态下的y轴线方向上移动的距离

    def __init__(self, L_p=0, L_p_mass_center=0, G_p=0, com='', uart_baudrate=115200):
        """初始化函数。

        Args:
            L_p: 工具参考点到电机输出轴表面的距离，单位mm（所有尺寸参数皆为mm）
            L_p_mass_center: 工具（负载）质心到 6 号关节输出面的距离
            G_p: 负载重量，单位kg，所有重量单位皆为kg
        Returns:
            无。
        Raises:
            无。
        """
        Dr.DrEmpower_can.__init__(self, com=com, uart_baudrate=uart_baudrate)
        self.L[4] += L_p
        self.G.append(G_p)
        self.l_p_mass_center = L_p_mass_center
        self.lp = L_p
    
    def inverse_kinematics(self, pl_temp=[0, 0, 0], theta_P_Y_R=[0, 0, 0], ud=0):
        """根据末端件坐标系原点位置和姿态角求解关节角的函数，并以弧度制保存在theta列表中(运动学逆解)。

        Args:
            ud: (up/down)用来选择逆解中的第几种解：
                ud = 0 时(默认值)，机械臂第三个关节在末端点上方,对应课程中的alpha构型；
                ud = 1 时，机械臂第三个关节在末端点下方，对应课程中的beta构型.
        Returns:
            True: 求解过程顺利通过时，返回True；
            False: 当求解过程出现异常或者末端点超出工作空间时，返回False。
        Raises:
            无。
        """
        theta0_bk = self.theta[0] # 针对机械臂末端点在z轴上情况，设置一个记录前一次关节1角度的变量，使得关节1保持不动
        theta_bk = self.theta.copy()
        x = pl_temp[0] # - 1 # 消除模型坐标与实际坐标的误差
        y = pl_temp[1] # + 5 # 消除模型坐标与实际坐标的误差
        z = pl_temp[2]
        theta_p = theta_P_Y_R[0] / 180 * cm.pi # 求出seta234
        theta_Y = theta_P_Y_R[1] / 180 * cm.pi
        theta_R = theta_P_Y_R[2] / 180 * cm.pi
        l1 = self.L[0]
        l2 = self.L[1]
        l3 = self.L[2]
        d3 = self.L[3]
        d4 = self.L[4]
        if z == (l1 + l2 + l3) or z > (l1 + l2 + l3):
            self.theta[0] = 0
            self.theta[1] = cm.pi/2
            self.theta[2] = 0
            self.theta[3] = 0
            self.theta[4] = theta_Y
            self.theta[5] = theta_R
        else:
            A = -y
            B = x
            self.theta[4] = theta_Y  # 求出seta5
            self.theta[5] = theta_R  # 求出seta6
            C = d4 * cm.cos(self.theta[4]) - d3
            t1 = A ** 2 + B ** 2 - C ** 2  # 设一个中间变量，以减少程序运算量
            if t1 >= 0:
                if A == 0 and B == 0 and C == 0:
                    self.theta[0] = theta0_bk # 此时机械臂末端点在z轴上，关节1取前一次的值（即不变）
                else:
                    if A + C == 0:
                        t = - (A - C) / (2 * B)  # 此时不存在B==0的情况
                    else:
                        t = (B - cm.sqrt(t1)) / (A + C)
                        # t = (B + cm.sqrt(t1)) / (A + C) # 经实验选取该值会导致后续角度无解
                    self.theta[0] = 2 * cm.atan(t)  # 求出seta1
                R1 = x - d4 * (cm.cos(self.theta[5]) * cm.sin(self.theta[0]) + cm.sin(self.theta[4]) * cm.cos(
                    self.theta[0]) * cm.sin(theta_p)) + d3 * cm.sin(self.theta[0]) - l3 * cm.cos(theta_p) * cm.cos(
                    self.theta[0])
                R2 = y + d4 * (cm.cos(self.theta[5]) * cm.cos(self.theta[0]) - cm.sin(self.theta[4]) * cm.sin(
                    self.theta[0]) * cm.sin(theta_p)) - d3 * cm.cos(self.theta[0]) - l3 * cm.cos(theta_p) * cm.sin(
                    self.theta[0])
                R3 = R1 * cm.cos(self.theta[0]) + R2 * cm.sin(self.theta[0])
                R4 = z + d4 * cm.sin(self.theta[4]) * cm.cos(theta_p) - l3 * cm.sin(theta_p)
                R5 = (R3 ** 2 + R4 ** 2 + l1 ** 2 - l2 ** 2) / (2 * l1)
                t2 = R3 ** 2 + R4 ** 2 - R5 ** 2  # 设一个中间变量，以减少程序运算量
                if t2 >= 0:
                    if (R3 + R5) == 0:
                        k = - (R3 - R5) / (2 * R4) # 此时不存在R4==0的情况
                    else:
                        if ud == 0:
                            k = (R4 + cm.sqrt(t2)) / (R3 + R5)  # alpha构型，第三个关节在末端上方
                        else:
                            k = (R4 - cm.sqrt(t2)) / (R3 + R5)  # beta构型，第三个关节在末端点下方
                    theta1 = 2 * cm.atan(k)
                    if theta1 < - cm.pi / 2:
                        self.theta[1] = 2 * cm.pi + theta1 # 求出seta2，atan()函数取值范围[-90, 90]，但因安装在桌面或地面上，关节2的不可以顺时针转超过90度，对应位置由逆时针代替
                    else:
                        self.theta[1] = theta1
                    R6 = (R3 - l1 * cm.cos(self.theta[1])) / l2
                    R7 = (R4 - l1 * cm.sin(self.theta[1])) / l2
                    seta23 = cm.atan2(R7, R6)
                    self.theta[2] = seta23 - self.theta[1]
                    self.theta[3] = theta_p - seta23
                    return True
                else:
                    print('末端位姿超出工作空间 (seta2)!')
                    self.theta = theta_bk.copy()
                    return False
            else:
                print('末端位姿超出工作空间 (seta1)!')
                self.theta = theta_bk.copy()
                return False
        return True

    def forward_kinematics_joint_z_postion(self, angle_list=[0, 90, -90, -90, 0, 0]):
        """根据当前关节角度计算4个运动关节的z轴坐标

                Args:
                    angle_list: 关节角度组成的列表。
                Returns:
                    运动关节的z坐标组成的列表。
                Raises:
                    无
                """
        angle = [cm.pi / 180 * i for i in angle_list]
        l1 = self.L[0]
        l2 = self.L[1]
        l3 = self.L[2]
        seta2 = angle[1]
        seta3 = angle[2]
        seta4 = angle[3]
        seta23 = seta2 + seta3
        seta234 = seta23 + seta4
        z3= l1 * cm.sin(seta2) # 第三个关节的z坐标
        z45 = l1 * cm.sin(seta2)+ l2 * cm.sin(seta23) # 第四个和第五个关节的z坐标
        #z45 = z3 + l2 * cm.sin(seta23) # 第四个和第五个关节的z坐标
        z6 = l1 * cm.sin(seta2)+ l2 * cm.sin(seta23) + l3 * cm.sin(seta234) # 第六个关节的z坐标
        return [z3, z45, z45, z6]

    def forward_kinematics_joint_x_postion(self, angle_list=[0, 90, -90, -90, 0, 0]):
        """根据当前关节角度计算4个运动关节在机械臂平面中的x坐标

                Args:
                    angle_list: 关节角度组成的列表。
                Returns:
                    运动关节的z坐标组成的列表。
                Raises:
                    无
                """
        angle = [cm.pi / 180 * i for i in angle_list]
        l1 = self.L[0]
        l2 = self.L[1]
        l3 = self.L[2]
        seta2 = angle[1]
        seta3 = angle[2]
        seta4 = angle[3]
        seta23 = seta2 + seta3
        seta234 = seta23 + seta4
        x3= l1 * cm.cos(seta2) # 第三个关节的x坐标
        x45 = x3 + l2 * cm.cos(seta23) #第四个和第五个关节的x坐标
        x6 = x45 + l3 * cm.cos(seta234) # 第六个关节的x坐标
        return [x3, x45, x45, x6]

    def forward_kinematics_pose(self, angle_list=[0, 90, -90, -90, 0, 0]):
        """根据六个关节角求解末端点位置和姿态函数(运动学正解)

        Args:
            angle_list: 六个关节角度组成的列表[joint1,joint2,joint3, joint4, joint5, joint6]（角度制）
        Returns:
            无
        Raises:
            无
        """

        angle = [cm.pi / 180 * i for i in angle_list]
        seta1 = angle[0]
        seta2 = angle[1]
        seta5 = angle[4]
        seta6 = angle[5]
        seta23 = angle[1] + angle[2]
        seta234 = angle[1] + angle[2] + angle[3]
        l1 = self.L[0]
        l2 = self.L[1]
        l3 = self.L[2]
        d3 = self.L[3]
        d4 = self.L[4]
        x = d4 * (cm.cos(seta5) * cm.sin(seta1) + cm.sin(seta5) * cm.cos(seta1) * cm.sin(seta234)) - d3 * cm.sin(
            seta1) + l3 * cm.cos(seta1) * cm.cos(seta234) + l1 * cm.cos(seta1) * (cm.cos(seta2) + cm.cos(seta23))
        y = -d4 * (cm.cos(seta5) * cm.cos(seta1) - cm.sin(seta5) * cm.sin(seta1) * cm.sin(seta234)) + d3 * cm.cos(
            seta1) + l3 * cm.sin(seta1) * cm.cos(seta234) + l1 * cm.sin(seta1) * (cm.cos(seta2) + cm.cos(seta23))
        z = -d4 * cm.sin(seta5) * cm.cos(seta234) + l1 * cm.sin(seta2) + l2 * cm.sin(seta23) + l3 * cm.sin(seta234)
        self.pl = [x, y, z].copy()
        self.theta_P_Y_R = [seta234 * RAD_DEG , seta5 *RAD_DEG , seta6 * RAD_DEG ].copy()

    def forward_kinematics_jacobi(self, angle_list=[0, 90, -90, -90, 0, 0]):
        """根据六个关节角求解机械臂当前位置的雅克比矩阵

                Args:
                    angle_list: 六个关节角度组成的列表[joint1,joint2,joint3, joint4, joint5, joint6]（角度制）。
                Returns:
                    雅克比矩阵Jacobi。
                Raises:
                    无
                """
        angle = [cm.pi / 180 * i for i in angle_list]
        seta1 = angle[0]
        seta2 = angle[1]
        seta3 = angle[2]
        seta4 = angle[3]
        seta5 = angle[4]
        seta6 = angle[5]
        l1 = self.L[0]
        l2 = self.L[1]
        l3 = self.L[2]
        d3 = self.L[3]
        Jr1 = [0, cm.sin(seta1), cm.sin(seta1), cm.sin(seta1), - cm.cos(seta4) * (
                    cm.cos(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta1) * cm.cos(seta2) * cm.cos(
                seta3)) - cm.sin(seta4) * (
                           cm.cos(seta1) * cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta1) * cm.cos(seta3) * cm.sin(
                       seta2)), cm.cos(seta5) * cm.sin(seta1) + cm.sin(seta5) * (cm.cos(seta4) * (
                    cm.cos(seta1) * cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta1) * cm.cos(seta3) * cm.sin(
                seta2)) - cm.sin(seta4) * (cm.cos(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta1) * cm.cos(
            seta2) * cm.cos(seta3)))]
        Jr2 = [0, -cm.cos(seta1), -cm.cos(seta1), -cm.cos(seta1), - cm.cos(seta4) * (
                    cm.sin(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta2) * cm.cos(seta3) * cm.sin(
                seta1)) - cm.sin(seta4) * (
                           cm.cos(seta2) * cm.sin(seta1) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta1) * cm.sin(
                       seta2)), cm.sin(seta5) * (cm.cos(seta4) * (
                    cm.cos(seta2) * cm.sin(seta1) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta1) * cm.sin(
                seta2)) - cm.sin(seta4) * (cm.sin(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta2) * cm.cos(
            seta3) * cm.sin(seta1))) - cm.cos(seta1) * cm.cos(seta5)]
        Jr3 = [1, 0, 0, 0,
               cm.cos(seta4) * (cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta2)) + cm.sin(seta4) * (
                           cm.cos(seta2) * cm.cos(seta3) - cm.sin(seta2) * cm.sin(seta3)), -cm.sin(seta5) * (
                           cm.cos(seta4) * (cm.cos(seta2) * cm.cos(seta3) - cm.sin(seta2) * cm.sin(seta3)) - cm.sin(
                       seta4) * (cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta2)))]
        Jr4 = [0, 0, l1 * cm.cos(seta1) * cm.sin(seta2), cm.cos(seta1) * (
                    l2 * (cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta2)) + l1 * cm.sin(seta2)), (
                           cm.cos(seta4) * (cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta2)) + cm.sin(
                       seta4) * (cm.cos(seta2) * cm.cos(seta3) - cm.sin(seta2) * cm.sin(seta3))) * (
                           d3 * cm.cos(seta1) - l2 * (
                               cm.sin(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta2) * cm.cos(seta3) * cm.sin(
                           seta1)) + l1 * cm.cos(seta2) * cm.sin(seta1)) + (
                           l2 * (cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta2)) + l1 * cm.sin(
                       seta2)) * (cm.cos(seta4) * (
                    cm.sin(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta2) * cm.cos(seta3) * cm.sin(
                seta1)) + cm.sin(seta4) * (cm.cos(seta2) * cm.sin(seta1) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(
            seta1) * cm.sin(seta2))), (cm.cos(seta1) * cm.cos(seta5) - cm.sin(seta5) * (cm.cos(seta4) * (
                    cm.cos(seta2) * cm.sin(seta1) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta1) * cm.sin(
                seta2)) - cm.sin(seta4) * (cm.sin(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta2) * cm.cos(
            seta3) * cm.sin(seta1)))) * (
                           l2 * (cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta2)) + l1 * cm.sin(
                       seta2) + l3 * (cm.cos(seta4) * (
                               cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta2)) + cm.sin(seta4) * (
                                                  cm.cos(seta2) * cm.cos(seta3) - cm.sin(seta2) * cm.sin(
                                              seta3)))) + cm.sin(seta5) * (
                           cm.cos(seta4) * (cm.cos(seta2) * cm.cos(seta3) - cm.sin(seta2) * cm.sin(seta3)) - cm.sin(
                       seta4) * (cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta2))) * (l2 * (
                    cm.sin(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta2) * cm.cos(seta3) * cm.sin(
                seta1)) + l3 * (cm.cos(seta4) * (
                    cm.sin(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta2) * cm.cos(seta3) * cm.sin(
                seta1)) + cm.sin(seta4) * (cm.cos(seta2) * cm.sin(seta1) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(
            seta1) * cm.sin(seta2))) - d3 * cm.cos(seta1) - l1 * cm.cos(seta2) * cm.sin(seta1))]
        Jr5 = [0, 0, l1 * cm.sin(seta1) * cm.sin(seta2), cm.sin(seta1) * (
                    l2 * (cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta2)) + l1 * cm.sin(seta2)), (
                           cm.cos(seta4) * (cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta2)) + cm.sin(
                       seta4) * (cm.cos(seta2) * cm.cos(seta3) - cm.sin(seta2) * cm.sin(seta3))) * (l2 * (
                    cm.cos(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta1) * cm.cos(seta2) * cm.cos(
                seta3)) + d3 * cm.sin(seta1) - l1 * cm.cos(seta1) * cm.cos(seta2)) - (cm.cos(seta4) * (
                    cm.cos(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta1) * cm.cos(seta2) * cm.cos(
                seta3)) + cm.sin(seta4) * (cm.cos(seta1) * cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta1) * cm.cos(
            seta3) * cm.sin(seta2))) * (
                           l2 * (cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta2)) + l1 * cm.sin(seta2)), (
                           cm.cos(seta5) * cm.sin(seta1) + cm.sin(seta5) * (cm.cos(seta4) * (
                               cm.cos(seta1) * cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta1) * cm.cos(seta3) * cm.sin(
                           seta2)) - cm.sin(seta4) * (cm.cos(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(
                       seta1) * cm.cos(seta2) * cm.cos(seta3)))) * (
                           l2 * (cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta2)) + l1 * cm.sin(
                       seta2) + l3 * (cm.cos(seta4) * (
                               cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta2)) + cm.sin(seta4) * (
                                                  cm.cos(seta2) * cm.cos(seta3) - cm.sin(seta2) * cm.sin(
                                              seta3)))) - cm.sin(seta5) * (
                           cm.cos(seta4) * (cm.cos(seta2) * cm.cos(seta3) - cm.sin(seta2) * cm.sin(seta3)) - cm.sin(
                       seta4) * (cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta2))) * (l2 * (
                    cm.cos(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta1) * cm.cos(seta2) * cm.cos(
                seta3)) + l3 * (cm.cos(seta4) * (
                    cm.cos(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta1) * cm.cos(seta2) * cm.cos(
                seta3)) + cm.sin(seta4) * (cm.cos(seta1) * cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta1) * cm.cos(
            seta3) * cm.sin(seta2))) + d3 * cm.sin(seta1) - l1 * cm.cos(seta1) * cm.cos(seta2))]
        Jr6 = [0, 0, - l1 * cm.cos(seta2) * cm.cos(seta1) ** 2 - l1 * cm.cos(seta2) * cm.sin(seta1) ** 2,
               cm.cos(seta1) * (l2 * (
                           cm.cos(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta1) * cm.cos(seta2) * cm.cos(
                       seta3)) - l1 * cm.cos(seta1) * cm.cos(seta2)) + cm.sin(seta1) * (l2 * (
                           cm.sin(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta2) * cm.cos(seta3) * cm.sin(
                       seta1)) - l1 * cm.cos(seta2) * cm.sin(seta1)), (cm.cos(seta4) * (
                        cm.cos(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta1) * cm.cos(seta2) * cm.cos(
                    seta3)) + cm.sin(seta4) * (cm.cos(seta1) * cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta1) * cm.cos(
                seta3) * cm.sin(seta2))) * (d3 * cm.cos(seta1) - l2 * (
                        cm.sin(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta2) * cm.cos(seta3) * cm.sin(
                    seta1)) + l1 * cm.cos(seta2) * cm.sin(seta1)) + (cm.cos(seta4) * (
                        cm.sin(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta2) * cm.cos(seta3) * cm.sin(
                    seta1)) + cm.sin(seta4) * (cm.cos(seta2) * cm.sin(seta1) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(
                seta1) * cm.sin(seta2))) * (l2 * (
                        cm.cos(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta1) * cm.cos(seta2) * cm.cos(
                    seta3)) + d3 * cm.sin(seta1) - l1 * cm.cos(seta1) * cm.cos(seta2)), (
                           cm.cos(seta1) * cm.cos(seta5) - cm.sin(seta5) * (cm.cos(seta4) * (
                               cm.cos(seta2) * cm.sin(seta1) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(seta1) * cm.sin(
                           seta2)) - cm.sin(seta4) * (cm.sin(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(
                       seta2) * cm.cos(seta3) * cm.sin(seta1)))) * (l2 * (
                        cm.cos(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta1) * cm.cos(seta2) * cm.cos(
                    seta3)) + l3 * (cm.cos(seta4) * (
                        cm.cos(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta1) * cm.cos(seta2) * cm.cos(
                    seta3)) + cm.sin(seta4) * (cm.cos(seta1) * cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta1) * cm.cos(
                seta3) * cm.sin(seta2))) + d3 * cm.sin(seta1) - l1 * cm.cos(seta1) * cm.cos(seta2)) + (
                           cm.cos(seta5) * cm.sin(seta1) + cm.sin(seta5) * (cm.cos(seta4) * (
                               cm.cos(seta1) * cm.cos(seta2) * cm.sin(seta3) + cm.cos(seta1) * cm.cos(seta3) * cm.sin(
                           seta2)) - cm.sin(seta4) * (cm.cos(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(
                       seta1) * cm.cos(seta2) * cm.cos(seta3)))) * (l2 * (
                        cm.sin(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta2) * cm.cos(seta3) * cm.sin(
                    seta1)) + l3 * (cm.cos(seta4) * (
                        cm.sin(seta1) * cm.sin(seta2) * cm.sin(seta3) - cm.cos(seta2) * cm.cos(seta3) * cm.sin(
                    seta1)) + cm.sin(seta4) * (cm.cos(seta2) * cm.sin(seta1) * cm.sin(seta3) + cm.cos(seta3) * cm.sin(
                seta1) * cm.sin(seta2))) - d3 * cm.cos(seta1) - l1 * cm.cos(seta2) * cm.sin(seta1))]
        Jacobi = um.matrix([Jr1, Jr2, Jr3, Jr4, Jr5, Jr6])
        return Jacobi

    def forward_kinematics_torque(self, angle_list=[0, 90, -90, -90, 0, 0], pay_load=0, F=[0, 0, 0]):
        """根据六个关节角求解保持机械臂当前位姿的前5个关节力矩（静力学）

        Args:
            angle_list: 六个关节角度组成的列表[joint1,joint2,joint3, joint4, joint5, joint6]（角度制）。
            pay_load: 负载重量，单位kg
            F: 末端在全局坐标系坐标轴方向上的受力，单位kg
        Returns:
            前5个关节力矩组成的列表。
        Raises:
            无。
        """
        angle = [cm.pi / 180 * i for i in angle_list]
        l1 = self.L[0]
        l2 = self.L[1]
        l3 = self.L[2]
        seta2 = angle[1]
        seta3 = angle[2]
        seta4 = angle[3]
        seta5 = angle[4]
        seta23 = seta2 + seta3
        seta234 = seta23 + seta4
        self.L[4] -= (self.lp - self.l_p_mass_center) # 减去的数值是机械爪质心到手爪中部的距离
        d4 = self.L[4]
        F_x = F[0] * 10
        F_y = F[1] * 10
        F_z = F[2] * 10
        G_link1 = self.G[0] * 10
        G3 = self.G[1] * 10
        G_link2 = self.G[2] * 10
        G45 = self.G[3] * 10
        G6 = self.G[4] * 10
        G_p = (self.G[5] + pay_load) * 10
        s1 = l1 * cm.cos(seta2)
        s2 = l2 * cm.cos(seta23)
        s3 = l3 * cm.cos(seta234)
        z_joints = self.forward_kinematics_joint_z_postion(angle_list)
        z3 = z_joints[0]
        z4 = z_joints[1]
        self.forward_kinematics_pose(angle_list)
        self.L[4] += (self.lp - self.l_p_mass_center) # 恢复减去的数值是机械爪质心到手爪中部的距离
        x = self.pl[0]
        y = self.pl[1]
        z = self.pl[2]
        if x >= 0:
            xx = cm.sqrt(x ** 2 + y ** 2)
        if x < 0:
            xx = - cm.sqrt(x ** 2 + y ** 2)
        M_arm = 0.5 * G_link1 * s1 + G3 * s1 + G_link2 * (s1 + 0.5 * s2) + G45 * (s1 + s2) + G6 * (s1 + s2 + s3) + G_p * xx  # 关节与末端点不在同一平面内，这里默认在同一平面内，以后再优化
        M2 = M_arm - F_z * xx - F_x * z
        M3 = 0.5 * G_link2 * s2 + G45 * s2 + G6 * (s1 + s2 + s3) + (G_p - F_z) * (xx - s1) - F_x * (z - z3)
        M4 = G6 * s3 + (G_p - F_z) * (xx - s1 - s2) - F_x * (z - z4)
        if seta234 <= cm.pi / 2:
            M5 = (F_z - G_p) * d4 * cm.cos(seta5) + F_y * d4 * cm.sin(seta5)
        if seta234 > cm.pi / 2:
            M5 = (G_p - F_z) * d4 * cm.cos(seta5) - F_y * d4 * cm.sin(seta5)
        M1 = F_y * xx
        return [M1/1000, M2/1000, M3/1000, M4/1000, M5/1000]

    def transfer_camera_to_arm(self, pl_tem_camera=[0, 0, 0]):
        """摄像头坐标系向全局坐标系变换函数

        Args:
            pl_tem_camera: 目标在摄像头坐标系中的坐标值[x_c,y_c,z_c]（mm）。此时要求摄像头坐标系与6号关节坐标系平行
        Returns:
            目标在q全局坐标系中的坐标值[x,y,z]（mm）。此时要求摄像头坐标系与6号关节坐标系平行
        Raises:
            无。
        """

        dcamera = self.pl_camera[0]
        lcamera = self.pl_camera[1]
        hcamera = self.pl_camera[2]
        theta_1 = self.theta[0]
        theta_2 = self.theta[1]
        theta_3 = self.theta[2]
        theta_4 = self.theta[3]
        theta_5 = self.theta[4]
        l1 = self.L[0]
        l2 = self.L[1]
        l3 = self.L[2]
        d3 = self.L[3]
        theta_23 = theta_2 + theta_3
        theta_234 = theta_23 + theta_4
        m11 = cm.cos(theta_1) * cm.cos(theta_234)
        m21 = cm.sin(theta_1) * cm.cos(theta_234)
        m31 = cm.sin(theta_234)
        m12 = cm.sin(theta_1) * cm.sin(theta_5) - cm.cos(theta_5) * cm.cos(theta_1) * cm.sin(theta_234)
        m22 = - cm.cos(theta_1) * cm.sin(theta_5) - cm.cos(theta_5) * cm.sin(theta_1) * cm.sin(theta_234)
        m32 = cm.cos(theta_5) * cm.cos(theta_234)
        m13 = cm.cos(theta_5) * cm.sin(theta_1) + cm.sin(theta_5) * cm.cos(theta_1) * cm.sin(theta_234)
        m23 = - cm.cos(theta_1) * cm.cos(theta_5) + cm.sin(theta_5) * cm.sin(theta_1) * cm.sin(theta_234)
        m33 = -cm.sin(theta_5) * cm.cos(theta_234)
        delta_x = l1 * cm.cos(theta_1) * cm.cos(theta_2) + l2 * cm.cos(theta_1) * cm.cos(theta_23) + l3 * cm.cos(theta_1) * cm.cos(
            theta_4 + theta_23) + lcamera * cm.cos(theta_1) * cm.cos(theta_4 + theta_23) - d3 * cm.sin(theta_1) + dcamera * (
                              cm.cos(theta_5) * cm.sin(theta_1) + cm.sin(theta_5) * cm.cos(theta_1) * cm.sin(theta_4 + theta_23)) + hcamera * (
                              cm.sin(theta_1) * cm.sin(theta_5) - cm.cos(theta_5) * cm.cos(theta_1) * cm.sin(theta_234))
        delta_y = l1 * cm.cos(theta_2) * cm.sin(theta_1) + l2 * cm.sin(theta_1) * cm.cos(theta_23) + l3 * cm.sin(theta_1) * cm.cos(
            theta_4 + theta_23) + lcamera * cm.sin(theta_1) * cm.cos(theta_4 + theta_23) + d3 * cm.cos(theta_1) - dcamera * (
                              cm.cos(theta_1) * cm.cos(theta_5) - cm.sin(theta_5) * cm.sin(theta_1) * cm.sin(theta_4 + theta_23)) - hcamera * (
                              cm.cos(theta_1) * cm.sin(theta_5) + cm.cos(theta_5) * cm.sin(theta_1) * cm.sin(theta_234))
        delta_z = l1 * cm.sin(theta_2) + l2 * cm.sin(theta_23) + l3 * cm.sin(theta_234) + lcamera * cm.sin(theta_234) - dcamera * cm.sin(
            theta_5) * cm.cos(theta_234) + hcamera * cm.cos(theta_5) * cm.cos(theta_234)
        # delta_x = l1 * cm.cos(theta_1) * cm.cos(theta_2) + l2 * cm.cos(theta_1) * cm.cos(theta_23) + l3 * cm.cos(theta_1) * cm.cos(
        #     theta_234) + lcamera * cm.cos(theta_1) * cm.cos(theta_234) - d3 * cm.sin(theta_1) + dcamera * (
        #                   cm.cos(theta_5) * cm.sin(theta_1) + cm.sin(theta_5) * cm.cos(theta_1) * cm.sin(theta_234))
        # delta_y = l1 * cm.cos(theta_2) * cm.sin(theta_1) + l2 * cm.sin(theta_1) * cm.cos(theta_23) + l3 * cm.sin(theta_1) * cm.cos(
        #     theta_234) + lcamera * cm.sin(theta_1) * cm.cos(theta_234) + d3 * cm.cos(theta_1) - dcamera * (
        #                   cm.cos(theta_1) * cm.cos(theta_5) - cm.sin(theta_5) * cm.sin(theta_1) * cm.sin(theta_234))
        # delta_z = l1 * cm.sin(theta_2) + l2 * cm.sin(theta_23) + l3 * cm.sin(theta_234) + lcamera * cm.sin(
        #     theta_234) - dcamera * cm.sin(theta_5) * cm.cos(theta_234)
        x = m11 * pl_tem_camera[0] + m12 * pl_tem_camera[1] + m13 * pl_tem_camera[2] + delta_x
        y = m21 * pl_tem_camera[0] + m22 * pl_tem_camera[1] + m23 * pl_tem_camera[2] + delta_y
        z = m31 * pl_tem_camera[0] + m32 * pl_tem_camera[1] + m33 * pl_tem_camera[2] + delta_z
        # print([x, y, z])
        return [x, y, z]



