#!/usr/bin/python3
import gripper
import math as cm
import time

# 角度和弧度转换
DEG_RAD = cm.pi / 180
RAD_DEG = 180 / cm.pi
##钢结构原程序设置
# max_list_temp = [85, 215, 149, 142, 179, 179]  # 关节模型角度最大值,1号关节目的是保护线缆，并且到达工作空间边缘；2号关节到达工作空间边缘；3、4号关节目的是不产生杆件干涉；5号关节因有滑环，不考虑；6号关节保证工作空间内全部到达
# min_list_temp = [-85, -15, -149, -142, -179, -179]  # 关节模型角度最小值，1号关节目的是不装到装在桌边的竖杆（安装摄像头）；2号关节目的是在伸直的时候不打到桌子；3、4号关节目的是不产生杆件干涉；5号关节因有滑环，不考虑；6号关节保证工作空间内全部到达
##3d打印材质、上课用
# max_list_temp = [90, 90, 0, 90, 179,90]
# min_list_temp=[-90, 0, -120 -80, 0, -90]
max_list_temp = [90, 90, 0, 179, 179, 90]
min_list_temp = [-90, 0, -130, -179, -179, -90]
class arm_robot(gripper.gripper):
    """六关节机械臂类（继承arm类），具有六个驱动关节，六个关节依次串联

    Attributes:
        c_angle_list: 机器人所有关节当前模型角度（角度制）组成的列表；
        pose_list: 用来存放机器人动作序列的列表；
        ID_list: 机器人关节所对应编号组成的列表；
        P1_list: 机器人关节模型角度与电机机实际角度转换的第一个系数的列表；两者的变换关系为：舵机角度=P1*模型角度+P2（模型关节轴线与电机转动轴线同向为1，反向为-1）
        P2_list: 机器人关节模型角度与电机实际角度转化的第二个系数的列表；两者的变换关系为：舵机角度=P1*模型角度+P2（电机处于模型初始位置安装时对应的实际角度）
        MAX_list: 由机械臂各个关节模型角最大值组成的列表；
        MIN_list: 由机械臂各个关节模型角最小值组成的列表；
        init_model_angle: 机械臂安装位置各关节模型角度值（固定值，不可改变）；
        max_speed: 机械臂最大运动速度，一般取各个关节电机中转速最低的那个；
        tutorial_t：示教编程中位姿停留时间
        tutorial_t_list：每个位姿的停留时间组成的列表
        bit_time：系统发送一次运动指令所需的时间
        torque_limits：机械臂各关节电机最大限制力矩组成的列表
    Methods:
        __init__: 对象初始化函数；
        ###############运动控制函数######################
        set_arm_pose：控制机械臂末端运动到指定位置和姿态函数
        set_arm_position: 控制机械臂末端运动到指定位置函数（不改变姿态角）
        set_arm_P_R_Y：控制机械臂末端运动到相对于机械臂本体指定姿态角函数（不改变末端位置）
        set_arm_relative_pose：控制机械臂相对当前位姿运动一定距离和角度函数
        set_arm_relative_position：控制机械臂相对当前位姿运动一定距离函数（不改变姿态角）
        set_arm_relative_P_R_Y：控制机械臂相对当前位姿运动一定角度函数（不改变末端位置）
        set_arm_poses：控制机械臂连续运动多个位置和姿态函数，即执行一定轨迹
        set_arm_poses_curve_pre：预先设置机械臂末端运动轨迹函数
        set_arm_poses_curve_start_point：控制机械臂运动到预设轨迹起点的函数，需与在set_arm_poses_curve_pre之后使用
        set_arm_poses_curve_do：控制机械臂执行预设轨迹函数
        set_arm_joints：控制机械臂各个关节运动到指定模型角度函数
        ###############数据回读函数####################
        show_pose：根据内存中的模型角度反馈机械臂当前位姿函数
        show_joint_z_position：根据内存中的模型角度反馈机械臂各关节中心在全局坐标系中的z轴坐标函数
        show_joint_x_position：根据内存中的模型角度反馈机械臂各关节中心在机械臂平面中的x轴坐标函数
        detect_pose：根据回读的电机角度反馈机械臂当前位姿函数
        detect_joint_z_position：根据回读的电机角度反馈机械臂各关节中心在全局坐标系中的z轴坐标函数
        detect_joint_x_position：根据回读的电机角度反馈机械臂各关节中心在机械臂平面中的x轴坐标函数
        get_property：查看机械臂关节参数
        get_pid_joint：查看机械臂关节 PID 参数
        ###############参数设置函数####################
        set_property：设置机械臂关节参数
        set_pid：设置机械臂关节 PID 参数
        init_pid：机器人初始化 PID 参数，在机器人初始化函数中使用
        ###############力控函数####################
        set_torques_for_pose_hold：重力补偿函数，保持机械臂当前位姿（调试中）
        zero_force_pull：零力拖动，保持机械臂当前位姿并可轻易拖动，拖动释放后静止（调试中）
        impedance_control_joints：机械臂关节阻抗控制函数
        ###############辅助功能函数####################
        save_pose: 基于机械臂当前末端位姿求解各关节模型角度，并保存pose_list中；
        model_to_servo: 将pose_list中保存的机器人关节模型角度转换成电机实际角度；
        do_motion: 动作执行函数，让机器人依次执行之前保存的动作；
        range_init: 机械臂关节转动范围设置函数，设置的是模型角度范围；
        add_pose: 将当前机械臂姿态保存到pose_list中
        read_joints：读取机械臂所有关节电机当前角度；
        read_joints_p_v：读取机械臂所有关节电机当前角度和转速
        servo_to_model: 将读取的舵机角度转换成关节模型角度；
        free: 将所有关节设置成待机模式函数，便于手动掰动关节，注意待机之前需要手扶；
        lock：将所有关节固定在当前角度函数，让机械臂保持在当前位置，保持刚度；
        clear_pose: 删除pose_list中某个pose或清空所有pose；
        pose_done：检查并等待机械臂运动到指定位置
    """
    c_angle_list = []
    pose_list = []
    servo_angle_list_list = []
    pose_list_servo = []
    pose_list_temp = []
    ID_list = [1, 2, 3, 4, 5, 6]
    P1_list = [1, -1, 1, -1, 1, 1]  # 对应关节电机输出轴与模型中z轴方向一致则为1，不一致则为-1
    # P2_list = [0, 0, 0, 0, 0, 0]  # 关节电机在机械臂初始位置安装时的实际角度，本机械臂关节皆为在零位安装
    # MAX_list = [90,0, 0, 160, 180, 180]
    # MIN_list = [-90, -90, -130, -160, -180, -180]
    #P1_list = [1, 1, -1, 1, 1, -1]  # 对应关节电机输出轴与模型中z轴方向一致则为1，不一致则为-1
    P2_list = [0, 0, 0, 0, 0, 0]  # 关节电机在机械臂初始位置安装时的实际角度，本机械臂关节皆为在零位安装
    MAX_list = [160, 180, 160, 160, 180, 180]
    MIN_list = [-160, -40, -160, -160, -180, -180]
    init_model_angle = [0, 0, 0, 0, 0, 0]
    max_speed = 30
    tutorial_t = 1  # 示教编程中位姿停留时间
    tutorial_t_list = [0]  # 每个位姿的停留时间组成的列表
    bit_time = 0  # 系统发送一次运动指令所需的时间
    torque_limits = []  # 机械臂各关节电机最大限制力矩组成的列表
    torque_factors = [1, 1, 1, 1, 1, 1]  # 于调节模型扭矩与电机扭矩的比例关系，当重力补偿或零力拖动效果不佳时可用该参数调节

    # 机械臂初始化函数
    def __init__(self, L_p=0, L_p_mass_center=0, MAX_list_temp=max_list_temp, MIN_list_temp=min_list_temp, G_p=0,
                 com='', uart_baudrate=115200):
        """机械臂对象初始化函数，初始化机械臂对象的属性参数

        Args:
            l_p: 工具参考点到电机输出轴表面的距离；
            MAX_list_temp: 机械臂关节模型角度最大值组成的列表
            MIN_list_temp: 机械臂关节模型角度最小值组成的列表
            G_p: 负载重量
        Returns:
            无
        Raises:
            无
        """
        time.sleep(1)  # 延时1秒，给与机械臂充分上电时间
        self.JOINT_NUMBER = 6  # 六轴机械臂有6个关节
        gripper.gripper.__init__(self, L_p=L_p, L_p_mass_center=L_p_mass_center, G_p=G_p, com=com,
                                 uart_baudrate=uart_baudrate)  # 将机械臂（运动模型）对象初始化
        self.init_pid(pids=[[10, 5, 0.55], [10.56, 4.95, 0.391], [10.56, 4.95, 0.391], [10, 9, 0.5], [12, 5, 0.1],
                            [12, 5, 0.096]])
        self.range_init(MAX_list_temp, MIN_list_temp)  # 关节模型角度范围初始化
        servo_angle_list = self.read_joints()  # 读取机械臂关节电机实际角度
        if servo_angle_list != False:
            self.c_angle_list = self.servo_to_model(servo_angle_list=servo_angle_list)  # 根据读取到的电机角度初始化关节模型角度（角度制）
        else:
            self.c_angle_list = self.init_model_angle[:]  # 若关节电机读取失败则关节模型角度取默认值
        self.theta = [DEG_RAD * i for i in self.c_angle_list]  # 初始化关节模型角度（弧度制）
        pose = self.theta[:]  # 将关节模型角度赋值给pose（弧度制）
        for i in range(len(pose)):
            pose[i] *= RAD_DEG  # 将弧度制转换成角度制
        self.pose_list.append(pose)  # 初始化 pose_list
        self.forward_kinematics_pose(self.c_angle_list)  # 根据初始化的关节模型角度计算初始的末端位置pl和姿态theta_P_Y_R
        pose_list_servo_temp = self.read_joints()  # 再读取一次关节电机角度，以进行后面的操作
        self.set_angles(self.ID_list, pose_list_servo_temp, 20, 20, 0)
        start = time.time()
        for k in range(len(self.ID_list)):
            self.set_angles(self.ID_list, pose_list_servo_temp, 20, 20, 0)
        self.bit_time = (time.time() - start) / (k + 1)  # 计算得到系统发送一次位置控制指令所需时间
        print('初始化成功')
        time.sleep(0.2)

    # def set_arm_pose(self,target_position=[0, 0, 0], target_orientation=[0, 0, 0],table_height=-150):
    #     """
    #     设置机械臂的目标姿态，并检查关节角度范围和桌面高度
    #     参数:
    #     target_position (list): 目标位置 [x, y, z] (单位: 米)
    #     target_orientation (list): 目标姿态 [roll, pitch, yaw] (单位: 弧度)
    #     max_list (list): 各关节角度的最大值 [joint1_max, joint2_max, ..., joint6_max] (单位: 弧度)
    #     min_list (list): 各关节角度的最小值 [joint1_min, joint2_min, ..., joint6_min] (单位: 弧度)
    #     table_height (float): 桌面的高度 (单位: 米)
    #
    #     返回:
    #     bool: 如果成功设置姿态返回 True，否则返回 False
    #     """
    #     theta_bk = self.theta.copy()
    #     # 调用逆解函数计算关节角度
    #     if not self.inverse_kinematics(target_position, target_orientation,0):
    #         print("逆解失败：无法计算目标姿态对应的关节角度")
    #         self.theta = theta_bk.copy()
    #         return False
    #
    #         # 检查关节角度是否在规定范围内
    #     joint_angles = [(180/cm.pi) * i for i in self.theta]
    #     if joint_angles is None:
    #         print("逆解失败：无法计算目标姿态对应的关节角度")
    #         self.theta = theta_bk.copy()
    #         return False
    #     for i, angle in enumerate(joint_angles):
    #         if angle <self.MIN_list[i] or angle > self.MAX_list[i]:
    #             print(f"关节 {i + 1} 的角度 {angle} 超出允许范围 [{self.MIN_list[i]}, {self.MAX_list[i]}]")
    #             self.theta = theta_bk.copy()
    #             return False
    #     # 检查手臂各关节是否低于桌面高度
    #     # 假设有一个函数 forward_kinematics() 可以根据关节角度计算机械臂末端的位置
    #     if self.show_joint_z_position(4) < table_height or self.show_joint_z_position(6)<table_height-50:
    #         print(f"机械臂末端位置低于桌面高度 {table_height}")
    #         self.theta = theta_bk.copy()
    #         return False
    #     # 设置机械臂关节角度
    #     success=self.set_arm_joints(angle_list=joint_angles,speed=10)
    #     if not success:
    #         print("设置关节角度失败")
    #         return False
    #
    #     print("机械臂姿态设置成功")
    #     print(joint_angles)
    #     return True

    def set_arm_linear_interpolation(self, pl_start=[0, 0, 0], pl_end=[0, 0, 0], steps=100, speed=1.0, param=10,
                                     mode=1):
        """
        控制机械臂末端从起始位置平滑移动到结束位置，同时保持姿态不变。

        Args:
            pl_start: 机械臂末端起始位置坐标组成的列表[x, y, z]
            pl_end: 机械臂末端结束位置坐标组成的列表[x, y, z]
            steps: 插补步数，步数越多，插补越平滑
            speed: 当前姿态被执行时转速最快关节的转动速度
            param: 控制参数，具体含义见set_angles函数
            mode: 控制电机转动模式，0为轨迹追踪模式，1为梯形轨迹模式，2为前馈模式
        Returns:
            无
        Raises:
            无
        """
        if len(pl_start) != 3 or len(pl_end) != 3:
            print("起始位置和结束位置必须是包含三个坐标的列表")
            return

        # 计算每一步的位置变化
        delta_x = (pl_end[0] - pl_start[0]) / steps
        delta_y = (pl_end[1] - pl_start[1]) / steps
        delta_z = (pl_end[2] - pl_start[2]) / steps

        # 当前位置初始化为起始位置
        current_pl = pl_start.copy()
        for _ in range(steps):
            # 更新当前位置
            current_pl[0] += delta_x
            current_pl[1] += delta_y
            current_pl[2] += delta_z

            # 设置机械臂到当前位置
            self.set_arm_position(pl_temp=current_pl, speed=speed, param=param, mode=mode)
            # 可选：添加一个小延迟以控制插补速度
            time.sleep(0.01)

    # 一、机械臂位置和姿态控制函数
    def set_arm_pose(self, pl_temp=[0, 0, 0], theta_P_R_Y=[0, 0, 0], speed=1.0, param=10, mode=1):
        """控制机械臂运动到指定位置和姿态
        Args:
            pl_temp: 机械臂末端位置坐标组成的列表[x, y, z]
            theta_P_R_Y： 机械臂末端姿态角组成的列表[pitch, roll, yaw]
            speed: 当前姿态被执行时转速最快关节的转动速度
        Returns:
            无
        Raises:
            无
        """
        if len(pl_temp) > 0:
            pl_bk = self.pl.copy()  # 记录此前末端位置坐标，此处不能直接使用 pl_bk = self.pl 否则 pl_bk 会始终随着 self.pl 变化
            theta_P_R_Y_bk = self.theta_P_Y_R.copy()  # 记录此前末端姿态角
            self.pl = pl_temp.copy()  # 将目标位置赋值给对象位置属性 pl
            self.theta_P_Y_R = theta_P_R_Y.copy()  # 将目标位置赋值给对象姿态属性 theta_P_R_Y
            self.clear_pose()
            if self.save_pose():  # 运动学逆解并将结果保存进 pose_list
                self.do_motion(speed=speed, mode=mode, param=param)  # 执行动作
            else:
                self.pl = pl_bk.copy()  # 若运动学逆解失败，则恢复此前的末端位置坐标
                self.theta_P_Y_R = theta_P_R_Y_bk.copy()  # 若运动学逆解失败，则恢复此前的末端姿态角

    def set_arm_position(self, pl_temp=[0, 0, 0], speed=1.0, param=10, mode=1):
        """控制机械臂末端运动到指定位置，不改变机械臂末端角姿态
        Args:
            pl_temp: 机械臂末端坐标组成的列表[x, y, z]
            speed: 当前姿态被执行时转速最快关节的转动速度
        Returns:
            无
        Raises
            无
        """
        if len(pl_temp) > 0:
            pl_bk = self.pl.copy()  # 记录此前末端位置坐标，此处不能直接使用 pl_bk = self.pl 否则 pl_bk 会始终随着 self.pl 变化
            self.pl = pl_temp.copy()  # 将目标位置赋值给对象位置属性 pl
            self.clear_pose()
            if self.save_pose():  # 运动学逆解并将结果保存进 pose_list
                self.do_motion(speed=speed, mode=mode, param=param)  # 执行动作
            else:
                self.pl = pl_bk.copy()  # 若运动学逆解失败，则恢复此前的末端位置坐标

    def set_arm_P_R_Y(self, theta_P_Y_R=[0, 0, 0], speed=1.0, param=10, mode=1):
        """控制机械臂末端运动到指定姿态，不改变末端位置

        Args:
            theta_P_Y_R: 机械臂末端姿态角组成的列表[pitch, yaw, roll]
            speed: 当前姿态被执行时转速最快关节的转动速度
        Returns:
            无
        Raises:
            无
        """
        theta_P_Y_R_bk = self.theta_P_Y_R.copy()  # 记录此前末端姿态角
        self.theta_P_Y_R = theta_P_Y_R.copy()  # 将目标位置赋值给对象姿态属性 theta_P_Y_R
        self.clear_pose()
        if self.save_pose():  # 运动学逆解并将结果保存进 pose_list
            self.do_motion(speed=speed, mode=mode, param=param)  # 执行动作
        else:
            self.theta_P_Y_R = theta_P_Y_R_bk.copy()  # 若运动学逆解失败，则恢复此前的末端姿态角

    def set_arm_relative_pose(self, pl_temp=[0, 0, 0], theta_P_Y_R=[0, 0, 0], speed=1.0, param=10, mode=1):
        """控制机械臂运动到相对当前的指定位置和姿态

        Args:
            pl_temp: 机械臂末端相对坐标组成的列表 [x, y, z]
            theta_P_Y_R： 机械臂末端相对姿态角组成的列表 [Pitch, Yaw, Roll]
            speed: 当前姿态被执行时转速最快关节的转动速度
        Returns:
            无
        Raises:
            无
        """
        if len(pl_temp) > 0:
            pl_bk = self.pl.copy()  # 记录此前末端坐标，此处不能直接使用 pl_bk = self.pl 否则 pl_bk 会始终随着 self.pl 变化
            theta_P_Y_R_bk = self.theta_P_Y_R.copy()  # 记录此前末端姿态角
            for i in range(len(pl_temp)):
                self.pl[i] = self.pl[i] + pl_temp[i]  # 机械臂末端坐标值加上相对坐标值
            for i in range(len(theta_P_Y_R)):
                self.theta_P_Y_R[i] = self.theta_P_Y_R[i] + theta_P_Y_R[i]  # 机械臂末端姿态角加上相对姿态角
            self.clear_pose()
            if self.save_pose():  # 运动学逆解，并将结果存进 pose_list
                self.do_motion(speed=speed, mode=mode, param=param)  # 执行动作
            else:
                self.pl = pl_bk.copy()  # 若运动学逆解失败，则恢复此前的末端位置坐标
                self.theta_P_Y_R = theta_P_Y_R_bk.copy()  # 若运动学逆解失败，则恢复此前的末端姿态角

    def set_arm_relative_position(self, pl_temp=[0, 0, 0], speed=1.0, param=10, mode=1):
        """控制机械臂运动到相对当前位置的指定位置，不改变机械臂角姿态

        Args:
            pl_temp: 机械臂末端相对部坐标组成的列表[x, y, z]
            speed: 当前姿态被执行时转速最快关节的转动速度
        Returns:
            无
        Raises:
            无
        """
        if len(pl_temp) > 0:
            pl_bk = self.pl.copy()  # 记录此前末端坐标，此处不能直接使用 pl_bk = self.pl 否则 pl_bk 会始终随着 self.pl 变化
            for i in range(len(pl_temp)):
                self.pl[i] = self.pl[i] + pl_temp[i]  # 机械臂末端坐标值加上相对坐标值
            self.clear_pose()
            if self.save_pose():  # 运动学逆解，并将结果存进 pose_list
                self.do_motion(speed=speed, mode=mode, param=param)  # 执行动作
            else:
                self.pl = pl_bk.copy()  # 若运动学逆解失败，则恢复此前的末端姿态角

    def set_arm_relative_P_R_Y(self, theta_P_Y_R=[0, 0, 0], speed=1.0, param=10, mode=1):
        """控制机械臂运动到相对于当前姿态的指定姿态，不改变末端位置

        Args:
            theta_P_Y_R： 机械臂末端相对姿态角组成的列表[Pitch, Yaw, Roll]
            speed: 当前姿态被执行时转速最快关节的转动速度
        Returns:
            无
        Raises:
            无
        """
        theta_P_Y_R_bk = self.theta_P_Y_R.copy  # 记录此前末端姿态角
        for i in range(len(theta_P_Y_R)):
            self.theta_P_Y_R[i] = self.theta_P_Y_R[i] + theta_P_Y_R[i]  # 机械臂末端姿态角加上相对姿态角
        self.clear_pose()
        if self.save_pose():  # 运动学逆解，并将结果存进 pose_list
            self.do_motion(speed=speed, mode=mode, param=param)  # 执行动作
        else:
            self.theta_P_Y_R = theta_P_Y_R_bk.copy()  # 若运动学逆解失败，则恢复此前的末端姿态角

    # 二、机械臂轨迹控制函数

    def set_arm_poses(self, pls_temp=[], theta_P_Y_Rs_temp=[], t=1):
        """控制机械臂末端按顺序连续运动到多个指定位置和姿态

        Args:
                pls_temp: 机械臂末端连续多个坐标组成的列表[x, y, z]
            theta_P_Y_R： 机械臂末端连续多个姿态角组成的列表[Pitch, Yaw, Roll]
                      t: 运动执行的大致时间
        Returns:
            无
        Raises:
            无
        """

        if len(pls_temp) > len(theta_P_Y_Rs_temp):
            for ii in range(len(pls_temp) - len(theta_P_Y_Rs_temp)):
                theta_P_Y_Rs_temp.append(
                    theta_P_Y_Rs_temp[len(theta_P_Y_Rs_temp) - 1])  # 如果输入的姿态角数目少于坐标值数目，则缺少的姿态角用最后一组姿态角补足
        if len(theta_P_Y_Rs_temp) > len(pls_temp):
            for ii in range(len(theta_P_Y_Rs_temp) - len(pls_temp)):
                pls_temp.append(pls_temp[len(pls_temp) - 1])  # 如果输入的坐标值数目少于坐标值数目，则缺少的坐标值用最后一组坐标值补足
        n = len(pls_temp)
        self.clear_pose()
        for i in range(n):  # 将输入的坐标值和姿态角全部带入逆解函数求解出关节模型角度
            pl_temp = pls_temp[i]
            if len(pl_temp) > 0:
                pl_bk = self.pl.copy()  # 记录此前末端坐标值，此处不能直接使用 pl_bk = self.pl 否则 pl_bk 会始终随着 self.pl 变化
                theta_P_Y_R_bk = self.theta_P_Y_R.copy()  # 记录此前末端姿态角
                self.pl = pl_temp.copy()  # 将轨迹点赋值给 pl
                self.theta_P_Y_R = theta_P_Y_Rs_temp[i].copy()  # 将轨迹点赋值给 theta_P_Y_R
                if not self.save_pose():  # 运动学逆解，并将求解结果存进 pose_list
                    self.pl = pl_bk.copy()  # 如果逆解失败则取此前记录的末端坐标值
                    self.theta_P_Y_R = theta_P_Y_R_bk.copy()  # 如果逆解失败则取此前记录的姿态角
        self.pose_list_servo = []  # pose_list_servo 在此置空，用于存放计算得到的关节电机实际角度列表
        for j in range(len(self.pose_list)):
            self.pose_list_servo.append(
                self.model_to_servo(model_angle_list=self.pose_list[j]))  # 将关节模型角度转换成关节电机实际角度并添加进 pose_list_servo
        n = len(self.pose_list_servo)
        self.set_arm_poses_curve_start_point(speed=10)  # 以梯形轨迹模式运动到轨迹起始点
        if t >= n * self.bit_time:  # 轨迹要求的执行时间大于系统发送 n 个动作指令（所有轨迹点）所需的时间
            print('t >= n * bit_time')  # 提示
            bit_wideth = n / t / 2  # 指令发送带宽的一半
            start = time.time()  # 记录指令发送开始时间
            self.set_angles(self.ID_list, self.pose_list_servo[0], 20, bit_wideth,
                            0)  # 采用轨迹跟踪模式发送首条位置指令，根据轨迹跟踪模式要求，bit_wideth 需为实际指令发送带宽的一半
            while (time.time() - start) < (t / n):  # 将轨迹执行时间按照点的数量均分
                time.sleep(0.001)
            bit_wideth1 = 1 / (time.time() - start) / 2  # 计算在 t>n 情况下的指令发送频率的一半
            for k in range(n):
                start = time.time()
                self.set_angles(self.ID_list, self.pose_list_servo[k], 20, bit_wideth1,
                                0)  # 采用轨迹跟踪模式，根据轨迹跟踪模式要求，bit_wideth1 需为实际指令发送带宽的一半
                while (time.time() - start) < (t / n):
                    time.sleep(0.001)
                bit_wideth1 = 1 / (time.time() - start) / 2  # 时刻监控在 t>n 情况下单条指令发送的时间
        else:
            print('t < n * bit_time')  # 提示轨迹要求的时间小于系统发送 n 个指令（所有轨迹点）所需的时间
            bit_wideth = 1 / self.bit_time / 2  # 指令发送频率取初始化阶段获得的系统单条指令发送所需时间的倒数，再取其一半作为 set_angles() 函数参数
            for k in range(n):
                self.set_angles(self.ID_list, self.pose_list_servo[k], 20, bit_wideth, 0)  # 使用轨迹追踪模式控制机械臂运动

    def set_arm_poses_curve_pre(self, pls_temp=[], theta_P_Y_Rs_temp=[]):
        """预先计算机械臂按顺序连续运动到多个指定位置和姿态所对应的关节电机角度

        Args:
            pls_temp: 机械臂末端连续多个坐标组成的列表[x, y, z]
            theta_P_Y_R： 机械臂末端连续多个姿态角组成的列表[pitch, yaw, roll]
        Returns:
            无
        Raises:
            pose_list_servo：关节电机角度列表所组成的列表
        """

        if len(pls_temp) > len(theta_P_Y_Rs_temp):
            for ii in range(len(pls_temp) - len(theta_P_Y_Rs_temp)):
                theta_P_Y_Rs_temp.append(
                    theta_P_Y_Rs_temp[len(theta_P_Y_Rs_temp) - 1])  # 如果输入的姿态角数目少于坐标值数目，则缺少的姿态角用最后一组姿态角补足
        if len(theta_P_Y_Rs_temp) > len(pls_temp):
            for ii in range(len(theta_P_Y_Rs_temp) - len(pls_temp)):
                pls_temp.append(pls_temp[len(pls_temp) - 1])  # 如果输入的坐标值数目少于姿态角数目，则缺少的姿态角用最后一组坐标值补足
        n = len(pls_temp)
        self.clear_pose()
        for i in range(n):
            pl_temp = pls_temp[i]
            if len(pl_temp) > 0:
                pl_bk = self.pl.copy()  # 记录此前末端坐标值，此处不能直接使用 pl_bk = self.pl 否则 pl_bk 会始终随着 self.pl 变化
                theta_P_Y_R_bk = self.theta_P_Y_R.copy()  # 记录此前末端姿态角
                self.pl = pl_temp.copy()  # 将轨迹点赋值给 pl
                self.theta_P_Y_R = theta_P_Y_Rs_temp[i].copy()  # 将轨迹点赋值给 theta_P_Y_R
                if not self.save_pose():  # 运动学逆解，并将求解结果存进 pose_list
                    self.pl = pl_bk.copy()  # 如果逆解失败则取此前记录的末端坐标值
                    self.theta_P_Y_R = theta_P_Y_R_bk.copy()  # 如果逆解失败则取此前记录的姿态角
        self.pose_list_servo = []  # pose_list_servo 在此置空，用于存放计算得到的关节电机实际角度列表
        for j in range(len(self.pose_list)):
            self.pose_list_servo.append(self.model_to_servo(model_angle_list=self.pose_list[j]))

    def set_arm_poses_curve_start_point(self, speed):
        """控制机械臂运动到轨迹中的首个位置和姿态, 用于set_arm_poses_cure_pre之后

        Args:
            speed: 机械臂动作执行时转速最快关节的转动速度
        Returns:
            无
        Raises:
            无
        """

        if speed > self.max_speed:
            speed = self.max_speed  # 最大速度限制，保证安全性和动作一致性
        self.forward_kinematics_pose(angle_list=self.pose_list[0])  # 使用运动学正解函数计算轨迹的第一个位置和姿态
        self.set_arm_pose(pl_temp=[self.pl[0], self.pl[1], self.pl[2]], theta_P_Y_R=self.theta_P_Y_R, speed=speed,
                          param=10, mode=1)
        self.pose_done()  # 监控并等待动作执行结束

    def set_arm_poses_curve_do(self, t=1):
        """控制机械臂按顺序连续运动到多个指定位置和姿态, 与set_arm_poses_cure_pre连用

        Args:
            t: 用来指定当前姿态被执行时的大致时长
        Returns:
            无
        Raises:
            无
        """

        # self.set_arm_poses_curve_start_point(speed=10) # 以梯形轨迹模式运动到轨迹起始点
        n = len(self.pose_list_servo)
        if t >= n * self.bit_time:  # 轨迹要求的执行时间大于系统发送 n 个动作指令（所有轨迹点）所需的时间
            print('t >= n * bit_time')  # 提示
            bit_wideth = n / t / 2  # 指令发送带宽的一半
            start = time.time()  # 记录指令发送开始时间
            self.set_angles(self.ID_list, self.pose_list_servo[0], 20, bit_wideth,
                            0)  # 采用轨迹跟踪模式发送首条位置指令，根据轨迹跟踪模式要求，bit_wideth 需为实际指令发送带宽的一半
            while (time.time() - start) < (t / n):  # 将轨迹执行时间按照点的数量均分
                time.sleep(0.001)
            bit_wideth1 = 1 / (time.time() - start) / 2  # 计算在 t>n 情况下的指令发送频率的一半
            for k in range(n):
                start = time.time()
                self.set_angles(self.ID_list, self.pose_list_servo[k], 20, bit_wideth1,
                                0)  # 采用轨迹跟踪模式，根据轨迹跟踪模式要求，bit_wideth1 需为实际指令发送带宽的一半
                while (time.time() - start) < (t / n):
                    time.sleep(0.001)
                bit_wideth1 = 1 / (time.time() - start) / 2  # 时刻监控在 t>n 情况下单条指令发送的时间
        else:
            print('t < n * bit_time')  # 提示轨迹要求的时间小于系统发送 n 个指令（所有轨迹点）所需的时间
            bit_wideth = 1 / self.bit_time / 2  # 指令发送频率取初始化阶段获得的系统单条指令发送所需时间的倒数，再取其一半作为 set_angles() 函数参数
            for k in range(n):
                self.set_angles(self.ID_list, self.pose_list_servo[k], 20, bit_wideth, 0)  # 使用轨迹追踪模式控制机械臂运动

    # 三、机械臂关节角度控制函数
    def set_arm_joints(self, angle_list=[0, 90, 0, 0, 0, 0], speed=1.0):
        """控制机械臂六个关节运动到指定模型角度

        Args:
            angle_list: 机械臂六个关节模型角度组成的列表[joint1, joint2, joint3， joint4, joint5, joint6]
            speed: 当前姿态被执行时转速最快关节的转动速度
        Returns:
            无
        Raises:
            无
        """

        self.clear_pose()
        if len(angle_list) == len(self.ID_list):
            pose = angle_list[:]
            for i in range(len(pose)):
                if pose[i] < self.MIN_list[i]:
                    print("第" + str(i + 1) + "个关节角度超出了最小极限角度")  # 检查关节模型角度是否小于最小允许值
                    return False
                if pose[i] > self.MAX_list[i]:
                    print("第" + str(i + 1) + "个关节角度超出了最大极限角度")  # 检查关节模型角度是否大于最大允许值
                    return False
            self.pose_list.append(pose)  # 将输入的关节角度列表保存进 pose_list
            self.tutorial_t_list.append(0)  # 指定该组姿态与下组姿态的时间间隔为 0
            self.do_motion(speed=speed)  # 执行动作
            self.forward_kinematics_pose(angle_list)  # 使用运动学正解计算指定机械臂关节模型角度后的位置和姿态
            return True
        else:
            print("角度参数有误！")
            return False

    # 四、机械臂位置参数回读函数
    def show_pose(self):
        """根据内存中的关节模型角度值，显示当前末端位置和姿态

        Args:
            无
        Returns:
            末端位置和姿态坐标
        Raises:
            无
        """

        model_angle_list = [RAD_DEG * i for i in self.theta]  # 用于安全观察，因此使用内存参数快速计算
        self.forward_kinematics_pose(model_angle_list)  # 调用运动学正解函数
        print("当前机械臂末端x坐标为: " + str(self.pl[0]) + "; y坐标为： " + str(self.pl[1]) + "; z坐标为： " + str(
            self.pl[2]) + "; Pitch角为: " +
              str(self.theta_P_Y_R[0]) + "; Yaw角为: " + str(self.theta_P_Y_R[1]) + "; Roll角为: " + str(
            self.theta_P_Y_R[2]))

    def show_joint_z_position(self, n=0):
        """根据内存中的关节模型角度值，查看当前各个关节的z坐标，关节序号0~6

        Args:
            n: 关节序号，0代表显示所有关节的z坐标
        Returns:
            运动关节的z坐标
        Raises:
            无
        """

        model_angle_list = [RAD_DEG * i for i in self.theta]  # 用于安全观察，因此使用内存参数快速计算
        z_position = self.forward_kinematics_joint_z_postion(model_angle_list).copy()  # 调用运动学正解函数（计算z坐标）
        if n < 0 or n > 6:  # 防输错
            print('关节编号超出索引范围0~6，请重新输入')
            return False
        if n == 0:
            print("关节3的z坐标: " + str(z_position[0]) + "; 关节4的z坐标： " + str(z_position[1]) + "; 关节5的z坐标： " + str(
                z_position[2]) + "; 关节6的z坐标: " +
                  str(z_position[3]))
            return True
        if n == 1:
            return 0
        if n == 2:
            return 0
        if n == 3:
            return z_position[0]
        if n == 4:
            return z_position[1]
        if n == 5:
            return z_position[2]
        if n == 6:
            return z_position[3]

    def show_joint_x_position(self, n=0):
        """根据内存中的关节模型角度值，查看当前关节的x坐标，关节序号0~6

        Args:
            n: 关节序号，0代表显示所有关节的x坐标
        Returns:
            关节的z坐标
        Raises:
            无
        """

        model_angle_list = [RAD_DEG * i for i in self.theta]  # 用于安全观察，因此使用内存参数快速计算
        x_position = self.forward_kinematics_joint_x_postion(model_angle_list).copy()
        if n < 0 or n > 6:
            print('关节编号超出索引范围0~6，请重新输入')
            return False
        if n == 0:
            print("在机械臂平面内，关节3x坐标为: " + str(x_position[0]) + "; 关节4x坐标为： " + str(x_position[1]) + "; 关节5x坐标为： " + str(
                x_position[2]) + "; 关节6x坐标为: " +
                  str(x_position[3]))
        if n == 1:
            return 0
        if n == 2:
            return 0
        if n == 3:
            return x_position[0]
        if n == 4:
            return x_position[1]
        if n == 5:
            return x_position[2]
        if n == 6:
            return x_position[3]

    def detect_joints(self):
        """根据读取到的一体化关节角度值，显示当前关节模型角度

        Args:
            无
        Returns:
            机器人关节模型角度
        Raises:
            无
        """

        servo_angle_list = self.read_joints()  # 读取各一体化关节角度（用于查询未知姿态，因此读取关节角度）
        model_angle_list = self.servo_to_model(servo_angle_list=servo_angle_list)  # 将一体化关节角度转换为关节模型角度
        print("关节 1 模型角度为: " + str(model_angle_list[0]))
        print("关节 2 模型角度为: " + str(model_angle_list[1]))
        print("关节 3 模型角度为: " + str(model_angle_list[2]))
        print("关节 4 模型角度为: " + str(model_angle_list[3]))
        print("关节 5 模型角度为: " + str(model_angle_list[4]))
        print("关节 6 模型角度为: " + str(model_angle_list[5]))
        return model_angle_list

    def detect_pose(self):
        """根据读取到的关节电机角度值，显示当前末端位置和姿态

        Args:
            无
        Returns:
            末端位置和姿态坐标
        Raises:
            无
        """

        servo_angle_list = self.read_joints()  # 读取各关节电机角度（用于查询未知姿态，因此读取关节角度）
        model_angle_list = self.servo_to_model(servo_angle_list=servo_angle_list)  # 将关节电机角度转换为关节模型角度
        self.forward_kinematics_pose(model_angle_list)  # 调用运动学正解函数
        print("当前机械臂末端x坐标为: " + str(self.pl[0]) + "; y坐标为： " + str(self.pl[1]) + "; z坐标为： " + str(
            self.pl[2]) + "; Pitch角为: " +
              str(self.theta_P_Y_R[0]) + "; Yaw角为: " + str(self.theta_P_Y_R[1]) + "; Roll角为: " + str(
            self.theta_P_Y_R[2]))

    def detect_joint_z_position(self, n=0):
        """根据读取到的关节电机角度值，查看当前各个关节的z坐标，关节序号0~6

        Args:
            n: 关节序号，0代表显示所有关节的z坐标
        Returns:
            关节的z坐标
        Raises:
            无
        """

        servo_angle_list = self.read_joints()  # 读取各关节电机角度（用于查询未知姿态，因此读取关节角度）
        model_angle_list = self.servo_to_model(servo_angle_list=servo_angle_list)  # 将关节电机角度转换为关节模型角度
        z_position = self.forward_kinematics_joint_z_postion(model_angle_list).copy()  # 调用运动学正解函数
        if n < 0 or n > 6:
            print('关节编号超出索引范围0~6，请重新输入')
            return False
        if n == 0:
            print("关节3的z坐标: " + str(z_position[0]) + "; 关节4的z坐标： " + str(z_position[1]) + "; 关节5的z坐标： " + str(
                z_position[2]) + "; 关节6的z坐标: " +
                  str(z_position[3]))
        if n == 1:
            return 0
        if n == 2:
            return 0
        if n == 3:
            return z_position[0]
        if n == 4:
            return z_position[1]
        if n == 5:
            return z_position[2]
        if n == 6:
            return z_position[3]

    def detect_joint_x_position(self, n=0):
        """根据读取到的关节电机角度值，查看当前各个关节的x坐标，关节序号0~6

        Args:
            n: 关节序号，0代表显示所有关节的x坐标
        Returns:
            关节的x坐标
        Raises:
            无
        """

        servo_angle_list = self.read_joints()  # 读取各关节电机角度（用于查询未知姿态，因此读取关节角度）
        model_angle_list = self.servo_to_model(servo_angle_list=servo_angle_list)  # 将关节电机角度转换为关节模型角度
        x_position = self.forward_kinematics_joint_x_postion(model_angle_list).copy()  # 调用运动学正解函数
        if n < 0 or n > 6:
            print('关节编号超出索引范围0~6，请重新输入')
            return False
        if n == 0:
            print("在机械臂平面内，关节3x坐标为: " + str(x_position[0]) + "; 关节4x坐标为： " + str(x_position[1]) + "; 关节5x坐标为： " + str(
                x_position[2]) + "; 关节6x坐标为: " +
                  str(x_position[3]))
        if n == 1:
            return 0
        if n == 2:
            return 0
        if n == 3:
            return x_position[0]
        if n == 4:
            return x_position[1]
        if n == 5:
            return x_position[2]
        if n == 6:
            return x_position[3]

    # 五、机械臂静力学重力补偿控制函数
    def set_torques_for_pose_hold(self, pay_load=0, F=[0, 0, 0]):
        """设置1~5号关节电机的扭矩，使机械臂在当前位置保持静止

        Args:
            pay_load: 负载重量，单位kg
            F: 末端在全局坐标系坐标轴方向上的受力，单位kg
        Returns:
            1~5号关节电机扭矩
        Raises:
            无
        """

        servo_angle_list = self.read_joints()  # 读取机械臂关节电机角度
        model_angle_list = self.servo_to_model(servo_angle_list=servo_angle_list)  # 将机械臂关节电机角度转换成关节模型角度
        M_model = self.forward_kinematics_torque(angle_list=model_angle_list, pay_load=pay_load, F=F).copy()  # 调用静力学函数
        M_servos = M_model.copy()  # 将静力学得到的关节模型力矩赋值给关节电机
        for i in range(len(M_model)):
            if i < 2:
                M_servos[i] = self.P1_list[i] * M_model[i] * 0.8 + 0  # 模型力矩与实际力矩转换，其中的转换系数可调节，考虑减速器摩擦，以便在保持姿态的同时保持力
            elif i == 2:
                M_servos[i] = self.P1_list[i] * M_model[i] * 0.8 + 0  # 模型力矩与实际力矩转换，其中的转换系数可调节，考虑减速器摩擦，以便在保持姿态的同时保持力
            elif i == 3:
                M_servos[i] = self.P1_list[i] * M_model[i] * 1.6 + 0  # 模型力矩与实际力矩转换，其中的转换系数可调节，考虑减速器摩擦，以便在保持姿态的同时保持力
            elif i == 4:
                M_servos[i] = self.P1_list[i] * M_model[i] * 0.6 + 0  # 模型力矩与实际力矩转换，其中的转换系数可调节，考虑减速器摩擦，以便在保持姿态的同时保持力
            else:
                M_servos[i] = self.P1_list[i] * M_model[i] * 1 + 0  # 模型力矩与实际力矩转换，其中的转换系数可调节，考虑减速器摩擦，以便在保持姿态的同时保持力
        print(
            "当前负载及外力条件下机械臂要保持静止则，1号电机扭力为: " + str(M_servos[0]) + "; 2号电机扭力为： " + str(M_servos[1]) + "; 3号电机扭力为： " + str(
                M_servos[2]) + "; 4号电机扭力为: " + str(M_servos[3]) + "; 5号电机扭力为: " + str(M_servos[4]))
        self.set_torques(id_list=self.ID_list[:5], torque_list=M_servos, mode=0)  # 使用多个关节电机力矩控制函数

    def zero_force_pull(self, pay_load=0):
        while 1:
            self.set_torques_for_pose_hold(pay_load=pay_load)
            print(self.detect_joints())

    def tutorial_program(self, pay_load=0, F=[0, 0, 0]):
        """在维持零力拖动的情况下记录机械臂连续姿态信息，以实现示教编程

        Args:
            pay_load: 负载重量，单位kg
            F: 末端在全局坐标系坐标轴方向上的受力，单位kg
        Returns:
            示教过程中，1~6号关节电机角度组成的列表的列表
        Raises:
            无
        """

        self.servo_angle_list_list = []
        start = time.time()
        while 1:
            servo_angle_list = self.read_joints()  # 读取并记录关节电机角度
            self.servo_angle_list_list.append(servo_angle_list)  # 将读取到的角度添加到关节电机角度列表的列表中
            model_angle_list = self.servo_to_model(servo_angle_list=servo_angle_list)  # 将读取到的关节电机角度转换成关节模型角度
            M_model = self.forward_kinematics_torque(angle_list=model_angle_list, pay_load=pay_load, F=F).copy()
            M_servos = M_model.copy()
            for i in range(len(M_model)):
                if i < 2:
                    M_servos[i] = self.P1_list[i] * M_model[i] * 0.8 + 0  # 模型力矩与实际力矩转换，其中的转换系数可调节，考虑减速器摩擦，以便在保持姿态的同时保持力
                elif i == 2:
                    M_servos[i] = self.P1_list[i] * M_model[i] * 0.8 + 0  # 模型力矩与实际力矩转换，其中的转换系数可调节，考虑减速器摩擦，以便在保持姿态的同时保持力
                elif i == 3:
                    M_servos[i] = self.P1_list[i] * M_model[i] * 1.6 + 0  # 模型力矩与实际力矩转换，其中的转换系数可调节，考虑减速器摩擦，以便在保持姿态的同时保持力
                elif i == 4:
                    M_servos[i] = self.P1_list[i] * M_model[i] * 0.6 + 0  # 模型力矩与实际力矩转换，其中的转换系数可调节，考虑减速器摩擦，以便在保持姿态的同时保持力
                else:
                    M_servos[i] = self.P1_list[i] * M_model[i] * 1 + 0  # 模型力矩与实际力矩转换，其中的转换系数可调节，考虑减速器摩擦，以便在保持姿态的同时保持力
            self.set_torques(id_list=self.ID_list[:5], torque_list=M_servos, mode=0)
            self.tutorial_t = time.time() - start
            print(self.servo_angle_list_list)
            print('n: ', len(self.servo_angle_list_list))
            print('t: ', self.tutorial_t)

    def tutorial_do(self, t=0):
        """执行在维持零力拖动的情况下记录机械臂连续姿态（轨迹）

        Args:
            t: 轨迹执行的大致时间
        Returns:
            无
        Raises:
            无
        """
        self.clear_uart()  # 情况串口残存数据
        if t <= 0:
            t = self.tutorial_t
        n = len(self.servo_angle_list_list)  # 记录动作序列的长度
        if t >= n * self.bit_time:  # 轨迹要求的执行时间大于系统发送 n 个动作指令（所有轨迹点）所需的时间
            print('t >= n * bit_time')  # 提示
            bit_wideth = n / t / 2  # 指令发送带宽的一半
            servo_angle_list = self.read_joints()
            time.sleep(0.1)
            for i in range(len(self.ID_list)):
                self.set_angle(id_num=self.ID_list[i], angle=servo_angle_list[i], speed=1, param=1, mode=2)
            self.set_angles(self.ID_list, servo_angle_list, 1, 1, 2)  # 将关节设置为位置闭环控制模式
            self.set_angles(self.ID_list, self.servo_angle_list_list[0], 10, 10, 1)
            self.pose_done()
            self.set_angles(self.ID_list, self.servo_angle_list_list[0], 20, bit_wideth,
                            0)  # 采用轨迹跟踪模式发送首条位置指令，根据轨迹跟踪模式要求，bit_wideth 需为实际指令发送带宽的一半
            start = time.time()  # 记录指令发送开始时间
            while (time.time() - start) < (t / n):  # 将轨迹执行时间按照点的数量均分
                time.sleep(0.001)
            bit_wideth1 = 1 / (time.time() - start) / 2  # 计算在 t>n * self.bit_time 情况下的指令发送频率的一半
            start_time = time.time()
            for k in range(n):
                start = time.time()
                self.set_angles(self.ID_list, self.servo_angle_list_list[k], 20, bit_wideth1,
                                0)  # 采用轨迹跟踪模式，根据轨迹跟踪模式要求，bit_wideth1 需为实际指令发送带宽的一半
                while (time.time() - start) < (t / n):
                    time.sleep(0.001)
                bit_wideth1 = 1 / (time.time() - start) / 2  # 时刻监控在 t>n 情况下单条指令发送的时间
            print('运行时长：', time.time() - start_time)
        else:
            print('t < n * bit_time')  # 提示轨迹要求的时间小于系统发送 n 个指令（所有轨迹点）所需的时间
            bit_wideth = 1 / self.bit_time / 2  # 指令发送频率取初始化阶段获得的系统单条指令发送所需时间的倒数，再取其一半作为 set_angles() 函数参数
            for k in range(n):
                self.set_angles(self.ID_list, self.servo_angle_list_list[k], 20, bit_wideth, 0)  # 使用轨迹追踪模式控制机械臂运动

    # 五、阻抗控制函数
    def impedance_control_joints(self, speed=0, tff=0, kp=1, kd=0.1):
        """设置机械臂每个关节的阻抗系数

        Args:
            vel: 关节目标速度(r/min)
            tff: 前馈扭矩(Nm)
            kp: 刚度系数(rad/Nm)
            kd: 阻尼系数(rad/s/Nm)
        Returns:
            1~5号电机扭矩
        Raises:
            无
        """

        for j in range(len(self.ID_list)):  # 提前 pose_list 中的最后一组关节模型角度列表（即机械臂最后姿态），设置此姿态下关节阻抗系数
            self.impedance_control(id_num=self.ID_list[j],
                                   angle=self.model_to_servo(self.pose_list[len(self.pose_list) - 1])[j],
                                   speed=speed, tff=tff, kp=kp, kd=kd)

    def init_pid(self, pids=[[10, 5, 0.55], [10.56, 4.95, 0.391], [10.56, 4.95, 0.391], [10, 9, 0.5], [12, 5, 0.1],
                             [12, 5, 0.096]]):
        '''初始化机器人pid
        Args:
            pids: 各关节 pid 组成的列表，其中
                pids[0] 代表 1 号关节
                pids[1] 代表 2 号关节
                pids[2] 代表 3 号关节
                pids[3] 代表 4 号关节
                pids[4] 代表 5 号关节
                pids[5] 代表 6 号关节
        Returns:
            无
        Raises:
            无
        '''

        for i in range(6):
            self.set_pid_joint(joint_num=i + 1, P=pids[i][0], I=pids[i][1], D=pids[i][2])

    # 六、功能辅助函数
    def save_pose(self, ud_temp=0):
        """求解并保存当前机器人姿态下各个关节模型角度，并将角度制的角度列表添加进 pose_list

        Args:
            ud_temp: (up/down)用来选择反解中的第几种解：
                ud_temp = 0 时(默认值)，alpha构型，第三个关节在末端上方；
                ud_temp = 1 时，beta构型，第三个关节在末端点下方.
        Returns:
            True: 反解过程顺利通过时，返回True
            False: 机械臂末端超出工作空间、机械臂末端超出安全范围、关节模型角度超出安全范围
        Raises:
            无
        """

        theta_bk = self.theta.copy()  # 记录此前的关节模型角度（弧度制）
        if not self.inverse_kinematics(pl_temp=self.pl, theta_P_Y_R=self.theta_P_Y_R, ud=ud_temp):
            print("机械臂末端超出了工作空间\n")
            return False
        safe_low = -150  # 考虑桌面，机械臂末端关节最低安全位置
        safe_low_z = -200  # 考虑桌面，手爪最低安全位置
        if self.show_joint_z_position(4) < safe_low - 0.1 or self.show_joint_z_position(6) < safe_low - 0.1 or self.pl[
            2] < safe_low_z:  # 检查末端是否超出最低安全位置，0.1为考虑浮点数计算误差所必须
            print('警告！警告！警告！不能执行，否则末端关节或手爪超出最低安全位置：', safe_low)
            print('请使用 detect_joint_z_position() 函数检查各关节当前z坐标')
            self.theta = theta_bk.copy()  # 若超出安全位置，则恢复此前的关节模型角度（弧度制）
            return False
        pose = self.theta[:]  # 将关节模型角度赋值给pose（弧度制）
        for i in range(len(pose)):
            pose[i] *= RAD_DEG  # 将弧度制转换成角度制
            if pose[i] < self.MIN_list[i] or pose[i] > self.MAX_list[i]:  # 将关节模型角度（角度制）与关节模型角度最大和最小值进行比较
                print("警告！警告！警告！不能执行，姿态序号为：" + str(len(self.pose_list) + 1) + "中第" + str(i + 1) + "个关节目标模型角度值为 " + str(
                    pose[i]) + "，超出该关节活动范围[" + str(self.MIN_list[i]) + ", " + str(self.MAX_list[i]) + "]")
                print('请使用 detect_pose() 函数检查各关节当前位姿')
                self.theta = theta_bk.copy()  # 若关节模型角度超出安全范围，则恢复此前记录的关节模型角度（弧度制）
                return False
        self.pose_list.append(pose)  # 将关节模型角度列表添加进 pose_list
        self.tutorial_t_list.append(0)  # 设置关节模型角度对应的姿态与下一姿态之间的时间间隔为 0
        return True

    def do_pose(self, speed=10, param=10, mode=1, o_r=0, n=0):
        """依次执行机器人通过示教保存pose；

        Args:
            speed: 当前姿态被执行时转速最快关节的转动速度
            mode: 控制电机转动模式，0为轨迹追踪模式，1为梯形轨迹模式，2为前馈模式
            o_r: 用来选择执行的顺序，
                o_r =0：从前往后执行
                o_r =1: 从后往前执行
            n: 用来控制执行的细节，如果n=0，这执行所有保存的姿态；如果n>0 则执行第n个姿态
        Returns:
            无
        Raises:
            无
        """
        self.do_motion(speed=speed, param=param, mode=mode, o_r=o_r, n=n, flg=1)

    # 动作执行
    def do_motion(self, speed=1.0, param=10, mode=1, o_r=0, n=0, flg=0):
        """依次执行机器人此前保存的pose；

        Args:
            speed: 当前姿态被执行时转速最快关节的转动速度
            mode: 控制电机转动模式，0为轨迹追踪模式，1为梯形轨迹模式，2为前馈模式
            o_r: 用来选择执行的顺序，
                o_r =0：从前往后执行
                o_r =1: 从后往前执行
            n: 用来控制执行的细节，如果n=0，这执行所有保存的姿态；如果n>0 则执行第n个姿态
            flg: 用于判定是否需要确保每一个姿态都执行到位，1代表是，0代表否
        Returns:
            无
        Raises:
            无
        """

        # 最大速度限制，保证安全性和动作一致性
        if speed > self.max_speed:
            speed = self.max_speed
        try:
            if n == 0:
                if o_r == 0:
                    for i in range(len(self.pose_list)):
                        self.set_angles(self.ID_list, self.model_to_servo(self.pose_list[i]), speed, param,
                                        mode)  # 依次执行 pose_list 中的pose（姿态）
                        if flg == 1:
                            self.pose_done()  # 等待并监控动作执行结束
                        time.sleep(self.tutorial_t_list[i])  # 设置每个姿态的停留时长
                else:
                    for j in range(len(self.pose_list)):
                        i = len(self.pose_list) - 1 - j  # 倒序执行
                        self.set_angles(self.ID_list, self.model_to_servo(self.pose_list[i]), speed, param,
                                        mode)  # 倒序执行 pose_list 中的pose（姿态）
                        if flg == 1:
                            self.pose_done()  # 等待并监控动作执行结束
                        time.sleep(self.tutorial_t_list[i])  # 设置每个姿态的停留时长
            else:
                self.set_angles(self.ID_list, self.model_to_servo(self.pose_list[n - 1]), speed, param, mode)
                if flg == 1:
                    self.pose_done()  # 等待并监控动作执行结束
                time.sleep(self.tutorial_t_list[n - 1])  # 设置每个姿态的停留时长
        except Exception as result:
            print('检测出异常 in do_motion{}'.format(result))

    # 关节角度范围设置
    def range_init(self, max_list=[], min_list=[]):
        """设置机械臂各个关节的模型角度范围[min, max]

        Args:
            max_list: 由机械臂所有关节转动范围最大值组成的列表
            min_list: 由机械臂所有关节转动范围最小值组成的列表
        Returns:
            无；
        Raises:
            无；
        """

        servo_number = len(self.ID_list)
        if len(max_list) == servo_number:
            self.MAX_list = max_list[:]
        else:
            self.MAX_list = [90, 215, 159, 153, 180, 180]
        if len(min_list) == servo_number:
            self.MIN_list = min_list[:]
        else:
            self.MIN_list = [-36, -15, -159, -153, -180, -180]
        max_servo = self.model_to_servo(model_angle_list=self.MAX_list)
        min_servo = self.model_to_servo(model_angle_list=self.MIN_list)
        for i in range(self.JOINT_NUMBER):
            if self.P1_list[i] == -1:
                exc = max_servo[i]
                max_servo[i] = min_servo[i]
                min_servo[i] = exc
        for i in range(self.JOINT_NUMBER):
            self.set_angle_range(id_num=self.ID_list[i], angle_min=min_servo[i], angle_max=max_servo[i])

    # 往 pose_list 中新增一个姿态
    def add_pose(self, t=0):
        """读取到的当前机器人所有关节角度并转换成模型角度，然后保存到pose_list中。

        Args:
            t: 该姿态与下一姿态之间的时间间隔（即姿态保持时间）
        Returns:
            True or False ；
        Raises:
            无；
        """

        servo_list = self.read_joints()  # 读取关节电机角度
        if servo_list != False:
            self.pose_list.append(self.servo_to_model(servo_angle_list=servo_list))  # 将关节电机角度转换成关节模型角度，并添加进 pose_list 中
            self.tutorial_t_list.append(t)  # 设置该姿态与下一姿态之间的时间间隔（即姿态保持时间）
            print("保存当前姿态成功！ 当前pose_list中共有" + str(len(self.pose_list)) + "个pose")
            return True
        else:
            print("当前姿态读取失败，请再试一次！")
            return False

    #
    # 读取关节电机角度
    def read_joints(self):
        """读取机器人当前姿态下各关节的电机角度。

        Args:
            无。
        Returns:
            False or servo_list.
            若有一个或者多个关节读取出错，则返回False, 反之，如果一切正常，则返回所有关节角度组成的列表。
        Raises:
            无。
        """

        servo_list = []
        flag = []
        self.clear_uart()  # 先清除一下串口中的残留数据
        for i in range(len(self.ID_list)):
            servo = self.get_angle(id_num=self.ID_list[i])  # 使用一体化关节的 get_state() 函数依次读取关节角度和速度
            if servo != False:
                servo_list.append(servo)  # 将读取到的角度添加进 servo_list
            else:
                servo = self.get_angle(id_num=self.ID_list[i])  # 如果首次读取失败就再读一次
                if servo != False:
                    servo_list.append(servo)  # 将读取到的角度添加进 servo_list
                else:
                    flag.append(self.ID_list[i])  # 如果持续读取失败则记录读取失败的关节电机编号
                    print("ID号为：" + str(self.ID_list[i]) + "的电机读取角度失败！")
        if len(flag) != 0:
            return False
        return servo_list

    def read_joints_p_v(self):
        """读取机器人当前姿态下各关节的角度和转速。

        Args:
            无。
        Returns:
            False or servo_list.
            若有一个或者多个关节读取出错，则返回False, 反之，如果一切正常，则返回所有关节角度和速度组成的列表。
        Raises:
            无。
        """
        servo_p_v_list = []
        flag = []
        for i in range(len(self.ID_list)):
            servo = self.get_state(id_num=self.ID_list[i])  # 使用一体化关节的 get_state() 函数依次读取关节角度和速度
            if servo != False:
                servo_p_v_list.append(servo)  # 将读取到的角度和速度添加进 servo_p_v_list
            else:
                servo = self.get_state(id_num=self.ID_list[i])  # 如果首次读取失败就再读一次
                if servo != False:
                    servo_p_v_list.append(servo)  # 将读取到的角度和速度添加进 servo_p_v_list
                else:
                    flag.append(self.ID_list[i])  # 如果持续读取失败则记录读取失败的关节电机编号
                    print("ID号为：" + str(self.ID_list[i]) + "的电机取角度和速度失败！")
        if len(flag) != 0:
            return False
        return servo_p_v_list

    # 角度转换
    def servo_to_model(self, servo_angle_list=[]):
        """将机械臂关节电机角度转换成模型角度。
        将当前姿态的舵机角度转换成模型角度，,两者之间的变换关系为
        舵机角度=P1*模型角度+P2，其中P1表示两者的正方向是否相同，相同为1，相反为-1
        第二个参数为差值项，装配时将初始模型角度和舵机角度代入求出。

        Args:
            servo_angle_list: 机器人某个姿态下所有关节电机角度（角度制）组成的列表
        Returns:
            该姿态下所有关节模型角度组成的列表；
        Raises:
            无
        """

        if servo_angle_list != False:
            model_angle_list = servo_angle_list[:]  # 目的是让 servo_angle_list 拥有与 model_angle_list 相同的长度
            for i in range(len(servo_angle_list)):
                model_angle_list[i] = round((servo_angle_list[i] - self.P2_list[i]) / self.P1_list[i],
                                            1)  # 通过关节模型角度与关节电机角度之间的关系进行转换
            return model_angle_list
        else:
            return False

    def model_to_servo(self, model_angle_list=[]):
        """将机器人关节模型角度转换成关节电机角度。

        Args:
            model_angle_list: 机器人某个姿态下所有关节模型角度（角度制）组成的列表
        Returns:
            该姿态下所有关机电机角度组成的列表；
        Raises:
            无
        """

        servo_angle_list = model_angle_list[:]  # 目的是让 servo_angle_list 拥有与 model_angle_list 相同的长度
        for i in range(len(model_angle_list)):
            servo_angle_list[i] = self.P1_list[i] * servo_angle_list[i] + self.P2_list[i]  # 通过关节模型角度与关节电机角度之间的关系进行转换
        return servo_angle_list

    #
    # 将所有关节设置成待机模式
    def free(self):
        """将机械臂所有关节电机设置成待机模式，以便节省能源。若要恢复使用必须先调用 lock()函数。
        Args:
            无；
        Returns:
            无；
        Raises:
            无；
        """

        self.set_mode(0, 1)

    #
    # 将所有关节固定在当前位置
    def lock(self):
        """将机械臂所有关节电机设置成锁死模式，保持当前姿态。

        Args:
            无；
        Returns:
            无；
        Raises:
            无；
        """

        self.set_mode(0, 2)

    #
    # 删除或清空pose_list里的动作
    def clear_pose(self, n=0):
        """清除机器人pose_list中保存的姿态。
        当需要重新计算一个轨迹或新姿态时，如果不想保留原有的轨迹或姿态，可以调用此函数将原有保存的所有姿态清空。

        Args:
            n: 指定需要清除的pose编号
                n=0: 表示清空pose_list中所有pose
                n>0: 删除第n个pose
                n<0, 删除倒数第n个pose
        Returns:
            无。
        Raises:
            无。
        """

        LEN = len(self.pose_list)
        if n == 0:
            self.pose_list = []  # 直接置空
            self.tutorial_t_list = []  # 直接置空
        elif n > 0:
            if n <= LEN:
                del self.pose_list[n - 1]  # 删除第 n 个姿态
                del self.tutorial_t_list[n - 1]  # 删除第 n 个姿态保持的时间
            else:
                print("元素引用序号超出pose_list长度")
        else:
            if LEN + n >= 0:
                del self.pose_list[LEN + n]  # 删除倒数第 n 个姿态
                del self.tutorial_t_list[LEN + n]  # 删除倒数第 n 个姿态保持的时间
            else:
                print("元素引用序号超出pose_list长度")

    def pose_done(self):
        """检查并等待机械臂末端运动到指定位置和姿态。

        Args:
            无。
        Returns:
            无
        Raises:
            无。
        """

        self.positions_done(self.ID_list)

    def get_property(self, joint_num=1, property=''):
        """查看机器人关节参数。

        Args:
            joint_num：需要查看参数的关节 ID 号。
             property：参数名称，详见 parameter_interface.py。
        Returns:
             property：读取的参数。
                False：读取失败。
        Raises:
            无。
        """
        if joint_num < 1 or joint_num > 6:
            print("请输入正确的关节编号：1~6")
            return False
        else:
            property = self.read_property(id_num=self.ID_list[joint_num - 1], property=property)
            print(property)
            return property

    def get_pid_joint(self, joint_num=1):
        """查看机器人关节 PID。

        Args:
            joint_num：需要查看 PID 的关节 ID 号。
        Returns:
              pid：关节 PID 组成的列表。
            False：读取失败。
        Raises:
            无。
        """
        if joint_num < 1 or joint_num > 6:
            print("请输入正确的关节编号：1~6")
            return False
        else:
            pid = self.get_pid(id_num=self.ID_list[joint_num - 1])
            return pid

    def set_property(self, joint_num=1, property='', value=0):
        """设置机器人关节参数。

        Args:
            joint_num：需要设置参数的关节 ID 号。
             property：参数名称，详见 parameter_interface.py。
                value：参数值。
        Returns:
            True：设置成功。
            Fals：设置失败。
        Raises:
            无。
        """
        if joint_num < 1 or joint_num > 6:
            print("请输入正确的关节编号：1~6")
            return False
        else:
            self.write_property(id_num=self.ID_list[joint_num - 1], property=property, value=value)
            time.sleep(0.1)
            value_return = self.read_property(id_num=self.ID_list[joint_num - 1], property=property)
            if value_return == value:
                print("机械臂第 ", joint_num, " 号关节的 ", str(property) + " 修改为：", value_return)
                return True
            else:
                print("机械臂第 ", joint_num, " 号关节的 ", str(property) + " 修改失败，请重试")
                return False

    def set_pid_joint(self, joint_num=1, P=10, I=10, D=10):
        """设置机器人关节 PID。

        Args:
            joint_num：需要查设置 PID 的关节 ID 号。
             P：PID 的 P 值。
             I：PID 的 I 值。
             D：PID 的 D 值。
        Returns:
            True：设置成功。
            Fals：设置失败。
        Raises:
            无。
        """
        if joint_num < 1 or joint_num > 6:
            print("请输入正确的关节编号：1~6")
            return False
        else:
            self.set_pid(id_num=self.ID_list[joint_num - 1], P=P, I=I, D=D)
            time.sleep(0.1)
            pid = self.get_pid(id_num=self.ID_list[joint_num - 1])
            time.sleep(0.1)
            if abs(pid[0] - P) < 0.1 and abs(pid[1] - I) < 0.1 and abs(pid[2] - D) < 0.1:
                print("机械臂第 ", joint_num, " 号关节的 PID 修改为：", pid)
            else:
                print("机械臂第 ", joint_num, " 号关节的 PID 修改失败，请重试")
            return True
