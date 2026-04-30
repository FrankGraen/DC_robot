import time
import math
import numpy as np
import arm_six_axis as am

gripper_offset = np.array([0, 0, 100])  # 例如，手爪中心点在末端执行器上方10mm
gripper_orientation = np.array([0, 0, 0])  # 例如，手爪姿态与末端执行器一致


class gripper(am.arm):
    id_num = 7  # 配置手爪ID号与控制手爪的一体化关节ID号一致
    d = 10  # 内部齿轮直径，单位 mm
    # 源程序错误点，init函数名错误，这个地方去掉负号，将2改成20. angle = wideth / 20 / (self.d / 2) / math.pi * 180
    def __init__(self, L_p=0, L_p_mass_center=0, G_p=0, com='', uart_baudrate=115200):
        am.arm.__init__(self, L_p=L_p, L_p_mass_center=L_p_mass_center, G_p=G_p, com=com, uart_baudrate=uart_baudrate)

    def grasp(self, wideth=10, speed=10, force=3):
        '''
        :param wideth: 手爪开合宽度，单位 mm
        :param speed: 手爪开合速度，单位 mm/s
        :param force: 手爪开合力，单位 N
        :return: 无
        '''
        if wideth > 50 or wideth < 0:
            print("请输入正确的开合宽度：0~90，已将 wideth 设置为 90")
            wideth = 50
        if speed > 10 or speed <= 0:
            print("请输入正确的开合速度：0~60，已将 speed 设置为 60")
            speed = 10
        if force > 120:
            print("请输入正确的开合力：>120，已将 force 设置为 50")
            force = 50
        angle = wideth / 20 / (self.d / 2) / math.pi * 180  # 将手爪宽度转换成角度
        w = speed / (self.d / 2) / (math.pi * 2) * 60  # 将手爪开合速度转换为关节转速 r/min
        torque = force * (self.d / 2 / 1000)  # 手爪开合里转换成关节力矩 Nm
        self.set_angle_adaptive(id_num=self.id_num, angle=angle, speed=w, torque=torque)
        #self.set_angle_adaptive(id_num=self.id_num, angle=20, speed=w, torque=torque)
        return True

    def grasp_done(self):
        """检测手爪动作是否完成

        Args:
            无
        Returns:
            无
        Raises:
            无
        """
        self.position_done(id_num=self.id_num)

    def detect_wideth_grasp(self):
        angle = self.get_angle(id_num=self.id_num)
        wideth = - angle / 180 * math.pi * (self.d / 2) * 2
        return wideth

    ##新增程序

    def compute_rotation_matrix(self, pitch, yaw, roll):
        """根据欧拉角（Pitch, Yaw, Roll）计算旋转矩阵"""
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)
        roll_rad = math.radians(roll)

        # 绕Y轴的旋转矩阵（Pitch）
        R_y = np.array([
            [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
            [0, 1, 0],
            [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
        ])

        # 绕Z轴的旋转矩阵（Yaw）
        R_z = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0],
            [0, 0, 1]
        ])

        # 绕X轴的旋转矩阵（Roll）
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(roll_rad), -math.sin(roll_rad)],
            [0, math.sin(roll_rad), math.cos(roll_rad)]
        ])

        # 总旋转矩阵顺序：R_x @ R_z @ R_y（需与实际机械臂定义一致！）
        return R_x @ R_z @ R_y

    def calculate_hand_pose(self, pl, theta_P_Y_R):
        # 将机械臂末端的位置转换为numpy数组
        pl = np.array(pl)
        pitch, yaw, roll = theta_P_Y_R

        # 将角度转换为弧度
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)
        roll_rad = math.radians(roll)

        # 绕Y轴的旋转矩阵（Pitch）
        R_y = np.array([
            [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
            [0, 1, 0],
            [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
        ])

        # 绕Z轴的旋转矩阵（Yaw）
        R_z = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0],
            [0, 0, 1]
        ])

        # 绕X轴的旋转矩阵（Roll）
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(roll_rad), -math.sin(roll_rad)],
            [0, math.sin(roll_rad), math.cos(roll_rad)]
        ])

        # 计算总旋转矩阵：顺序为R_x @ R_z @ R_y
        R_total = R_x @ R_z @ R_y

        # 计算全局坐标系中的偏移
        global_offset = R_total @ gripper_offset

        # 计算机械臂手爪末端的全局位置
        hand_position = pl + global_offset

        # 姿态保持不变
        hand_orientation = [pitch, yaw, roll]

        return hand_position, hand_orientation

    def hand_to_arm_pose(self, hand_pos, hand_orientation):
        """将手爪末端位姿转换为机械臂末端位姿"""
        # 计算机械臂末端的旋转矩阵
        R_arm = self.compute_rotation_matrix(*hand_orientation)

        # 将局部偏移转换到全局坐标系
        global_offset = R_arm @ gripper_offset

        # 计算机械臂末端位置
        arm_pos = hand_pos - global_offset

        # 机械臂末端姿态与手爪末端相同
        arm_orientation = hand_orientation
        print(arm_pos + arm_orientation)
        return arm_pos, arm_orientation

    def hand_to_arm_theta(self, hand_pos, hand_orientation):

        # 计算机械臂末端的目标位姿
        arm_target_pos, arm_target_ori = self.hand_to_arm_pose(hand_pos, hand_orientation)

        # 将目标姿态转换为旋转矩阵（用于逆运动学）
        R_target = self.compute_rotation_matrix(*arm_target_ori)

        # 初始猜测值（需合理设置）
        initial_guess = np.radians([0, 30, -60, 0, 90, 0])

        # 计算逆解
        theta_solution = self.inverse_kinematics(pl_temp=arm_target_pos, theta_P_Y_R=arm_target_ori, ud=0)
        print("关节角度（度）:", np.degrees(self.theta))
        return self.theta
    # 示例D-H参数（需根据实际机械臂修改）
    DH_PARAMS = [
        {'theta': 0, 'd': 0.1, 'a': 0, 'alpha': np.pi / 2},  # Joint 1
        {'theta': 0, 'd': 0, 'a': 0.5, 'alpha': 0},  # Joint 2
        {'theta': 0, 'd': 0, 'a': 0.5, 'alpha': 0},  # Joint 3
        {'theta': 0, 'd': 0.1, 'a': 0, 'alpha': np.pi / 2},  # Joint 4
        {'theta': 0, 'd': 0, 'a': 0, 'alpha': -np.pi / 2},  # Joint 5
        {'theta': 0, 'd': 0.1, 'a': 0, 'alpha': 0},  # Joint 6
    ]
