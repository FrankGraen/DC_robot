import time
import math
import arm_robot as robot  # 忽略这里的报错

l_p = 0  # 工具参考点到电机输出轴表面的距离，单位mm（所有尺寸参数皆为mm）
l_p_mass_center = 0  # 工具（负载）质心到 6 号关节输出面的距离
G_p = 0  # 负载重量，单位kg，所有重量单位皆为kg
uart_baudrate = 115200  # 串口波特率，与CAN模块的串口波特率一致，（出厂默认为 115200，最高460800）
com = 'COM4'  # 在这里输入 COM 端口号
# com='/dev/ttyUSB0' # 在 jetgggson nano（ubuntu）下控制机器人，相应的输入连接的串口
# com='/dev/ttyAMA0' # 在树莓派（raspbian）下控制机器人，相应的输入连接的串口
# com='/dev/cu.usbserial-110' # 在苹果电脑 mac 下控制机器人，相应地输入串口
# # 机械臂对象初始化函数
ro = robot.arm_robot(L_p=l_p, L_p_mass_center=l_p_mass_center, G_p=G_p, com=com, uart_baudrate=uart_baudrate)
ro.detect_pose()

#ro.zero_force_pull(pay_load=0)
#ro.set_torques_for_pose_hold(pay_load=pay_load)
# ro.free()
# ro.lock
'''''''控制机械臂六个关节运动到指定模型角度'''''''
#ro.set_arm_joints(angle_list=[0, 80, 90, 90, 0, 0], speed=2.0) # 对应
# # *********************************************************

'''''''查看当前关节模型角度函数(通过回读一体化关节角度计算)'''''''
# ro.detect_joints()
# # *********************************************************

'''''''查看当前关节电机角度函数(通过回读一体化关节角度计算)'''''''
# ro.read_joints()
# # *********************************************************

'''''''控制机械臂各电机到指定电机角度'''''''
#ro.set_angles(id_list=[1, 2, 3], angle_list=[0, 0, 0], speed=2, param=10, mode=1):
# # *********************************************************

'''''''计算机械臂末端位置与姿态'''''''
#ro.forward_kinematics_pose(angle_list=[0, 80, 90, 90, 0, 0]) # angle_list为模型角度
# # *********************************************************

'''''''计算机械臂末端位置与姿态'''''''
#ro.inverse_kinematics(pl_temp=[0, 0, 0], theta_P_Y_R=[0, 0, 0], ud=0)
#print(ro.theta)
# # *********************************************************

'''''''查看各关节电机角度'''''''
# ro.get_angle(id_num=1)  #查看指定关节电机角度
# ro.read_joints()  #查看所有关节电机角度
# # *********************************************************

'''''''不同的运动模式，所匹配的 PID 略有不同，正式运动控制前建议优化一下各关节PID，以下为进行点到点运动控制时的一组较优 PID，用户可进一步优化'''''''
# ro.set_pid_joint(1, P=10, I=5, D=0.55)
# ro.set_pid_joint(2, P=10.56, I=4.95, D=0.39)
# ro.set_pid_joint(3, P=10.56, I=4.95, D=0.39)
# ro.set_pid_joint(4, P=10, I=9, D=0.5)
# ro.set_pid_joint(5,  P=12, I=5, D=0.1)
# ro.set_pid_joint(6,  P=12, I=5, D=0.096)
# # *********************************************************

'''''''回读关节 PID 参数函数'''''''
# ro.get_pid_joint(joint_num=1)
#
'''''''设置关节 PID 参数函数'''''''
# ro.set_pid_joint(joint_num=1, P=10, I=5, D=0.55)
#
'''''''回读关节参数函数'''''''
# ro.get_property(joint_num=1, property='dr.config.gear_ratio') # 关节 减速比
#
'''''''设置关节参数函数'''''''
# ro.set_property(joint_num=1, property='dr.controller.config.speed_limit', value=300) # 关节最大限制转速，此处转速为电机转速，输出端需要乘以减速比
#

'''''''查看当前位姿函数'''''''
# ro.show_pose()
# # *********************************************************

'''''''查看机械臂当前各运动关节并返回指定关节的z坐标函数'''''''
# n = ro.show_joint_z_position(6)
# print(n)
# # *********************************************************

'''''''查看当前机械臂平面内各运动关节并返回指定关节的x坐标函数'''''''
# n = ro.show_joint_x_position(6)
# print(n)
# # *********************************************************

'''''''查看当前位姿函数(通过回读一体化关节角度计算)'''''''
# ro.detect_pose()
# # *********************************************************



'''''''运动到指定位置和姿态函数'''''''
# ro.set_arm_pose(pl_temp=[ro.L[0] + ro.L[1] + ro.L[2], -ro.L[4] + ro.L[3], 0], theta_P_Y_R=[0, 0, 0], speed=10, param=10, mode=1)
# # *********************************************************

'''''''手爪开合函数（当机械臂末端安装大然手爪后可用，手爪关节ID号需为7）'''''''
# ro.grasp(wideth=10, speed=10, force=50)
# ro.grasp_done() # # 检测手爪开合是否到位
# # *********************************************************

'''''''运动到指定位置函数'''''''
# ro.set_arm_position(pl_temp=[ro.L[0] + ro.L[1] + ro.L[2], -ro.L[4] + ro.L[3], 0], speed=10, param=10, mode=1)
# # *********************************************************

'''''''运动到指定姿态函数'''''''
# ro.set_arm_P_Y_R(theta_P_Y_R=[0, 0, 0], speed=10, param=10, mode=1)
# # *********************************************************

'''''''运动到相对位置和姿态函数'''''''
# ro.set_arm_relative_pose(pl_temp=[-20, 0, 0],  theta_P_Y_R=[0, 0, 0], speed=10, param=10, mode=1)
# # *********************************************************

'''''''运动到相对位置函数'''''''
# ro.set_arm_relative_position(pl_temp=[10, 0, 0], speed=10, param=10, mode=1)
# # *********************************************************

'''''''运动到相对姿态函数'''''''
# ro.set_arm_relative_P_Y_R(theta_P_Y_R=[0, 10, 0], speed=10, param=10, mode=1)
# # *********************************************************

'''''''''''''''''''''轨迹跟踪函数'''''''''''''''''''''
'''''''画正方形'''''''
# def draw_rectangle(pl=[283, 0, -126.5], l=30, h=30):
#     ''''在水平面上画正方形
#     pl: 长方形左上角坐标（起始点），其中pl[2]代表作图平面与全局坐标系z轴的焦点的z坐标
#     l: 宽度
#     h: 高度
#     '''
#     n= 50 # 每条边分割的点数（数量越多画得越慢）
#     l_delta = l/n
#     h_delta = h/n
#     pl_list = []
#     pl_list.append(pl)
#     l1 = pl[1]
#     for i in range(1, n+1):
#         pl_temp = [pl[0], pl[1]-i*l_delta, pl[2]]
#         pl_list.append(pl_temp)
#     print(pl_temp)
#     for i in range(1, n+1):
#         pl_temp1 = [pl_temp[0]-i*h_delta, pl_temp[1], pl_temp[2]]
#         pl_list.append(pl_temp1)
#     print(pl_temp1)
#     for i in range(1, n+1):
#         pl_temp2 = [pl_temp1[0], pl_temp1[1]+i*l_delta, pl_temp1[2]]
#         pl_list.append(pl_temp2)
#     print(pl_temp2)
#     for i in range(1, n+1):
#         pl_temp3 = [pl_temp2[0]+i*h_delta, pl_temp2[1], pl_temp2[2]]
#         pl_list.append(pl_temp3)
#     print(pl_temp3)
#     print(pl_list)
#     return pl_list
#
#
# pl_list = draw_rectangle(pl=[300, 100, 50], l=200, h=150) #

# ########轨迹运动之前可先调整一下 pid，以获得更好的曲线平滑度
# ro.set_pid_joint(1, P=10, I=5, D=0.55)
# ro.set_pid_joint(2, P=10.56, I=4.95, D=0.39)
# ro.set_pid_joint(3, P=10.56, I=4.95, D=0.39)
# ro.set_pid_joint(4, P=10, I=9, D=0.5)
# ro.set_pid_joint(5,  P=12, I=5, D=0.1)
# ro.set_pid_joint(6,  P=12, I=5, D=0.096)

# ########控制机械臂末端连续运动到多个指定位置和姿态函数(必须单独一次性使用)
# ro.set_arm_poses(pls_temp=pl_list, theta_P_Y_Rs_temp=[[0, 90, 0]], t=10) # 控制机械臂末端连续运动到多个指定位置和姿态函数(必须单独一次性使用)
# ro.set_arm_poses_curve_pre(pls_temp=pl_list, theta_P_Y_Rs_temp=[[0, 90, 0]]) # 预设机械臂末端轨迹函数
# ro.set_arm_poses_curve_start_point(10) # 运动到轨迹起始位置函数
# while True:
#     ro.set_arm_poses_curve_do(5) # 末端轨迹执行函数，参数为大致运行时间
# # ************************************************************

'''''''画椭圆'''''''
# def draw_ellipse(pl=[200, 0, 0], a=10, b=20):
#     ''''椭圆方程: (x-pl[0])²/a²+(y-pl[1])²/b²=1
#         pl: 椭圆中心点坐标（起始点）,其中pl[2]代表作图水平面在z轴上的位置
#         a: x轴对应轴长
#         b: x轴对应轴长y
#         '''
#     n = 800 # 每条边分割的点数（数量越多画得越慢）, n过小会在末尾与起始之间有明显停顿
#     angle_delta = math.pi/n * 2
#     pl_list = []
#     for i in range(0, n):
#         x = pl[0] + a * math.cos(angle_delta*i)
#         y = pl[1] + b * math.sin(angle_delta*i)
#         pl_list.append([x, y, pl[2]])
#     print(pl_list)
#     return pl_list
#
#
# pl_list = draw_ellipse(pl=[300, 0, 100], a=50, b=220) # 求点
#
# # ########轨迹运动之前可先调整一下 pid，以获得更好的曲线平滑度
# # ro.set_pid_joint(1, P=10, I=5, D=0.55)
# # ro.set_pid_joint(2, P=10.56, I=4.95, D=0.39)
# # ro.set_pid_joint(3, P=10.56, I=4.95, D=0.39)
# # ro.set_pid_joint(4, P=10, I=9, D=0.5)
# # ro.set_pid_joint(5,  P=12, I=5, D=0.1)
# # ro.set_pid_joint(6,  P=12, I=5, D=0.096)
#
# # ro.set_arm_poses(pls_temp=pl_list, theta_P_Y_Rs_temp=[[0, 90, 0]], t=10)
# ro.set_arm_poses_curve_pre(pls_temp=pl_list, theta_P_Y_Rs_temp=[[0, 90, 0]]) # 预设机械臂末端轨迹函数
# ro.set_arm_poses_curve_start_point(10) # 运动到轨迹起始位置函数
# # ro.set_arm_poses_curve_do(5) # 末端轨迹执行函数，参数为大致运行时间
# while True:
#     ro.set_arm_poses_curve_do(5) # 末端轨迹执行函数，参数为大致运行时间
# # ************************************************************

