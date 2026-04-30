import DrEmpower_can as dreC
#import arm_robot as robot
com = 'com7'
uart_baudrate = 115200

L = [150, 150, 80, 58, 42] # 机械臂尺寸参数列表：[l1, l2, l3, d3, d4]，详见库函数说明
l_p_mass_center = 0 # 末端件（负载/工具）质心到 6 号关节输出面的距离
lp = 0 # 末端件（工具）中心到 6 号关节输出面的距离
pl = [0, 0, 0]
theta = [0, 0, 0, 0, 0, 0] # 6个关节角度
theta_P_Y_R = [0, 0, 0] # 3个末端姿态角
G = [0.15, 0.35, 0.15, 0.485, 0.227]  # 重量参数，单位kg，分别为杆件2、关节3、杆件3、关节4重量(两个电机)、负载重量(一个电机+实际负载)
max_list_temp = [85, 215, 149, 142, 179, 179]  # 关节模型角度最大值,1号关节目的是保护线缆，并且到达工作空间边缘；2号关节到达工作空间边缘；3、4号关节目的是不产生杆件干涉；5号关节因有滑环，不考虑；6号关节保证工作空间内全部到达
min_list_temp = [-85, -15, -149, -142, -179, -179]  # 关节模型角度最小值，1号关节目的是不装到装在桌边的竖杆（安装摄像头）；2号关节目的是在伸直的时候不打到桌子；3、4号关节目的是不产生杆件干涉；5号关节因有滑环，不考虑；6号关节保证工作空间内全部到达
dr = dreC.DrEmpower_can(com=com, uart_baudrate=uart_baudrate )
# dr.set_angle(id_num=4, angle=0, speed=10, param=10, mode=1)
#dr.set_angles(id_list=[1, 2, 3,4,5,6], angle_list=[0, 0, 0,0,0,0], speed=10, param=10, mode=1)
# dr.get_angle(id_num=1)
    #dr.get_speed(id_num=1)
    #dr.get_state(id_num=1)
    #dr.read_property(id_num=0, property='')
    #dr.set_angles(id_list=[1, 2, 3], angle_list=[0, 0, 0], speed=10, param=10, mode=1)
    #dr.step_angle(id_num=1, angle=0, speed=0, param=0, mode=0)
    #dr.step_angles(id_list=[1, 2, 3], angle_list=[0, 0, 0], speed=10, param=10, mode=1)
    #dr.set_angle_adaptive(id_num=1, angle=30, speed=10, torque=10)
    #dr.set_angles_adaptive(id_list=[1, 2, 3], angle_list=[50, 60, 70], speed_list=[10, 10, 10], torque_list=[10, 10, 10])
    #dr.impedance_control(id_num=0, angle=0, speed=0, tff=0, kp=0, kd=0)
