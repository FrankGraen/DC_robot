#!/usr/bin/python3
#coding=utf8
import sys
sys.path.append('d:/project/robot_show/robot/daran')
import cv2
import copy
import time
import math
import camera
import threading
import numpy as np
import time
import math
import gripper
import DrEmpower_can as dr
import arm_robot as robot
from cv_ImgAddText import *
from lab_config import color_range

import kinematic as k
from transform import *

import SerialServoRunning as ssr

#设置，打印，分辨率
size = (480, 360)
MyCamera = camera.USBCamera(size)
print('Frame size: ' + str(size[0]) + 'X' + str(size[1]) + '\n')

##机械臂云台轴心到图像下边缘的距离
y0 = 14.1

##检测到物体时计数，到达一定次数后才触发夹取
count = 0

##初始化机械臂
l_p = 0 # 工具参考点到电机输出轴表面的距离，单位mm（所有尺寸参数皆为mm）
l_p_mass_center = 0 # 工具（负载）质心到 6 号关节输出面的距离
G_p = 0 # 负载重量，单位kg，所有重量单位皆为kg
max_list_temp = [85, 215, 149, 142, 179, 179]  # 关节模型角度最大值,1号关节目的是保护线缆，并且到达工作空间边缘；2号关节到达工作空间边缘；3、4号关节目的是不产生杆件干涉；5号关节因有滑环，不考虑；6号关节保证工作空间内全部到达
min_list_temp = [-85, -15, -149, -142, -179, -179]  # 关节模型角度最小值，1号关节目的是不装到装在桌边的竖杆（安装摄像头）；2号关节目的是在伸直的时候不打到桌子；3、4号关节目的是不产生杆件干涉；5号关节因有滑环，不考虑；6号关节保证工作空间内全部到达

# # 机械臂对象初始化函数函数
ro = robot.arm_robot(L_p=l_p, L_p_mass_center=l_p_mass_center, MAX_list_temp=max_list_temp, MIN_list_temp=min_list_temp, G_p=G_p)

#初始化到工作空间
ro.set_arm_pose(pl_temp=[200,58, 140], theta_P_R_Y=[0, 90, 0], speed=2, param=10, mode=1)


#数值映射
#将一个数从一个范围映射到另一个范围
def leMap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

##鼠标左键双击退出
Running = True
def closeEvent(event,x,y,flags,param):
    global Running 
    if event==cv2.EVENT_LBUTTONDBLCLK:
        Running = False
cv2.namedWindow('image')
cv2.setMouseCallback('image', closeEvent)

#找出面积最大的轮廓
#参数为要比较的轮廓的列表
def getAreaMaxContour(contours) :
        contour_area_temp = 0
        contour_area_max = 0
        area_max_contour = None;

        for c in contours : #历遍所有轮廓
            contour_area_temp = math.fabs(cv2.contourArea(c)) #计算轮廓面积
            if contour_area_temp > contour_area_max :
                contour_area_max = contour_area_temp
                if contour_area_temp > 300:  #只有在面积大于300时，最大面积的轮廓才是有效的，以过滤干扰
                    area_max_contour = c

        return area_max_contour, contour_area_max#返回最大的轮廓

COLOR = 'None'
get_color = False
orgframe = None
action_finish = True

center_x, center_y = 0, 0
last_x, last_y = 0, 0

stop = False
start_move = False
rect = None
##机械臂移动策略
def moveTarget():
    global last_x, last_y
    global center_x, center_y
    global y0, COLOR, size
    global get_color, action_finish
    global start_move, stop
    global rect

    start = True
    start_m = False
   
    # 放置方块的x坐标
    x_ = -10.8
    # y坐标的临时变量
    y_ = 0

    # 放置方块的y坐标
    y_red = 6
    y_green = 1
    y_blue = -4.3
    
    # 记录堆放的方块数
    number_red = 0
    number_green = 0
    number_blue = 0

##   参数分别为红绿蓝方块堆放的依次高度，如红色第一层高度0.6cm，第二层高度3.7cm，所谓堆放高度是指机械臂放下方块时距离地面的高度
    z_red_list = [0.8, 3.8]
    z_red_temp_list = copy.deepcopy(z_red_list)
    z_green_list = [0.8, 3.8]
    z_green_temp_list = copy.deepcopy(z_green_list)
    z_blue_list = [0.8, 3.8]
    z_blue_temp_list = copy.deepcopy(z_blue_list)
    while True:
        if get_color:
            get_color = False
            action_finish = False
            if start_move and start:
                start_move = False
                start_m = True
            if start_m:
                # 如果机械臂在左半区夹取时偏左则加一个适当的值进行矫正
                # 同理如果偏右则减去一个值，对于右半区也是一样的道理
                if center_x < 240:
                    center_x += 0
                elif center_x >= 240:
                    center_x += 0
                x = round((leMap(center_x, 0, size[0], 7.5, -7.5)), 2)
                y = leMap( center_y, 0, size[1], 0, 11)
                y = round((y + y0), 2)
                #print(x, y, center_x)
##                判断机械臂是否已经移动到方块上方
##                如果不是则机械臂移动时间1000，防止太快移动
##                如果是则移动时间20
                if start:
                    start = False
##                    移动到目标位置上方6cm，靠后1.8cm处，防止机械臂档物体
                    ki.moveAdapt(x, y - 1.8, 6, 0, -90, 1000)
                else:
##                    如果方块堆放的位置已经堆满方块
                    if number_red > 1 and COLOR == 'red':
                        z_red_temp_list = copy.deepcopy(z_red_list)
                        number_red = 0
                        stop = True
                        time.sleep(5)
                        stop = False
                    if number_green > 1 and COLOR == 'green':
                        z_green_temp_list = copy.deepcopy(z_green_list)
                        number_green = 0
                        stop = True
                        time.sleep(5)
                        stop = False
                    if number_blue > 1 and COLOR == 'blue':
                        z_blue_temp_list = copy.deepcopy(z_blue_list)
                        number_blue = 0
                        stop = True
                        time.sleep(5)
                        stop = False
                        
##                    如果机械臂已经移动到方块上方，时间20
                    ki.moveAdapt(x, y - 1.8 , 6, 0, -90, 20)
                    
##                    如果检测到方块没有移动一段时间后，开始夹取
                    if start_move:
                        start_m = False
                        start = True
                        start_move = False
##                        爪子张开
                        ssr.serial_setServo(1, 200, 500)
                        servo2_angle = getAngle(x, y, rect[2])
                        ssr.serial_setServo(2, servo2_angle, 500)
                        time.sleep(0.5)
##                        移到目标位置，高度2
                        ki.moveAdapt(x, y - 0.8, 2, -90, -0, 1000)
                        
##                        爪子闭合
                        ssr.serial_setServo(1, servo1, 500)
                        time.sleep(1)
                        
##                        机械臂抬起
                        ki.moveAdapt(0, 20, 16, 0, -90, 1000)
                        
##                        对不同颜色方块进行分类放置
                        ssr.serial_setServo(2, 500, 500)
                        if COLOR == 'red':
                            y_ = y_red
                            ki.moveAdapt(x_, y_, 12, -90, 0, 1000)
                            ssr.serial_setServo(2, 355, 500)
                            ki.moveAdapt(x_, y_, z_red_temp_list[0], -90, 0, 1500)
                            del z_red_temp_list[0]
                            time.sleep(0.5)
                            number_red += 1 
                        elif COLOR == 'green':
                            y_ = y_green
                            ki.moveAdapt(x_, y_, 12, -90, 0, 1000)
                            ssr.serial_setServo(2, 500, 500)
                            ki.moveAdapt(x_, y_, z_green_temp_list[0], -90, 0, 1500)
                            del z_green_temp_list[0]
                            time.sleep(0.5)
                            number_green += 1
                        elif COLOR == 'blue':
                            y_ = y_blue
                            ki.moveAdapt(x_ + 0.4, y_, 12, -90, 0, 1500)
                            ssr.serial_setServo(2, 600, 500)
                            ki.moveAdapt(x_ + 0.4, y_, z_blue_temp_list[0], -90, 0, 1500)
                            del z_blue_temp_list[0]
                            time.sleep(0.5)
                            number_blue += 1 

##                        爪子张开  ，放下物体
                        ssr.serial_setServo(1, 400, 400)
                        time.sleep(0.4)
                        ssr.serial_setServo(1, 200, 400)
                        time.sleep(0.4)

                        ki.moveAdapt(-12, y_, 11, -90, 0, 1000)                       
##                        回到初始位置
                        ssr.serial_setServo(1, servo1, 500)
                        ssr.serial_setServo(2, 500, 500)
                        ki.moveAdapt(0, 10, 10, -30, -90, 1500)                
            action_finish = True
        else:
           time.sleep(0.01)
          
#运行子线程
th1 = threading.Thread(target=moveTarget)
th1.setDaemon(True)
th1.start()
    
range_rgb = {'red': (0, 0, 255),
              'blue': (255, 0,0),
              'green': (0, 255, 0),
              'black': (0, 0, 0),
              }

Color_BGR = (0, 0, 0)
color_list = []
count = 0
color_max = None
while True:      
    orgFrame = MyCamera.getframe()
    if orgFrame is not None:
        orgframe = orgFrame.copy()    
        image = orgframe.copy()
        frame = cv2.GaussianBlur(orgframe, (3,3), 0)#高斯模糊
        Frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)#将图片转换到LAB空间
        
        areaMaxContour = 0
        max_area = 0
        area_max = 0

        for i in color_range:
            if i == 'red' or i == 'green' or i == 'blue':
                frame = cv2.inRange(Frame, color_range[i][0], color_range[i][1])#对原图像和掩模进行位运算
                opened = cv2.morphologyEx(frame, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))#开运算
                closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))#闭运算
                contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]#找出轮廓
                areaMaxContour, area_max = getAreaMaxContour(contours)#找出最大轮廓
                if areaMaxContour is not None:
                    if area_max > max_area:#找最大面积
                        max_area = area_max
                        color_max = i
                        areaMaxContour_max = areaMaxContour
        if max_area > 5000:
            rect = cv2.minAreaRect(areaMaxContour_max)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(image, [box], -1, range_rgb[color_max], 2)
            pt1_x, pt1_y = box[0, 0], box[0, 1]
            pt3_x, pt3_y = box[2, 0], box[2, 1]           
            center_x, center_y = (pt1_x + pt3_x)/2, (pt1_y + pt3_y)/2
            cv2.putText(image, '(' + str(center_x )+ ',' + str(center_y) + ')', (min(box[0, 0], box[2, 0]), pt3_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, range_rgb[color_max], 1)
            if action_finish:
                if color_max == 'red':  #红色最大
                    color = 1
                elif color_max == 'green':  #绿色最大
                    color = 2
                elif color_max == 'blue':  #蓝色最大
                    color = 3                   
                else:
                    color = 0
                color_list.append(color)
##                多次判断
                if len(color_list) == 2:
##                    计算方块是否在移动
                    distance = math.sqrt((center_x - last_x)*(center_x - last_x) + (center_y - last_y)*(center_y - last_y))
                    # 累计判断
                    if distance < 4:
                        count += 1
                    else:
                        count = 0
                    if count > 10:
                        count = 0
                        start_move = True
                    last_x, last_y = center_x, center_y
                    # 取平均值
                    color = int(round(np.mean(color_list)))
                    color_list = []
                    if color == 1:
                        COLOR = 'red'
                        Color_BGR = range_rgb["red"]
                    elif color == 2:
                        COLOR = 'green'
                        Color_BGR = range_rgb["green"]
                    elif color == 3:
                        COLOR = 'blue'
                        Color_BGR = range_rgb["blue"]
                    else:
                        color_max = 'None'
                        Color_BGR = range_rgb["black"]
                    get_color = True
        else:
            if action_finish:
                Color_BGR = (0, 0, 0)
                COLOR = "None"
        if stop:
            orgframe = cv2ImgAddText(image, COLOR + '方块堆放位置已被堆满，请移走堆放的方块！\n       机械臂在5秒之后会继续夹取',
                                 30, int(size[1]/2), textColor= (255, 0, 0), textSize = 20)
        cv2.putText(image, 'Double click to quit!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, "Color: " + COLOR, (10, orgframe.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, Color_BGR, 2)
        cv2.imshow("image", image)
        key = cv2.waitKey(1)
        if key == 27 or Running is False:
            break
MyCamera.shutdown()
