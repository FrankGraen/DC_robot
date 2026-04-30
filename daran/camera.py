#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import cv2
import sys
import time
import threading
import numpy as np

# usb摄像头获取画面
class USBCamera():
    
    # 初始化
    def __init__(self, resolution=None):
        self.orgFrame = None
        self.Running = True
        # 如果摄像头正常则开启，否则抛出异常提示，退出程序
        cameraVideo = self.checkcamera()
        if cameraVideo is not None:
            self.cap = cv2.VideoCapture(cameraVideo)
            # 如果分辨率有设置
            if resolution is not None:
                # 将分辨率取整，取正
                resolution = (abs(int(resolution[0])), abs(int(resolution[1])))
                # 如果分辨率不在32～1920之间，打印提示信息，退出程序
                if 1920 < resolution[0] or resolution[0] < 32 or 1920 < resolution[1] or resolution[1] < 32:               
                    print('Wrong resolution, resolution should be between 32 and 1920')
                    sys.exit(0)
            # 如果没有设置分辨率，则自动获取摄像头的分辨率
            else:
                resolution = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.resolution = resolution
            print('USBCamera loaded... warming camera\n')
            # 开启摄像头画面更新子线程
            t = threading.Thread(target=self.update)
            t.setDaemon(True)
            t.start()
        # 如果没有检测到摄像头，打印提示信息，退出程序
        else:
            print('''No camera detected or wrong device!
        1 Please make sure the camera is connected to the Raspberry Pi, then rerun this program
        2 if you have already connected the camera, please unplug and plug it in again!''')
            sys.exit(0)        
                        
    # 获取摄像头的设备驱动
    def checkcamera(self):
        print('Checking Device...... \n')
        
        # 直接尝试打开摄像头
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print('Camera detected and opened successfully!\n')
            cap.release()  # 释放摄像头资源
            return 0  # 返回设备号
        else:
            print('No camera detected or wrong device!\n')
            return None

    # 返回一帧图片
    def getframe(self):
        # 如果图片不为空则返回图片
        if self.orgFrame is not None:
            return self.orgFrame
    
    # 画面刷新
    def update(self):
        while True:
            if self.Running is True:
                if self.cap.isOpened():
                    # 读取图片
                    ret, orgframe = self.cap.read()          
                    if ret:
                        # 将摄像头画面缩小以便处理
                        self.orgFrame = cv2.resize(orgframe, (self.resolution[0], self.resolution[1]), interpolation=cv2.INTER_CUBIC)
                    else:
                        time.sleep(0.01)
                else:
                    time.sleep(0.01)
            else:
                break
    
    # 关闭摄像头
    def shutdown(self):
        print('stoping USBCamera\n')
        # 释放摄像头，以便下次使用
        self.cap.release()
        cv2.destroyAllWindows()
        self.Running = False

# 手眼标定函数（简化版）
def hand_eye_calibration_simple(robot_poses, camera_points):
    """
    使用4个点进行简化手眼标定
    :param robot_poses: 机械臂基座坐标系下的位姿矩阵列表
    :param camera_points: 相机坐标系下的点坐标列表
    :return: 相机坐标系到机械臂基座坐标系的变换矩阵 T_base_camera
    """
    # 使用前两个点进行简化标定
    T_base_tool_1 = robot_poses[0]
    P_camera_1 = camera_points[0]
    
    T_base_tool_2 = robot_poses[1]
    P_camera_2 = camera_points[1]
    
    # 计算相机坐标系到机械臂基座坐标系的变换矩阵
    T_base_camera = np.dot(T_base_tool_1, np.linalg.inv(P_camera_1))
    T_base_camera = np.dot(T_base_camera, P_camera_2)
    T_base_camera = np.dot(T_base_camera, np.linalg.inv(T_base_tool_2))
    
    return T_base_camera


# 检测圆点并输出坐标
def detect_circle(frame):
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 使用Hough圆变换检测圆
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,  # 调整 minDist 以增加圆心间距
                               param1=50, param2=80,  # 增加 param2 的值以减少检测到的圆
                               minRadius=10, maxRadius=50)  # 限制圆的半径范围
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # 绘制圆心
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            # 绘制圆轮廓
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # 输出圆心坐标
            print(f"Circle Center: ({i[0]}, {i[1]})")
    else:
        print("No circle detected!")

# 使用范例
if __name__ == '__main__':

    MyCamera = USBCamera((480, 360))
    
    # 假设采集了4组数据
    robot_poses = [
        np.array([280.7,38.7,130], [3.6, 88,18]),  # 第1组数据
        np.array([364.4,28.8,125.5], [3.6, 90,5.5]),  # 第2组数据
        np.array([321.7,-57.4,127.8], [-2.4, 91.3,8.1]),  # 第3组数据
        np.array([347.7, -49.8, 131.7], [2.9, 90.4,-3.8]),   # 第4组数据
        np.array([347.8, -2.05, 126], [-2.4, 91.3,8.1]) 
    ]

    camera_points = [
        np.array([320, 66]),  # 第1组数据
        np.array([324, 214]),  # 第2组数据
        np.array([232, 174]),  # 第3组数据
        np.array([224, 236]),  # 第4组数据
        np.array([280, 202])
    ]

    # # 进行简化手眼标定
    # T_base_camera = hand_eye_calibration_simple(robot_poses, camera_points)
    # print("简化手眼标定结果：\n", T_base_camera)

    if not MyCamera.cap.isOpened():
        print("摄像头打开失败")
        exit(1)
    while MyCamera.cap.isOpened():
        frame = MyCamera.getframe()
        if frame is not None:
            # 检测圆点
            detect_circle(frame)
            # 显示提示信息
            cv2.putText(frame, 'Double click to quit!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break           
    MyCamera.shutdown()