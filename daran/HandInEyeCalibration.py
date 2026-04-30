import numpy as np
import cv2
# 通过九点标定获取的圆心相机坐标


# 手眼标定方法
class HandInEyeCalibration:
    m_bk=[]
    STC_points_camera = np.array([

        [ 417, 214],  # 472.9118347167969, 614.8732299804688
        [ 350.5, 214],  # 464.9814453125, 363.4068298339844
        [ 289, 213],  # 456.2049865722656, 118.54296112060547

        [416,277],
        [350,276],
        [289,275],

        [ 416, 339],  # 472.9118347167969, 614.8732299804688
        [ 350.5, 338],  # 464.9814453125, 363.4068298339844
        [ 289, 338],  # 456.2049865722656, 118.54296112060547
    ])
    # 通过九点标定获取的圆心机械臂坐标
    STC_points_robot = np.array([
        [172, 165],  #z:58,P:0,Y:90,R:-17
        [172, 104],  
        [172, 60],

        [213,164],  #z坐标为： 57.6; Pitch角为: 1.6; Yaw角为: 94.9; Roll角为: -13.7
        [222,113],
        [238.2,59.1],

        [268.7,158.4],
        [270,111],
        [264,60.4]
    ])

    def __init__(self):
        m = self.get_m(points_camera=self.STC_points_camera, points_robot=self.STC_points_robot)
        self.m_bk = m.copy()
    def get_m(self, points_camera, points_robot):
        """
        取得相机坐标转换到机器坐标的仿射矩阵
        :param points_camera:
        :param points_robot:
        :return:
        """
        # 确保两个点集的数量级不要差距过大，否则会输出None
        m, s = cv2.estimateAffine2D(points_camera, points_robot)
        #print("s is",s)
        return m

    def get_transformation_params(self, m):
        """
        从仿射矩阵中提取缩放、旋转和平移参数
        :param m:
        :return:
        """
        if m is None:
            return None, None, None

        # 缩放因子 k
        sx = np.linalg.norm(m[0, :2])
        sy = np.linalg.norm(m[1, :2])
        k = (sx + sy) / 2

        # 旋转角度 theta
        theta_rad = np.arctan2(m[1, 0], m[0, 0])
        theta_deg = np.degrees(theta_rad)
        # 平移量
        tx = m[0, 2]
        ty = m[1, 2]

        return k, theta_deg, (tx, ty)

    def get_points_robot(self, x_camera, y_camera):
        """
        相机坐标通过仿射矩阵变换取得机器坐标
        :param x_camera:
        :param y_camera:
        :return:
        """
        # m = self.get_m(STC_points_camera, STC_points_robot)
        robot_x = (self.m_bk[0][0] * x_camera) + (self.m_bk[0][1] * y_camera) + self.m_bk[0][2]
        robot_y = (self.m_bk[1][0] * x_camera) + (self.m_bk[1][1] * y_camera) + self.m_bk[1][2]
        return robot_x, robot_y,self.m_bk

def main():
        # 创建HandInEyeCalibration实例
        calibration = HandInEyeCalibration()

        # 验证相机坐标
        test_camera_points = [
            [319.5, 248],  # 201 ,86
            [385.5, 305],  # 242.7,140
            [233,195]
    
        ]

        # 转换为机械臂坐标
        for point in test_camera_points:
            x_camera, y_camera = point
            robot_x, robot_y,m = calibration.get_points_robot(x_camera, y_camera)
            print(f"Camera point: ({x_camera}, {y_camera}) -> Robot point: ({robot_x}, {robot_y})")
        print("m = ",m)
        # 计算缩放因子、旋转角度和平移量
        k, theta, (tx, ty) = calibration.get_transformation_params(m)
        print(f"k= {k}，theta= {theta}, txty=({tx},{ty})")

def detect_points():
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头打开失败")
        exit(1)
    while cap.isOpened():
        ret,frame = cap.read()
        # cv2.imshow('Camera', frame)
        camera_points=detect_blocks(frame)
    # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            with open("square_centers.txt", 'w') as f:
                for center in  camera_points:
                    f.write(f"{center[0]} {center[1]}\n")
            break
    cap.release()
    cv2.destroyAllWindows()

#设置，打印，分辨率
size = (480, 360)
#robot_arm = arm.arm_robot()
# 颜色阈值配置（HSV格式）
COLOR_RANGES = {
    "red":    [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [179, 255, 255])],
    "green":  [([35, 50, 50], [85, 255, 255])],
    "blue":   [([100, 50, 50], [130, 255, 255])],
    "yellow": [([20, 100, 100], [30, 255, 255])]
}

def detect_blocks(image):
    # 读取图像并转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    detected_blocks = []
    square_centers = []

    for color_name, ranges in COLOR_RANGES.items():
        # 创建颜色掩膜
        mask = np.zeros_like(hsv[:, :, 0])
        for (lower, upper) in ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        
        # 形态学操作
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 过滤小区域和非方形对象
            if area > 500 and w > 0 and h > 0:
                aspect_ratio = w / h
                if 0.8 <= aspect_ratio <= 1.2:
                    # 记录检测结果
                    detected_blocks.append({
                        "color": color_name,
                        "position": (x, y),
                        "size": (w, h),
                        "center": (x + w//2, y + h//2)
                    })
                    square_centers.append((x+w/2, y + h//2))
    
                    # 在图像上绘制结果x + w//2
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(image, f"{color_name}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Detection Results", image)
    print(square_centers)
    return square_centers
    


if __name__ == "__main__":
    main()
    # detect_points()