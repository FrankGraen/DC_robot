
import cv2
import numpy as np
import arm_robot as arm

l_p = 0  # 工具参考点到电机输出轴表面的距离，单位mm（所有尺寸参数皆为mm）
l_p_mass_center = 0  # 工具（负载）质心到 6 号关节输出面的距离
G_p = 0  # 负载重量，单位kg，所有重量单位皆为kg
uart_baudrate = 115200  # 串口波特率，与CAN模块的串口波特率一致，（出厂默认为 115200，最高460800）
com = 'COM7'  # 在这里输入 COM 端口号
#robot_arm = arm.arm_robot(L_p=l_p, L_p_mass_center=l_p_mass_center, G_p=G_p, com=com, uart_baudrate=uart_baudrate)

# 设置，打印，分辨率
size = (480, 360)
# 颜色阈值配置（HSV格式）
COLOR_RANGES = {
    "red":    [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [179, 255, 255])],
    "green":  [([35, 50, 50], [85, 255, 255])],
    "blue":   [([100, 50, 50], [130, 255, 255])],
    "yellow": [([20, 100, 100], [30, 255, 255])]
}
# 分拣区位置定义（假设位置为(x, y, z)坐标）
SORTING_POSITIONS = {
    "red": (50, 200, 50),
    "blue": (100, 200, 50),
    "green": (200, 200, 50),
    "yellow": (300, 200, 50)
}

# 堆叠位置定义（假设每次堆叠增加的高度为20）
STACK_HEIGHT_INCREMENT = 30

# 存储检测结果
previous_block = None

def detect_blocks(image):
    # 读取图像并转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    detected_blocks = []

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
                    
                    # 在图像上绘制结果
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(image, f"{color_name}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Detection Results", image)
    return detected_blocks

# def pick_and_place(block_position, target_position, stack_height=0):
#     """夹取并放置物体"""
#     # 移动到物体上方
#     robot_arm.set_arm_position(pl_temp=[block_position[0], block_position[1], block_position[2] + 50], speed=1.0, param=10, mode=1)
#     # 移动到物体位置
#     robot_arm.set_arm_position(pl_temp=[block_position[0], block_position[1], block_position[2]], speed=1.0, param=10, mode=1)
#     # 夹取物体
#     robot_arm.grab()
#     # 移动到物体上方
#     robot_arm.set_arm_position(pl_temp=[block_position[0], block_position[1], block_position[2] + 50], speed=1.0, param=10, mode=1)
#     # 移动到目标位置上方
#     robot_arm.set_arm_position(pl_temp=[target_position[0], target_position[1], target_position[2] + 50 + stack_height], speed=1.0, param=10, mode=1)
#     # 移动到目标位置
#     robot_arm.set_arm_position(pl_temp=[target_position[0], target_position[1], target_position[2] + stack_height], speed=1.0, param=10, mode=1)
#     # 释放物体
#     robot_arm.release()
#     # 移动到目标位置上方
#     robot_arm.set_arm_position(pl_temp=[target_position[0], target_position[1], target_position[2] + 50 + stack_height], speed=1.0, param=10, mode=1)

if __name__ == "__main__":
    
    # stack_heights = {color: 0 for color in SORTING_POSITIONS.keys()}
       
    cap=cv2.VideoCapture(1)
    if not cap.isOpened():
        print("摄像头打开失败")
        exit(1)
    while cap.isOpened():
        ret,frame = cap.read()
        cv2.imshow('Camera', frame)
        # results = detect_blocks(frame)
        
        # for block in detected_blocks:
        #     color = block['color']
        #     center = block['center']
            
        #     if color in SORTING_POSITIONS:
        #         target_position = SORTING_POSITIONS[color]
        #         stack_height = stack_heights[color]
                
        #         print(f"Detected {color} block at position: {center}, moving to sorting position: {target_position} with stack height: {stack_height}")
                
                # # 夹取并放置物体
                # pick_and_place(robot_arm, center, target_position, stack_height)
                
                # # 更新堆叠高度
                # stack_heights[color] += STACK_HEIGHT_INCREMENT
                
                # # 更新前一个检测到的方块
                # previous_block = block
                
                # # 如果检测到前后两个色块颜色一致，进行堆叠
                # if previous_block and previous_block['color'] == color:
                #     print(f"Detected two consecutive {color} blocks, stacking them.")
                #     # 计算堆叠位置
                #     stack_position = (target_position[0], target_position[1], target_position[2] + stack_heights[color] - STACK_HEIGHT_INCREMENT)
                #     # 夹取并放置物体到堆叠位置
                #     pick_and_place(robot_arm, center, stack_position, stack_heights[color] - STACK_HEIGHT_INCREMENT)
                    
                #     # 更新堆叠高度
                #     stack_heights[color] += STACK_HEIGHT_INCREMENT