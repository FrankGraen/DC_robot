import os
import sys
from rknn.api import RKNN

# 配置参数
MODEL_PATH = './yolov5s.onnx'      # 你的 ONNX 模型路径
RKNN_MODEL = './yolov5s_int8.rknn' # 输出的 RKNN 模型路径
DATASET_PATH = './dataset.txt'     # 用于量化的图片列表文件

def convert():
    # 1. 初始化 RKNN 对象
    rknn = RKNN(verbose=True)

    # 2. 配置模型参数 (针对 RK3588)
    # mean_values 和 std_values 根据 YOLOv5 默认归一化设置
    print('--> Config model')
    rknn.config(
        mean_values=[[0, 0, 0]], 
        std_values=[[255, 255, 255]],
        target_platform='rk3588',
        optimization_level=3 # 开启全量优化
    )

    # 3. 加载 ONNX 模型
    print('--> Loading model')
    ret = rknn.load_onnx(model=MODEL_PATH)
    if ret != 0:
        print('Load model failed!')
        exit(ret)

    # 4. 构建并量化模型
    # do_quantization=True 是提速的关键！
    print('--> Building model')
    ret = rknn.build(
        do_quantization=True, 
        dataset=DATASET_PATH
    )
    if ret != 0:
        print('Build model failed!')
        exit(ret)

    # 5. 导出 RKNN 模型
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)

    print('done')
    rknn.release()

if __name__ == '__main__':
    convert()