import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def generate_grayscale_distribution(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 获取图像的宽和高
    height, width = image.shape

    # 生成坐标网格
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # 创建3D图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制灰度分布曲面
    ax.plot_surface(x, y, image, cmap='viridis')

    # 设置轴标签
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_zlabel('Grayscale Value')

    # 显示图像
    plt.show()

# 替换为你自己的图像路径
image_path = 'path/to/your/image.jpg'
generate_3d_grayscale_distribution(image_path)
