import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def generate_3d_grayscale_distribution(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 获取图像尺寸
    height, width = image.shape

    # 计算灰度值分布
    grayscale_distribution = np.zeros((256,))

    for i in range(height):
        for j in range(width):
            grayscale_value = image[i, j]
            grayscale_distribution[grayscale_value] += 1

    # 生成3D灰度分布图
    x = np.arange(256)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x.ravel(), y.ravel(), 0, 1, 1, grayscale_distribution)

    ax.set_xlabel('灰度值')
    ax.set_ylabel('图像高度')
    ax.set_zlabel('像素数量')

    plt.show()

# 替换为你自己的图像路径
image_path = 'path/to/your/image.jpg'
generate_3d_grayscale_distribution(image_path)
