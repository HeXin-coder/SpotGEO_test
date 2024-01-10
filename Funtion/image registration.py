import cv2
import numpy as np

def register_images(image1, image2):
    # 将图像转换为灰度图
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 使用ORB特征检测器和描述器
    orb = cv2.ORB_create()

    # 寻找关键点和描述符
    key_points1, descriptors1 = orb.detectAndCompute(gray1, None)
    key_points2, descriptors2 = orb.detectAndCompute(gray2, None)

    # 使用暴力匹配器匹配描述符
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # 根据匹配关系选择好的关键点
    src_pts = np.float32([key_points1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([key_points2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 使用RANSAC算法寻找变换矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 进行图像配准
    registered_image = cv2.warpPerspective(image1, M, (image2.shape[1], image2.shape[0]))

    return registered_image

# 读取两幅图像
image1 = cv2.imread(1.png')
image2 = cv2.imread('2.png')

# 进行图像配准
registered_image = register_images(image1, image2)

# 显示结果
cv2.imshow('Registered Image', registered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
