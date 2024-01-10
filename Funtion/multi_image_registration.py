import cv2
import numpy as np

# def homomorphic_filter(image):
#     # 同态滤波
#     img_float = np.float32(image)
#     img_log = np.log1p(img_float)
#     img_filtered = cv2.dft(np.float32(img_log), flags=cv2.DFT_COMPLEX_OUTPUT)
#     img_filtered = np.fft.fftshift(img_filtered)
#     img_filtered = cv2.magnitude(img_filtered[:, :, 0], img_filtered[:, :, 1])
#     img_filtered = np.log1p(img_filtered)
#     img_filtered = np.exp(img_filtered)
#     img_filtered = np.uint8(cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX))
#     return img_filtered
#
# def preprocess_image(image):
#     # 同态滤波和均值滤波
#     homomorphic = homomorphic_filter(image)
#     blurred = cv2.blur(homomorphic, (5, 5))
#     return blurred

def extend_edges(image):
    # 拓展图像边缘
    h, w = image.shape
    extended = np.zeros((h + 2, w + 2), dtype=image.dtype)
    extended[1:-1, 1:-1] = image
    return extended

def extract_center(image):
    # 截取图像中心区域
    h, w = image.shape
    center_h = h // 3
    center_w = w // 3
    center = image[center_h:2 * center_h, center_w:2 * center_w]
    return center

def main():
    # 循环处理多帧图像
    num_frames = 5
    images = [cv2.imread(f'{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(1, num_frames + 1)]

    processed_images = []
    for i in range(num_frames):
        # processed_image = preprocess_image(images[i])
        processed_image = images[i]
        processed_images.append(processed_image)

    # 假设所有图像尺寸一致
    h, w = processed_images[0].shape

    # 拓展图像边缘
    extended_images = [extend_edges(processed_images[i]) for i in range(num_frames)]

    # 截取中心区域
    center_images = [extract_center(extended_images[i]) for i in range(num_frames)]

    # 计算变换矩阵
    matrix_list = []
    for i in range(num_frames - 1, 0, -1):
        result = cv2.matchTemplate(center_images[i], center_images[i - 1], cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        matrix = np.float32([[1, 0, max_loc[0] - center_images[i - 1].shape[1] // 2],
                             [0, 1, max_loc[1] - center_images[i - 1].shape[0] // 2]])
        matrix_list.append(matrix)

    # 对所有图像进行配准
    aligned_images = [processed_images[-1]]
    for i in range(num_frames - 1, 0, -1):
        aligned_image = cv2.warpAffine(aligned_images[-1], matrix_list[i - 1], (w, h))
        aligned_images.append(aligned_image)

    # 合并图像
    result_image = np.zeros_like(aligned_images[0])
    for i in range(num_frames):
        result_image = cv2.addWeighted(result_image, 1, aligned_images[i], 1 / num_frames, 0)

    cv2.imshow('Aligned Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
