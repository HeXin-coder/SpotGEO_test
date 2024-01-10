% 读取图像
image = imread('path/to/your/image.jpg');

% 获取图像的宽和高
[height, width] = size(image);

% 生成坐标网格
[x, y] = meshgrid(1:width, 1:height);

% 创建3D图
figure;
surf(x, y, double(image), 'EdgeColor', 'none');

% 设置轴标签
xlabel('Width');
ylabel('Height');
zlabel('Grayscale Value');

% 显示图像
title('Grayscale Distribution');
