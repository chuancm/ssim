#ssim
1. 读取三个偏振态下的图片A.png B.png C.png（灰度图像）
2. 生成随机张量对应着随机结构，从70个结构中随机填充40*40的矩阵
3. 对A.png进行fft变化，并进行一次高通滤波
4. 对随机张量所生成的结构的图片转换成灰度图像，并进行相同的操作
5. 利用ssim描述图像差别程度
6. 利用梯度优化算法进行优化，循环
