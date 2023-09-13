# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

import torch

import pytorch_msssim
import torchvision.utils as vutils
from torch.autograd import Variable
from torch import optim
file1=pd.read_excel(r'0du.xlsx',header=None)/255


#读取图像
img1 = cv.imread('A.png',0)
img2 = cv.imread('test.png',0)








#傅里叶变换
f1 = np.fft.fft2(img1)
fshift1 = np.fft.fftshift(f1)
#设置高通滤波器
rows, cols = img1.shape
crow,ccol = int(rows/2), int(cols/2)
fshift1[crow-30:crow+30, ccol-30:ccol+30] = 0
#傅里叶逆变换
ishift1 = np.fft.ifftshift(fshift1)
iimg1 = np.fft.ifft2(ishift1)
iimg1 = np.abs(iimg1)





#傅里叶变换
f2 = np.fft.fft2(img2)
fshift2 = np.fft.fftshift(f2)
#设置高通滤波器
rows, cols = img2.shape
crow,ccol = int(rows/2), int(cols/2)
fshift2[crow-30:crow+30, ccol-30:ccol+30] = 0
#傅里叶逆变换
ishift2 = np.fft.ifftshift(fshift2)
iimg2 = np.fft.ifft2(ishift2)
iimg2 = np.abs(iimg2)


# plt.imsave("image.png",iimgb,cmap='gray')

# #显示原始图像和高通滤波处理图像
# plt.subplot(121), plt.imshow(imgb,'gray'), plt.title('Original Image')
# plt.axis('off')
# plt.subplot(122), plt.imshow(iimgb, 'gray'), plt.title('Result Image')
# plt.axis('off')
# plt.show()






iimg1= iimg1.cuda()

iimg2= iimg2.cuda()


    

# 输出三个ssim值
# print(pytorch_msssim.ssim(iimga, iimg1))
# print(pytorch_msssim.ssim(iimgb, iimg2))
# print(pytorch_msssim.ssim(iimgc, iimg3))

# plt.imshow(img1)
# plt.show()
# plt.imshow(img2)
# plt.show()
# plt.imshow(img3)
# plt.show()

#ssim_loss = pytorch_msssim.SSIM(window_size = 11)

#print(ssim_loss(img1, img2))

ssim_value1 = pytorch_msssim.ssim(iimg1, iimg2).item()  ###


print("Initial ssim1:", ssim_value1)

# Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)

