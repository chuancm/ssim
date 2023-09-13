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
file2=pd.read_excel(r'45du.xlsx',header=None)/255
file3=pd.read_excel(r'90du.xlsx',header=None)/255


#读取图像
imga = cv.imread('A.png',0)
imgb = cv.imread('B.png',0)
imgc = cv.imread('C.png',0)
print(np.size(imga))

#生成图像
# num=Variable(np.random.randint(0,69,(40,40)))
num=torch.tensor(np.random.randint(0, 69, (40, 40)))


img1_origin=np.zeros((40,40,3))
img2_origin=np.zeros((40,40,3))
img3_origin=np.zeros((40,40,3))


for i in range(0,40):
    for j in range (0,40):
        # print(file1[ num[i][j] ] [0])
        
        img1_origin[i,j,0]=file1.iloc[int(num[i,j]),0]
        img1_origin[i,j,1]=file1.iloc[int(num[i,j]),1]
        img1_origin[i,j,2]=file1.iloc[int(num[i,j]),2]

        img2_origin[i,j,0]=file2.iloc[int(num[i,j]),0]
        img2_origin[i,j,1]=file2.iloc[int(num[i,j]),1]
        img2_origin[i,j,2]=file2.iloc[int(num[i,j]),2]

        img3_origin[i,j,0]=file3.iloc[int(num[i,j]),0]
        img3_origin[i,j,1]=file3.iloc[int(num[i,j]),1]
        img3_origin[i,j,2]=file3.iloc[int(num[i,j]),2]



img1=cv.resize(img1_origin,(200,200))
img2=cv.resize(img2_origin,(200,200))
img3=cv.resize(img3_origin,(200,200))

img1_origin = torch.tensor(np.array(img1_origin).reshape(1,3,40,40))
img2_origin = torch.tensor(np.array(img2_origin).reshape(1,3,40,40))
img3_origin = torch.tensor(np.array(img3_origin).reshape(1,3,40,40))

# img1=cv.cvtColor(img1,cv.COLOR_RGB2GRAY)
# img2=cv.cvtColor(img1,cv.COLOR_RGB2GRAY)
# img3=cv.cvtColor(img1,cv.COLOR_RGB2GRAY)
R, G, B = img1[:,:,0], img1[:,:,1], img1[:,:,2]
img1_gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
R, G, B = img2[:,:,0], img2[:,:,1], img2[:,:,2]
img2_gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
R, G, B = img3[:,:,0], img3[:,:,1], img3[:,:,2]
img3_gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

# plt.imsave("img1.png",img1)





print(np.size(img1))
#傅里叶变换
f1 = np.fft.fft2(img1_gray)
fshift1 = np.fft.fftshift(f1)
#设置高通滤波器
rows, cols = imga.shape
crow,ccol = int(rows/2), int(cols/2)
fshift1[crow-30:crow+30, ccol-30:ccol+30] = 0
#傅里叶逆变换
ishift1 = np.fft.ifftshift(fshift1)
iimg1 = np.fft.ifft2(ishift1)
iimg1 = np.abs(iimg1)

#傅里叶变换
f2 = np.fft.fft2(img2_gray)
fshift2 = np.fft.fftshift(f2)
#设置高通滤波器
rows, cols = imga.shape
crow,ccol = int(rows/2), int(cols/2)
fshift2[crow-30:crow+30, ccol-30:ccol+30] = 0
#傅里叶逆变换
ishift2 = np.fft.ifftshift(fshift2)
iimg2 = np.fft.ifft2(ishift2)
iimg2 = np.abs(iimg2)

#傅里叶变换
f3 = np.fft.fft2(img3_gray)
fshift3 = np.fft.fftshift(f3)
#设置高通滤波器
rows, cols = imga.shape
crow,ccol = int(rows/2), int(cols/2)
fshift3[crow-30:crow+30, ccol-30:ccol+30] = 0
#傅里叶逆变换
ishift3 = np.fft.ifftshift(fshift3)
iimg3 = np.fft.ifft2(ishift3)
iimg3 = np.abs(iimg3)






#傅里叶变换
fa = np.fft.fft2(imga)
fshifta = np.fft.fftshift(fa)
#设置高通滤波器
rows, cols = imga.shape
crow,ccol = int(rows/2), int(cols/2)
fshifta[crow-30:crow+30, ccol-30:ccol+30] = 0
#傅里叶逆变换
ishifta = np.fft.ifftshift(fshifta)
iimga = np.fft.ifft2(ishifta)
iimga = np.abs(iimga)

#傅里叶变换
fb = np.fft.fft2(imgb)
fshiftb = np.fft.fftshift(fb)
#设置高通滤波器
rows, cols = imgb.shape
crow,ccol = int(rows/2), int(cols/2)
fshiftb[crow-30:crow+30, ccol-30:ccol+30] = 0
#傅里叶逆变换
ishiftb = np.fft.ifftshift(fshiftb)
iimgb = np.fft.ifft2(ishiftb)
iimgb = np.abs(iimgb)

#傅里叶变换
fc = np.fft.fft2(imgc)
fshiftc = np.fft.fftshift(fc)
#设置高通滤波器
rows, cols = imga.shape
crow,ccol = int(rows/2), int(cols/2)
fshiftc[crow-30:crow+30, ccol-30:ccol+30] = 0
#傅里叶逆变换
ishiftc = np.fft.ifftshift(fshiftc)
iimgc = np.fft.ifft2(ishiftc)
iimgc = np.abs(iimgc)


# plt.imsave("image.png",iimgb,cmap='gray')

# #显示原始图像和高通滤波处理图像
# plt.subplot(121), plt.imshow(imgb,'gray'), plt.title('Original Image')
# plt.axis('off')
# plt.subplot(122), plt.imshow(iimgb, 'gray'), plt.title('Result Image')
# plt.axis('off')
# plt.show()





tensor_iimga = torch.tensor(np.array(iimga).reshape(1,1,200,200))
tensor_iimgb = torch.tensor(np.array(iimgb).reshape(1,1,200,200))
tensor_iimgc = torch.tensor(np.array(iimgc).reshape(1,1,200,200))

tensor_iimg1 = torch.tensor(np.array(iimg1).reshape(1,1,200,200))
tensor_iimg2 = torch.tensor(np.array(iimg2).reshape(1,1,200,200))
tensor_iimg3 = torch.tensor(np.array(iimg3).reshape(1,1,200,200))

iimga= tensor_iimga.cuda()
iimgb= tensor_iimgb.cuda()
iimgc= tensor_iimgc.cuda()
iimg1= tensor_iimg1.cuda()
iimg2= tensor_iimg2.cuda()
iimg3= tensor_iimg3.cuda()

    

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

ssim_value1 = pytorch_msssim.ssim(iimga, iimg1) ###
ssim_value2 = pytorch_msssim.ssim(iimgb, iimg2)
ssim_value3 = pytorch_msssim.ssim(iimgc, iimg3)

print("Initial ssim1:", ssim_value1)
print("Initial ssim2:", ssim_value2)
print("Initial ssim3:", ssim_value3)
# Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)



optimizer = optim.Adam([num], lr=1)

epoch = 1   ###

# ssim_loss = pytorch_msssim.SSIM()
while ssim_value1 < 0.9 and ssim_value2<0.9 and ssim_value3<0.9:
    optimizer.zero_grad()
    ssim_out = 1-ssim_value3
    ssim_out.requires_grad_(True)

    ssim_value = round(- ssim_out.item(), 4)  ### round 保留4位小数
    # save image
    if ssim_value3>0.5 :
        vutils.save_image(img1_origin, rf'C:\Users\13068\OneDrive - 南京大学\桌面\科研\sic\偏振复用\SIMPNG\0du_{epoch}_{ssim_value1}.png')  ### 官网教程没有保存每一次迭代的图像，这里我们保存下来。 如果没有SSIMPNG这个文件夹，手动创建一个。
        vutils.save_image(img2_origin, rf'C:\Users\13068\OneDrive - 南京大学\桌面\科研\sic\偏振复用\SIMPNG\45du_{epoch}_{ssim_value2}.png')
        vutils.save_image(img3_origin, rf'C:\Users\13068\OneDrive - 南京大学\桌面\科研\sic\偏振复用\SIMPNG\90du_{epoch}_{ssim_value3}.png')

        data=num.numpy()
        df = pd.DataFrame(data)
        excel_file = rf'C:\Users\13068\OneDrive - 南京大学\桌面\科研\sic\偏振复用\structure_data\structure_data_{epoch}.xlsx'
        df.to_excel(excel_file, index=False, header=False)
    # print("ssim1:",ssim_value1)
    # print("ssim2:",ssim_value2)
    # print("ssim3:",ssim_value3)    
    ssim_out.backward()
    optimizer.step()
    epoch += 1   ###