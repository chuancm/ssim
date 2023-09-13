# -*- coding: utf-8 -*-
"""
Created on Sat May  8 16:03:10 2021

@author: renyo
"""

#After picture change,save the matrix,orginal data is 4 col matrix
#modify   15 file
#modify   69 save txt name 

import cv2
import numpy as np
import pandas as pd
file=pd.read_excel(r'/Users/chuan/Desktop/sic 光栅/振幅图案/amp_onecolor.xlsx',header=None)
file.iloc[0]
d=[]
for i in range(len(file)):
    d.append([])
    d[i].append(file.iloc[i][1])
    d[i].append(file.iloc[i][2])
    d[i].append(file.iloc[i][3])
label=[]
for i in range(len(file)):
    label.append(file.iloc[i][0])
label=np.array(label)


#import picture
img1 = cv2.imread(r'/Users/chuan/Desktop/sic 光栅/振幅图案/test_onecolor.png')
A=img1.shape[0]
B=img1.shape[1]
img1=img1.reshape(A*B,3)


d=np.array(d)
d=d.reshape(len(file),3)


'''alabel1=np.array([0])
alabel2=np.linspace(140,300,33)
alabel3=np.concatenate((alabel1,alabel2))'''

out_put=np.array(A*B*3*[0])
out_put=out_put.reshape(-1,3)


for i in range(A*B):
    loss0=np.power((img1[i,2]-d[0,0]),2)+np.power((img1[i,1]-d[0,1]),2)+np.power((img1[i,0]-d[0,2]),2)
    #loss0=loss0.sum()
    key=0
    for j in range(len(file)):
        loss1=np.power((img1[i,2]-d[j,0]),2)+np.power((img1[i,1]-d[j,1]),2)+np.power((img1[i,0]-d[j,2]),2)
        #loss1=loss1.sum()
        if loss1>=loss0:
            key=key
        else:
            loss0=loss1
            key=j
            
    out_put[i,0]=label[key]
    out_put[i,1]=label[key]
    out_put[i,2]=label[key]
    
nimg1=out_put.reshape(A,B,3)
out=nimg1[:,:,0]
print(out)

np.savetxt('amp_onecolor.txt',out,fmt='%d')
