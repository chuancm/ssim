import cv2
import numpy as np
import pandas as pd

file1=pd.read_excel(r'0du.xlsx',header=None)
file2=pd.read_excel(r'45du.xlsx',header=None)
file3=pd.read_excel(r'90du.xlsx',header=None)

#file.iloc[0]
d1=[]
d2=[]
d3=[]

for i in range(len(file1)):
    d1.append([])
    d1[i].append(file1.iloc[i][0])
    d1[i].append(file1.iloc[i][1])
    d1[i].append(file1.iloc[i][2])

    d2.append([])
    d2[i].append(file2.iloc[i][0])
    d2[i].append(file2.iloc[i][1])
    d2[i].append(file2.iloc[i][2])

    d3.append([])
    d3[i].append(file3.iloc[i][0])
    d3[i].append(file3.iloc[i][1])
    d3[i].append(file3.iloc[i][2])

print(d1)
print(d2)
print(d3)
img1 = cv2.imread(r'a.png')
img2 = cv2.imread(r'b.png')
img3 = cv2.imread(r'c.png')

A=img1.shape[0]
B=img1.shape[1]
img1=img1.reshape(A,B,3)
img2=img2.reshape(A,B,3)
img3=img3.reshape(A,B,3)

m1=[]
m2=[]
m3=[]
m4=[]
m5=[]
m6=[]
m7=[]
m8=[]

for i in range(0,A-1):
    for j in range (0,B-1):
        if img1[i][j].all()== 0 and img2[i][j].all() == 0 and img3[i][j].all() == 0: #三者相交
            m1.append([i,j])
        if img1[i][j].all()> 0 and img2[i][j].all() == 0 and img3[i][j].all() == 0:#23没1
            m2.append([i,j])
        if img1[i][j].all()== 0 and img2[i][j].all() > 0 and img3[i][j].all() == 0:#13没2
            m3.append([i,j])
        if img1[i][j].all()== 0 and img2[i][j].all() == 0 and img3[i][j].all() > 0:#12没3
            m4.append([i,j])
        if img1[i][j].all()> 0 and img2[i][j].all() > 0 and img3[i][j].all() == 0:#只有3
            m5.append([i,j])
        if img1[i][j].all()> 0 and img2[i][j].all() == 0 and img3[i][j].all() > 0:#只有2
            m6.append([i,j])
        if img1[i][j].all()== 0 and img2[i][j].all() > 0 and img3[i][j].all() > 0:#只有1
            m7.append([i,j])
        if img1[i][j].all()> 0 and img2[i][j].all() > 0 and img3[i][j].all() > 0:
            m8.append([i,j])



m2=np.array(m2)
m2=np.array(m2)
m3=np.array(m3)
m4=np.array(m4)
m5=np.array(m5)
m6=np.array(m6)
m7=np.array(m7)
m8=np.array(m8)

print(len(m1),len(m2),len(m3),len(m4),len(m5),len(m6),len(m7),len(m8))
d1=np.array(d1)
d2=np.array(d2)
d3=np.array(d3)


l=len(file1)
l=9
print(l)
for w1 in range(1,l):
    for w2 in range(1,l):
        for w3 in range(1,l):
            for w4 in range(1,l):
                for w5 in range(1,l):
                    for w6 in range(1,l):
                        for w7 in range(1,l):
                            for w8 in range(1,l):
                                a=[w1,w2,w3,w4,w5,w6,w7,w8]
                                b=list(set(a))
                                if len(a)==len(b):
                                    nimg1=img1
                                    nimg2=img1
                                    nimg3=img1
                                    
                        
                                    for i in range(0,len(m1)-1):
                                        nimg1[m1[i][0],m1[i][1],0]=d1[w1,0]
                                        nimg1[m1[i][0],m1[i][1],1]=d1[w1,1]
                                        nimg1[m1[i][0],m1[i][1],2]=d1[w1,2]

                                    for i in range(0,len(m2)-1):
                                        nimg1[m2[i][0],m2[i][1],0]=d1[w2,0]
                                        nimg1[m2[i][0],m2[i][1],1]=d1[w2,1]
                                        nimg1[m2[i][0],m2[i][1],2]=d1[w2,2]

                                    for i in range(0,len(m3)-1):
                                        nimg1[m3[i][0],m3[i][1],0]=d1[w3,0]
                                        nimg1[m3[i][0],m3[i][1],1]=d1[w3,1]
                                        nimg1[m3[i][0],m3[i][1],2]=d1[w3,2]

                                    for i in range(0,len(m4)-1):
                                        nimg1[m4[i][0],m4[i][1],0]=d1[w4,0]
                                        nimg1[m4[i][0],m4[i][1],1]=d1[w4,1]
                                        nimg1[m4[i][0],m4[i][1],2]=d1[w4,2]

                                    for i in range(0,len(m5)-1):
                                        nimg1[m5[i][0],m5[i][1],0]=d1[w5,0]
                                        nimg1[m5[i][0],m5[i][1],1]=d1[w5,1]
                                        nimg1[m5[i][0],m5[i][1],2]=d1[w5,2]

                                    for i in range(0,len(m6)-1):
                                        nimg1[m6[i][0],m6[i][1],0]=d1[w6,0]
                                        nimg1[m6[i][0],m6[i][1],1]=d1[w6,1]
                                        nimg1[m6[i][0],m6[i][1],2]=d1[w6,2]

                                    for i in range(0,len(m7)-1):
                                        nimg1[m7[i][0],m7[i][1],0]=d1[w7,0]
                                        nimg1[m7[i][0],m7[i][1],1]=d1[w7,1]
                                        nimg1[m7[i][0],m7[i][1],2]=d1[w7,2]

                                    for i in range(0,len(m8)-1):
                                        nimg1[m8[i][0],m8[i][1],0]=d1[w8,0]
                                        nimg1[m8[i][0],m8[i][1],1]=d1[w8,1]
                                        nimg1[m8[i][0],m8[i][1],2]=d1[w8,2]
                                    
                                    cv2.imwrite(str(w1)+"_"+str(w2)+"_"+str(w3)+"_"+str(w4)+"_"+str(w5)+"_"+str(w6)+"_"+str(w7)+"_"+str(w8)+"_"+"0du.jpg",nimg1)


                                    for i in range(0,len(m1)-1):
                                        nimg2[m1[i][0],m1[i][1],0]=d2[w1,0]
                                        nimg2[m1[i][0],m1[i][1],1]=d2[w1,1]
                                        nimg2[m1[i][0],m1[i][1],2]=d2[w1,2]

                                    for i in range(0,len(m2)-1):
                                        nimg2[m2[i][0],m2[i][1],0]=d2[w2,0]
                                        nimg2[m2[i][0],m2[i][1],1]=d2[w2,1]
                                        nimg2[m2[i][0],m2[i][1],2]=d2[w2,2]

                                    for i in range(0,len(m3)-1):
                                        nimg2[m3[i][0],m3[i][1],0]=d2[w3,0]
                                        nimg2[m3[i][0],m3[i][1],1]=d2[w3,1]
                                        nimg2[m3[i][0],m3[i][1],2]=d2[w3,2]

                                    for i in range(0,len(m4)-1):
                                        nimg2[m4[i][0],m4[i][1],0]=d2[w4,0]
                                        nimg2[m4[i][0],m4[i][1],1]=d2[w4,1]
                                        nimg2[m4[i][0],m4[i][1],2]=d2[w4,2]

                                    for i in range(0,len(m5)-1):
                                        nimg2[m5[i][0],m5[i][1],0]=d2[w5,0]
                                        nimg2[m5[i][0],m5[i][1],1]=d2[w5,1]
                                        nimg2[m5[i][0],m5[i][1],2]=d2[w5,2]

                                    for i in range(0,len(m6)-1):
                                        nimg2[m6[i][0],m6[i][1],0]=d2[w6,0]
                                        nimg2[m6[i][0],m6[i][1],1]=d2[w6,1]
                                        nimg2[m6[i][0],m6[i][1],2]=d2[w6,2]

                                    for i in range(0,len(m7)-1):
                                        nimg2[m7[i][0],m7[i][1],0]=d2[w7,0]
                                        nimg2[m7[i][0],m7[i][1],1]=d2[w7,1]
                                        nimg2[m7[i][0],m7[i][1],2]=d2[w7,2]

                                    for i in range(0,len(m8)-1):
                                        nimg2[m8[i][0],m8[i][1],0]=d2[w8,0]
                                        nimg2[m8[i][0],m8[i][1],1]=d2[w8,1]
                                        nimg2[m8[i][0],m8[i][1],2]=d2[w8,2]
                                    
                                    cv2.imwrite(str(w1)+"_"+str(w2)+"_"+str(w3)+"_"+str(w4)+"_"+str(w5)+"_"+str(w6)+"_"+str(w7)+"_"+str(w8)+"_"+"45du.jpg",nimg2)



                                    for i in range(0,len(m1)-1):
                                        nimg3[m1[i][0],m1[i][1],0]=d3[w1,0]
                                        nimg3[m1[i][0],m1[i][1],1]=d3[w1,1]
                                        nimg3[m1[i][0],m1[i][1],2]=d3[w1,2]

                                    for i in range(0,len(m2)-1):
                                        nimg3[m2[i][0],m2[i][1],0]=d3[w2,0]
                                        nimg3[m2[i][0],m2[i][1],1]=d3[w2,1]
                                        nimg3[m2[i][0],m2[i][1],2]=d3[w2,2]

                                    for i in range(0,len(m3)-1):
                                        nimg3[m3[i][0],m3[i][1],0]=d3[w3,0]
                                        nimg3[m3[i][0],m3[i][1],1]=d3[w3,1]
                                        nimg3[m3[i][0],m3[i][1],2]=d3[w3,2]

                                    for i in range(0,len(m4)-1):
                                        nimg3[m4[i][0],m4[i][1],0]=d3[w4,0]
                                        nimg3[m4[i][0],m4[i][1],1]=d3[w4,1]
                                        nimg3[m4[i][0],m4[i][1],2]=d3[w4,2]

                                    for i in range(0,len(m5)-1):
                                        nimg3[m5[i][0],m5[i][1],0]=d3[w5,0]
                                        nimg3[m5[i][0],m5[i][1],1]=d3[w5,1]
                                        nimg3[m5[i][0],m5[i][1],2]=d3[w5,2]

                                    for i in range(0,len(m6)-1):
                                        nimg3[m6[i][0],m6[i][1],0]=d3[w6,0]
                                        nimg3[m6[i][0],m6[i][1],1]=d3[w6,1]
                                        nimg3[m6[i][0],m6[i][1],2]=d3[w6,2]

                                    for i in range(0,len(m7)-1):
                                        nimg3[m7[i][0],m7[i][1],0]=d3[w7,0]
                                        nimg3[m7[i][0],m7[i][1],1]=d3[w7,1]
                                        nimg3[m7[i][0],m7[i][1],2]=d3[w7,2]

                                    for i in range(0,len(m8)-1):
                                        nimg3[m8[i][0],m8[i][1],0]=d3[w8,0]
                                        nimg3[m8[i][0],m8[i][1],1]=d3[w8,1]
                                        nimg3[m8[i][0],m8[i][1],2]=d3[w8,2]
                                    
                                    # print(nimg1[m1[i][0],m1[i][1],0],nimg1[m1[i][0],m1[i][1],1],nimg1[m1[i][0],m1[i][1],2])
                                    # print(nimg2[m1[i][0],m1[i][1],0],nimg2[m1[i][0],m1[i][1],1],nimg2[m1[i][0],m1[i][1],2])
                                    # print(nimg3[m1[i][0],m1[i][1],0],nimg3[m1[i][0],m1[i][1],1],nimg3[m1[i][0],m1[i][1],2])
                                    # print(nimg1)
                                    # print(nimg2)
                                    # print(nimg3)
                                
                                    cv2.imwrite(str(w1)+"_"+str(w2)+"_"+str(w3)+"_"+str(w4)+"_"+str(w5)+"_"+str(w6)+"_"+str(w7)+"_"+str(w8)+"_"+"90du.jpg",nimg3)

