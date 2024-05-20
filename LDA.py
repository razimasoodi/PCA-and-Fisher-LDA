#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
#from sklearn.model_selection import train_test_split
import os
import cv2 as cv


# In[2]:


DATADIR="C:/Users/razeeee/jaffe"
path=os.path.join(DATADIR)
img_list=[]
class1=[]
class2=[]
class3=[]
class4=[]
class5=[]
class6=[]
class7=[]
data=[]
img_size=64
for img in os.listdir(path):
    img_array=cv.imread(os.path.join(path,img),cv.IMREAD_GRAYSCALE)
    if img_array is not None:
        new_image=cv.resize(img_array,(img_size,img_size))
        img_list.append(new_image)
    if img[3:5]=='AN':
        img_array=cv.imread(os.path.join(path,img),cv.IMREAD_GRAYSCALE)
        if img_array is not None:
            new_image=cv.resize(img_array,(img_size,img_size))
            class1.append(new_image)
    elif img[3:5]=='DI':
        img_array=cv.imread(os.path.join(path,img),cv.IMREAD_GRAYSCALE)
        if img_array is not None:
            new_image=cv.resize(img_array,(img_size,img_size))
            class2.append(new_image)
    elif img[3:5]=='FE':
        img_array=cv.imread(os.path.join(path,img),cv.IMREAD_GRAYSCALE)
        if img_array is not None:
            new_image=cv.resize(img_array,(img_size,img_size))
            class3.append(new_image)
    elif img[3:5]=='HA':
        img_array=cv.imread(os.path.join(path,img),cv.IMREAD_GRAYSCALE)
        if img_array is not None:
            new_image=cv.resize(img_array,(img_size,img_size))
            class4.append(new_image)
    elif img[3:5]=='NE':
        img_array=cv.imread(os.path.join(path,img),cv.IMREAD_GRAYSCALE)
        if img_array is not None:
            new_image=cv.resize(img_array,(img_size,img_size))
            class5.append(new_image)
    elif img[3:5]=='SA':
        img_array=cv.imread(os.path.join(path,img),cv.IMREAD_GRAYSCALE)
        if img_array is not None:
            new_image=cv.resize(img_array,(img_size,img_size))
            class6.append(new_image) 
    elif img[3:5]=='SU':
        img_array=cv.imread(os.path.join(path,img),cv.IMREAD_GRAYSCALE)
        if img_array is not None:
            new_image=cv.resize(img_array,(img_size,img_size))
            class7.append(new_image) 
images=np.array(img_list).reshape(170,(img_size*img_size)) 
#images=images/256
classes=[class1,class2,class3,class4,class5,class6,class7]
for i in classes:
    t=np.array(i).reshape((len(i)),(img_size*img_size))
    data.append(t)


# In[3]:


for i in [1,66,150]:
    plt.imshow(images[i].reshape(img_size,img_size), cmap="gray")
    plt.show()


# In[4]:


def findmean(data,images):
    mean_array=np.zeros((7,64*64))
    for i in range(7):
        mean_array[i]=np.mean(data[i],axis=0)
    meanT=np.mean(images,axis=0)    
    return (mean_array,meanT) 


# In[5]:


mius,miuT=findmean(data,images)


# In[6]:


#def findSbSw(data,mius,miuT):
Sb=np.zeros((64*64,64*64))
s=np.zeros((7,64*64))
#for i in data:
 #   Sb=len(i)*((mius-miuT).T)@(mius-miuT)
#for i in range(7):
 #   s[i]=(mius[i]-miuT)
  #  Sb+=(len(data[i]))*(s.T@s)
for i in range(7):
    s=(mius[i]-miuT).reshape((-1,1))
    #print(s.shape)
    Sb+=(len(data[i]))*(s@s.T)     
Sw=np.zeros((64*64,64*64))
k=np.zeros((7,64*64))
#for i in range(7):
 #   for j in range(len(data[i])):
  #      k[i]=(data[i][j]-mius[i])
   #     Sw+=(k.T@k)
for i in range(7):
    for j in data[i]:
        Sw+=((j-mius[i]).T@(j-mius[i]))       
#for i in range(7):
 #   Sw+=((data[i]-mius[i]).T@(data[i]-mius[i]))   


# In[7]:


swinvsb=np.linalg.pinv(Sw)@Sb
Evalue,Evector=np.linalg.eigh(swinvsb)
Evalue=Evalue.reshape((-1,1))


# In[8]:


most_var=[]
sh=[]
for j in range(4096):    
    sh.append(j)
s=sorted(zip(Evalue,sh)) 
s.reverse()
for i in s:
    most_var.append(i[1])
mostvar=np.array(most_var)


# In[9]:


y1= images @ Evector[ : ,mostvar[0]]
y1=y1.reshape((-1,1))
y6= images @Evector[ : ,mostvar[0:6]]
y29= images @ Evector[ : ,mostvar[0:29]]


# In[10]:


t1= (y1@Evector[ : ,mostvar[0]].reshape((-1,1)).T)
t6= (y6@Evector[ : ,mostvar[0:6]].T)
t29= (y29@Evector[ : ,mostvar[0:29]].T)


# In[11]:


cons1=t1.reshape(170,64,64)
cons6=t6.reshape(170,64,64)
cons29=t29.reshape(170,64,64)


# In[12]:


for i in [1,66,150]:
    plt.imshow(cons29[i], cmap="gray")
    plt.show() 


# In[13]:


for i in [1,66,150]:
    plt.imshow(cons6[i], cmap="gray")
    plt.show()


# In[14]:


for i in [1,66,150]:
    plt.imshow(cons1[i], cmap="gray")
    plt.show()


# In[15]:


def MSE(img,recons):
    mse=(np.sum((img-recons)**2))
    return mse


# In[16]:


mse=[]
s=0
for i in range(1,300):
    #d=(Evector[ : ,mostvar[ :i]]).reshape((-1,1))
    y=images @Evector[ : ,mostvar[ :i]]
    #y= images @ d
    t= y @ Evector[ : ,mostvar[ :i]].T
    f=MSE(images,t)
    mse.append(f)
    if np.abs(f-s)<0.1:
        print('best k=',i,'and mse=',f)
        break
    s=f    


# In[17]:


plt.plot(mse)

