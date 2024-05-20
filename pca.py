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


DATADIR="jaffe"
path=os.path.join(DATADIR)
images=[]
img_size=64
for img in os.listdir(path):
    img_array=cv.imread(os.path.join(path,img),cv.IMREAD_GRAYSCALE)
    if img_array is not None:
        new_image=cv.resize(img_array,(img_size,img_size))
        images.append(new_image)
b=np.array(images)    
new_img = b.reshape(170,(img_size*img_size))


# In[3]:


plt.imshow(b[100], cmap="gray")
plt.show()
plt.imshow(b[66], cmap="gray")
plt.show()
plt.imshow(b[150], cmap="gray")
plt.show()


# In[4]:


X=np.zeros((170,4096))
for i in range(4096):
    X[ : ,i]+=(new_img[ : ,i]-new_img[ : ,i].mean())/new_img[ : ,i].std()


# In[5]:


sigma=np.cov(X.T)
lamda,v=np.linalg.eigh(sigma)
lamda=lamda.reshape((-1,1))
sigma.shape


# In[6]:


most_var=[]
sh=[]
for j in range(4096):    
    sh.append(j)
s=sorted(zip(lamda,sh)) 
s.reverse()
for i in s:
    most_var.append(i[1])
mostvar=np.array(most_var)


# In[7]:


y1= X @ v[ : ,mostvar[0]]
y1=y1.reshape((-1,1))
y40= X @ v[ : ,mostvar[0:40]]
y120= X @ v[ : ,mostvar[0:120]]


# In[8]:


t1= (y1@v[ : ,mostvar[0]].reshape((-1,1)).T)
t40= (y40@v[ : ,mostvar[0:40]].T)
t120= (y120@v[ : ,mostvar[0:120]].T)


# In[9]:


cons1=t1.reshape(170,64,64)
cons40=t40.reshape(170,64,64)
cons120=t120.reshape(170,64,64)


# In[10]:


plt.imshow(cons120[100], cmap="gray")
plt.show()
plt.imshow(cons120[66], cmap="gray")
plt.show()
plt.imshow(cons120[150], cmap="gray")
plt.show()


# In[11]:


plt.imshow(cons40[100], cmap="gray")
plt.show()
plt.imshow(cons40[66], cmap="gray")
plt.show()
plt.imshow(cons40[150], cmap="gray")
plt.show()


# In[12]:


plt.imshow(cons1[100], cmap="gray")
plt.show()
plt.imshow(cons1[66], cmap="gray")
plt.show()
plt.imshow(cons1[150], cmap="gray")
plt.show()


# In[13]:


def MSE(img,recons):
    mse=-(np.sum((img-recons)**2))
    return mse


# In[14]:


mse=[]
s=0
for i in range(64*64):
    d=(v[ : ,most_var[i]]).reshape((-1,1))
    y= X @ d
    t= y @ d.T
    f=MSE(X,t)
    mse.append(f)
    if np.abs(f-s)<0.1:
        print('best k=',i)
        break
    s=f    


# In[15]:


plt.plot(mse)


# In[16]:


w= X @ v[ : ,mostvar[0:2]]
#plt.style.use('seaborn')
plt.scatter(w[ : ,0],w[ : ,1])


# In[17]:


q= X @ v[ : ,mostvar[0:3]]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(q[ : ,0],q[ : ,1],q[ : ,2])

