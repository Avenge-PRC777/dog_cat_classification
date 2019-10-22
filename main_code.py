# %% [code]
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import os
import random
import gc #THIS IS GARBAGE COLLECTOR

# %% [code]
#All data file paths on kaggle start with the root dir
#root dir is ../input

#Chnage the train and test directory path according to your dataset path.
train_dir='../input/dogs-vs-cats-redux-kernels-edition/train'
test_dir='../input/dogs-vs-cats-redux-kernels-edition/test'
train_dogs=['../input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]
train_cats=['../input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]
test_imgs=['../input/dogs-vs-cats-redux-kernels-edition/test/{}'.format(i) for i in os.listdir(test_dir)]

train_imgs = train_dogs[:2000] + train_cats[:2000]
random.shuffle(train_imgs)
print(train_imgs[3:15])

del train_dogs
del train_cats
gc.collect()

# %% [code]
nrows=150
ncols=150
nchannels=3
def processimage(listofimages):
    X=[]
    y=[]
    for i in listofimages:
        X.append(cv2.resize(cv2.imread(i,cv2.IMREAD_COLOR),(nrows,ncols),interpolation=cv2.INTER_CUBIC))
        if 'dog' in i[50:]:
            y.append(1)
        elif 'cat' in i[50:]:
            y.append(0)
    return X,y

# %% [code]
X,y=processimage(train_imgs)

# %% [code]
#plt.figure(figsize=(20,10))
#column=5
#for i in range(column):
#    plt.subplot(5/column+1,column,i+1)
#    plt.imshow(X[i])

# %% [code]
del train_imgs
gc.collect()
X=np.array(X)
y=np.array(y)
#import seaborn as sns
#sns.countplot(y)

# %% [code]
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.20,random_state=2)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
del X
del y
gc.collect()
ntrain=len(X_train)
nval=len(X_val)

# %% [code]
#Keras is an open source neural network library
batch_size=32
from keras import layers
from keras import optimizers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array,load_img



# %% [code]

