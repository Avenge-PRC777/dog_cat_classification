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
train_dir='../input/dogs-vs-cats-redux-kernels-edition/train'
test_dir='../input/dogs-vs-cats-redux-kernels-edition/test'
train_dogs=['../input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]
train_cats=['../input/dogs-vs-cats-redux-kernels-edition/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]
test_imgs=['../input/dogs-vs-cats-redux-kernels-edition/test/{}'.format(i) for i in os.listdir(test_dir)]

train_imgs = train_dogs[:12500] + train_cats[:12500]
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
from keras.applications import InceptionResNetV2
conv_base=InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(150,150,3))

# %% [code]
conv_base.summary()

#####################################TRANSFER LEARNING#################################################################
# %% [code]
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
model=models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()

# %% [code]
# TRANSFER LEARNING
batch_size=1000
conv_base.trainable=False #SINCE THE conv_base is already trained dont train it
from keras import optimizers
model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False),metrics=['acc'])
train_IDGObject=ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
validation_IDGObject=ImageDataGenerator(rescale=1./255)
train_data=train_IDGObject.flow(X_train,y_train,batch_size=batch_size)
val_data=validation_IDGObject.flow(X_val,y_val,batch_size=batch_size)
history=model.fit_generator(train_data,steps_per_epoch=ntrain//batch_size,epochs=15,verbose=2,validation_data=val_data,validation_steps=nval//batch_size)

# %% [code]
model.save('/kaggle/working/transfermodel.h5')
from IPython.display import FileLink, FileLinks
FileLinks('.') #lists all downloadable files on server

# %% [code]
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'b',label='Training Accuracy')
plt.plot(epochs,val_acc,'r',label='Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs,loss,'b',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# %% [code]
X_test,y_test=processimage(test_imgs[0:10])
x=np.array(X_test)
test_IDG=ImageDataGenerator(rescale=1./255) #DO not forget the rescale word
i=0
text_labels=[]
plt.figure(figsize=(30,20))
for batch in test_IDG.flow(x,batch_size=1):
    pred=model.predict(batch)
    if(pred>0.5):
        text_labels.append('dog')
    else:
        text_labels.append('cat')
    plt.subplot(2,5,i+1)
    plt.title("This is a "+text_labels[i])
    imgplot=plt.imshow(batch[0])
    i+=1
    if(i%10==0):
        break
plt.show()
#######################################################END################################################################

######################################################VGG Architecture####################################################
# %% [code]
#Keras is an open source neural network library
batch_size=32
from keras import layers
from keras import optimizers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array,load_img

# %% [code]
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid')) #OUTPUT LAYER
model.summary()
#Params are the total parameters till that step that need to be learnt

# %% [code]
model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False),metrics=['acc'])
#learning rate for RMSPropagation technique is 0.0001
#loss function to minimize is binary cross entropy
#Preprocessing "Again" on data
train_IDGObject=ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
validation_IDGObject=ImageDataGenerator(rescale=1./255)
train_data=train_IDGObject.flow(X_train,y_train,batch_size=batch_size)
val_data=validation_IDGObject.flow(X_val,y_val,batch_size=batch_size)

# %% [code]
#Training
#64 epochs, 100 steps per epoch
history=model.fit_generator(train_data,steps_per_epoch=ntrain//batch_size,epochs=64,validation_data=val_data,validation_steps=nval//batch_size)

# %% [code]
#plotting accuracy, saving model and getting details from history object
model.save_weights('model_weights.h5')
model.save('model_keras.h5')
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'b',label='Training Accuracy')
plt.plot(epochs,val_acc,'r',label='Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs,loss,'b',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# %% [code]
X_test,y_test=processimage(test_imgs[0:10])
x=np.array(X_test)
test_IDG=ImageDataGenerator(1./255)
i=0
text_labels=[]
plt.figure(figsize=(30,20))
for batch in test_IDG.flow(x,batch_size=1):
    pred=model.predict(batch)
    if(pred>0.5):
        text_labels.append('dog')
    else:
        text_labels.append('cat')
    plt.subplot(1,5,i+1)
    plt.title("This is a "+text_labels[i])
    imgplot=plt.imshow(x[i,:,:,:])
    i+=1
    if(i%5==0):
        break
plt.show()
