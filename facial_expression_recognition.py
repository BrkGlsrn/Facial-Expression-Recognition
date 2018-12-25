# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 15:15:32 2018

@author: TCBGULSEREN
"""


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam,SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import pickle

#Stringli düz yazıdan listeye alıyor boşluğa göre
def Convert(string): 
    li = list(string.split(" ")) 
    return li 

#Dosyayı okuma
train = pd.read_csv("fer2013.csv")
print(train.shape)

#Dataları ayırma
emotions =train.iloc[:,0:1]
pixels =train.iloc[:,1:2]

#csv deki string olan datayı arraye çevirme
pixels = np.array(pixels)
test = []

for i in range (0,35887):
    test.append(Convert(pixels[i][0]))
X_train = pd.DataFrame(data = test)

#Str to float and Normalization
X_train = X_train.astype(np.float)
X_train = X_train / 255.0

'''
#Resimleri görüntüleme
img_size = 48
#resimlerin oldugu Xtrain de herhangi bir fotoyu çiz
plt.imshow(X_train.iloc[1250,:].values.reshape(img_size , img_size))
plt.axis('off')
'''

#Reshape keras 3 boyutlu kabul ediyor 48x48x1 gibi
X_train = X_train.values.reshape(-1,48,48,1)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
emotions =ohe.fit_transform(emotions).toarray()

#Train ve test olarak datayı bölme
from sklearn.model_selection import train_test_split
X_train , X_val ,Y_train , Y_val = train_test_split(X_train,emotions,test_size=0.1,random_state=2)


#Modeli oluşturmaya başlıyoruz
model = Sequential()

#First Convolution
model.add(Conv2D(filters = 64, kernel_size =(5,5),
                 padding = 'Same',activation ='relu',
                 input_shape = (48,48,1)))
model.add(MaxPool2D(pool_size=(5,5),strides=(2, 2)))
#model.add(Dropout(0.25))

#Second Convolution without input shape
model.add(Conv2D(filters = 64, kernel_size =(3,3),
                 padding = 'Same',activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size =(3,3),
                 padding = 'Same',activation ='relu'))
model.add(MaxPool2D(pool_size=(3,3),strides=(2, 2)))
#model.add(Dropout(0.25))

#Third Convolution without input shape
model.add(Conv2D(filters = 128, kernel_size =(3,3),
                 padding = 'Same',activation ='relu'))
model.add(Conv2D(filters = 128, kernel_size =(3,3),
                 padding = 'Same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2, 2)))
#model.add(Dropout(0.25))

#Neural Network part
model.add(Flatten())
model.add(Dense(1024,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1024,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(7,activation = 'softmax'))

#optimizer = Adam(lr=0.005 , beta_1 = 0.9 , beta_2 = 0.999)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=1, nesterov=True)

#Compile the model
model.compile(optimizer='Adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

epochs = 30
batch_size = 100


datagen = ImageDataGenerator(
        featurewise_center = False , #set input mean to 0 over the dataset
        samplewise_center = False, #set each sample mean to zero
        featurewise_std_normalization = False, #divide inputs by std of the dataset
        samplewise_std_normalization = False, #divide each input by its std
        zca_whitening = False, #dimension reduction
        rotation_range = 0.05, #randoml rotate images 5 degrees
        zoom_range = 0.05, #randomly zoom 5 degree
        width_shift_range = 0.05, #randomly horizontal shift 5 degrees
        height_shift_range = 0.05, #randomly vertical shift 5 degrees
        horizontal_flip = False, # randomly flip images
        vertical_flip = False) #randomly flip images

datagen.fit(X_train)

#Model fitleniyor
history = model.fit_generator(datagen.flow(X_train,Y_train,batch_size = batch_size),
                                 validation_data = (X_val,Y_val),
                                 steps_per_epoch = X_train.shape[0] // batch_size,
                                 epochs = epochs)


#Model kaydediliyor
dosya = 'face_expression_recognition'
pickle.dump(model,open(dosya,'wb'))

#Model Yukleniyor
model_pickle = pickle.load(open('face_expression_recognition','rb'))

#Tahmin yapılıyor
y_pred = model_pickle.predict(X_val)

#one hot encoderı tekli arraye dönüştürüyor
y_pred_classes = np.argmax(y_pred,axis=1) 
Y_true =np.argmax(Y_val,axis=1)

#cm ile görüntüleme
cm = confusion_matrix(Y_true,y_pred_classes)
print(cm)









    
    

    








