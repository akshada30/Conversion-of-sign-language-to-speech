# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 01:32:15 2021

@author: akshada
"""

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=(1./255), shear_range=0.2, zoom_range=0.2, horizontal_flip=(True))
test_datagen = ImageDataGenerator(rescale=1./255)

path = r"C:\Users\akshada\Downloads\archive (3)\asl"

x_train = train_datagen.flow_from_directory(r"C:\Users\akshada\Downloads\archive (3)\asl\train", target_size=(64, 64), batch_size=32,
                                            class_mode='categorical')
x_test = test_datagen.flow_from_directory(r"C:\Users\akshada\Downloads\archive (3)\asl\test", target_size=(64, 64), batch_size=32,
                                            class_mode='categorical' )
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPool2D, Dropout, Flatten

model = Sequential()
model.add(Convolution2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Convolution2D(64, (3,3), input_shape=(64,64,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Convolution2D(128, (3,3), input_shape=(64,64,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=29, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
H = model.fit_generator(x_train, steps_per_epoch=len(x_train)//32, epochs=50, validation_data=x_test, validation_steps=100)
#H.training_indices()
model.save('ASL201.h5')

import matplotlib.pyplot as plt
plt.plot(H.history['accuracy'])
plt.plot(H.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy/loss')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.show()
plt.savefig("plot_1.jpg")

asl_dict = dict()
asl_dict = x_train.class_indices
asl_dict
Index=[]
for i in asl_dict.keys():
    Index.append(i)  
Alp = []
for i in asl_dict.values():
    Alp.append(i)

import pandas as pd
data = pd.DataFrame(zip(Index, Alp), columns=['Index', 'Alp'])

data.to_csv("data_asl1.csv")

