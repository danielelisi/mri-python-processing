import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from numpy import argmax
import pandas as pd
import numpy as np
import SimpleITK
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.models import load_model
import os

rootdir = 'C:/Users/gagan/Desktop/mri_training/labeled_data_test'
listOfData = []
labels = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        curentLabel = ''
        filepath = (os.path.join(subdir, file))
        splitFileName = file.split("_")
        survivalDays = int(splitFileName[0])
        input_image = SimpleITK.ReadImage(filepath)
        data = SimpleITK.GetArrayFromImage(input_image)
        if (survivalDays < 150):
            currentLabel = '0-150'
        elif (survivalDays >= 150 and survivalDays < 300):
            currentLabel = '150-300'
        elif (survivalDays >= 300 and survivalDays < 450):
            currentLabel = '300-450'
        elif (survivalDays >= 450 and survivalDays < 600):
            currentLabel = '450-600'
        else:
            currentLabel = '600+'

        for index in range(50):
            currentIndex = index + 50
            listOfData.append(data[currentIndex])
            labels.append(currentLabel)

print(labels.count('0-150'))
print(labels.count('150-300'))
print(labels.count('300-450'))
print(labels.count('450-600'))
print(labels.count('600+'))

df_x = np.asarray(listOfData)
df_x = df_x.reshape(len(listOfData), 155, 240, 240)

y = np.array(labels)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
df_y = onehot_encoder.fit_transform(integer_encoded)

df_x = np.array(df_x)
df_y = np.array(df_y)

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.25, random_state=4)

#***TO DO: add more layers (conv2d, maxpooling and dense, dropout), test different activations, test different optimizers, change stride length
#Create 2D CNN model architecture (use channels for depth)
model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(8, 8), activation='relu', strides=(1,1), input_shape=(155, 240, 240), padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=16, kernel_size=(8, 8), activation='relu', strides=(1,1)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, verbose=1)
model.save('mri_model.h5')