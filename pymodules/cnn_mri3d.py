import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from sklearn.model_selection import train_test_split
from numpy import argmax
import pandas as pd
import numpy as np
import SimpleITK
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import os

#Iterate over directory of brain tumor images
#First part of the filename is survival days of the patient which helps determine the label the image will be stored as
rootdir = 'C:/Users/gagan/Desktop/mri_training/labeled_data_test'
listOfData = []
labels = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = (os.path.join(subdir, file))
        splitFileName = file.split("_")
        survivalDays = int(splitFileName[0])
        input_image = SimpleITK.ReadImage(filepath)
        data = SimpleITK.GetArrayFromImage(input_image)
        listOfData.append(data)
        if (survivalDays < 150):
            labels.append('0-150')
        elif (survivalDays >= 150 and survivalDays < 300):
            labels.append('150-300')
        elif (survivalDays >= 300 and survivalDays < 450):
            labels.append('300-450')
        elif (survivalDays >= 450 and survivalDays < 600):
            labels.append('450-600')
        else:
            labels.append('600+')

print(labels.count('0-150'))
print(labels.count('150-300'))
print(labels.count('300-450'))
print(labels.count('450-600'))
print(labels.count('600+'))

#Change list to array and reshape to add channel
df_x = np.asarray(listOfData)
df_x = df_x.reshape(len(listOfData), 155, 240, 240, 1)

#Change list to nd array, change labels to integers ex) label1 becomes 0, label2 becomes 1, label3 becomes 2 etc...
y = np.array(labels)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)

#Change integers to one hot encoded values ex) if there are 3 total integers then (0 = 1 0 0), (1 = 0 1 0), (2 = 0 0 1)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
df_y = onehot_encoder.fit_transform(integer_encoded)

#Change into nd array
df_x = np.array(df_x)
df_y = np.array(df_y)

#Create test/train split
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.25, random_state=4)

# number of convolutional filters to use
nb_filters = [8, 32, 64, 128, 256, 512]

# level of pooling to perform (POOL x POOL)
nb_pool = [2, 3]

# level of convolution to perform (CONV x CONV)
nb_conv = [3, 5]

img_depth = 155
img_rows = 240
img_cols = 240
numberOfLabels = len(set(labels))


#CNN Model v3
model = Sequential()
model.add(Conv3D(
        nb_filters[1],
        (nb_conv[0], nb_conv[0], nb_conv[0]),
        data_format='channels_last',
        input_shape=(img_depth, img_rows, img_cols, 1),
        activation='relu'
))
model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
model.add(Conv3D(
        nb_filters[2],
        (nb_conv[0], nb_conv[0], nb_conv[0]),
        activation='relu'
))
model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
model.add(Conv3D(
        nb_filters[3],
        (nb_conv[0], nb_conv[0], nb_conv[0]),
        activation='relu'
))
model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
model.add(Conv3D(
        nb_filters[4],
        (nb_conv[0], nb_conv[0], nb_conv[0]),
        activation='relu'
))
model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
model.add(Conv3D(
        nb_filters[5],
        (nb_conv[0], nb_conv[0], nb_conv[0]),
        activation='relu'
))
model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(numberOfLabels))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.summary()

'''
#CNN Model v2
model = Sequential()
model.add(Conv3D(
        nb_filters[1],
        (nb_conv[1], nb_conv[1], nb_conv[1]),
        data_format='channels_last',
        input_shape=(img_depth, img_rows, img_cols, 1),
        activation='relu'
))
model.add(MaxPooling3D(pool_size=(nb_pool[1], nb_pool[1], nb_pool[1])))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer="normal"))
model.add(Dropout(0.5))
model.add(Dense(numberOfLabels, kernel_initializer="normal"))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['mse', 'accuracy'])
model.summary()
'''

'''
#CNN model v1
model = Sequential()
model.add(Conv3D(
        nb_filters[0],
        (nb_conv[0], nb_conv[0], nb_conv[0]),
        data_format='channels_last',
        input_shape=(img_depth, img_rows, img_cols, 1),
        activation='relu'
))
model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(numberOfLabels))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.summary()
'''

#Train the model on x epochs and save the entire model (architecture/weights/biases/optimizer)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, verbose=1)
model.save('mri_model.h5')

