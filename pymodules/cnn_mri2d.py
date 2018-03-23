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

listOfData = []

#****TO DO: CHANGE THIS TO A LOOP ITERATING OVER A DIRECTORY
#Less then 500 days survived
input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/less500/147_Brats17_TCIA_242_1_flair.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/less500/147_Brats17_TCIA_242_1_t1.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/less500/147_Brats17_TCIA_242_1_t1ce.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/less500/147_Brats17_TCIA_242_1_t2.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/less500/153_Brats17_TCIA_167_1_flair.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/less500/153_Brats17_TCIA_167_1_t1.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/less500/153_Brats17_TCIA_167_1_t1ce.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/less500/153_Brats17_TCIA_167_1_t2.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/less500/434_Brats17_TCIA_184_1_flair.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/less500/434_Brats17_TCIA_184_1_t1.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/less500/434_Brats17_TCIA_184_1_t1ce.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/less500/434_Brats17_TCIA_184_1_t2.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

#More then 500 days survived
input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/more500/519_Brats17_TCIA_469_1_flair.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/more500/519_Brats17_TCIA_469_1_t1.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/more500/519_Brats17_TCIA_469_1_t1ce.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/more500/519_Brats17_TCIA_469_1_t2.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/more500/747_Brats17_TCIA_121_1_flair.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/more500/747_Brats17_TCIA_121_1_t1.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/more500/747_Brats17_TCIA_121_1_t1ce.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/more500/747_Brats17_TCIA_121_1_t2.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/more500/1458_Brats17_TCIA_278_1_flair.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/more500/1458_Brats17_TCIA_278_1_t1.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/more500/1458_Brats17_TCIA_278_1_t1ce.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

input_image = SimpleITK.ReadImage('C:/Users/gagan/Desktop/mri_training/more500/1458_Brats17_TCIA_278_1_t2.nii')
data = SimpleITK.GetArrayFromImage(input_image)
listOfData.append(data)

labels = ['less500', 'less500', 'less500', 'less500', 'less500', 'less500', 'less500', 'less500', 'less500', 'less500', 'less500', 'less500',
          'more500', 'more500', 'more500', 'more500', 'more500', 'more500', 'more500', 'more500', 'more500', 'more500', 'more500', 'more500']

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