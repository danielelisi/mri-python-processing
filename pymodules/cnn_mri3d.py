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

#Set labels for all mri scans
labels = ['less500', 'less500', 'less500', 'less500', 'less500', 'less500', 'less500', 'less500', 'less500', 'less500', 'less500', 'less500',
          'more500', 'more500', 'more500', 'more500', 'more500', 'more500', 'more500', 'more500', 'more500', 'more500', 'more500', 'more500']

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

#***TO DO: add more layers (conv3d, maxpooling and dense), test different activations, test different optimizers
#Create 3D CNN model architecture
model = Sequential()
model.add(Conv3D(8, (3, 3, 3), data_format='channels_last', activation='relu', input_shape=(155, 240, 240, 1)))
model.add(MaxPooling3D(pool_size=(5,5,5)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.summary()

#Train the model on x epochs and save the entire model (architecture/weights/biases/optimizer)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, verbose=1)
model.save('mri_model.h5')

