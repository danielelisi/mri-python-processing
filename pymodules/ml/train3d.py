import keras
from keras.models import Sequential
from keras.models import load_model
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
from util_3d import *
import h5py
import warnings
import sys

label_encoder = LabelEncoder()

'''
Parameters
    rootdir - directory path to labeled dataset

Iterate over directory of brain tumor images
First part of the filename is survival days of the patient which helps determine the label the image will be stored as
'''
def loadData(rootdir):
    listOfData = []
    labels = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            print(' ')
            print(file)
            filepath = (os.path.join(subdir, file))
            splitFileName = file.split("_")
            survivalDays = int(splitFileName[0])
            input_image = SimpleITK.ReadImage(filepath)
            data = SimpleITK.GetArrayFromImage(input_image)
            # data = trim_array_3d(data)
            listOfData.append(data)

            if (survivalDays <= 250):
                labels.append('Less than 250 Days')
            elif (survivalDays >= 251 and survivalDays <= 500):
                labels.append('250 to 500 Days')
            else:
                labels.append('More than 500 Days')

    print('Label 1 amount:', labels.count('Less than 250 Days'))
    print('Label 2 amount:', labels.count('250 to 500 Days'))
    print('Label 3 amount:', labels.count('250 to 500 Days'))
    return listOfData, labels


'''
Parameters
    listOfData - list of labels for data
Change list to array and reshape to add channel
'''
def reshapeData(listOfData):
    df_x = np.asarray(listOfData)
    df_x = df_x.reshape(len(listOfData), len(listOfData[0]), len(listOfData[0][0]), len(listOfData[0][0][0]), 1)
    return df_x

'''
Parameters
    labels - list of labels for data

Change list of labels to nd array
Change nd array of labels to integers ex) label1 becomes 0, label2 becomes 1, label3 becomes 2 etc...
Change integers to one hot encoded values ex) if there are 3 total integers then (0 = 1 0 0), (1 = 0 1 0), (2 = 0 0 1)
'''
def encodeLabels(labels):
    y = np.array(labels)
    integer_encoded = label_encoder.fit_transform(y)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    df_y = onehot_encoder.fit_transform(integer_encoded)
    return df_y

'''
Parameters
    nb_filters - number of convolutional filters to use
    nb_pool - level of pooling to perform (POOL x POOL)
    nb_conv - level of convolution to perform (CONV x CONV)
    nb_channels - number of channels to use (1 = greyscale, 3 = RGB)
Creates model structure of the neural network
'''
def createModel(nb_filters=[8, 16, 32, 64, 128, 256], nb_pool=[2, 3], nb_conv=[3, 5], nb_channels=1):
    img_depth = len(listOfData[0])
    img_cols = len(listOfData[0][0])
    img_rows = len(listOfData[0][0][0])
    numberOfLabels = len(set(labels))

    # CNN Model v3
    model = Sequential()
    model.add(Conv3D(
        nb_filters[1],
        (nb_conv[0], nb_conv[1], nb_conv[1]),
        data_format='channels_last',
        input_shape=(img_depth, img_rows, img_cols, nb_channels),
        activation='relu'
    ))
    model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[1], nb_pool[1])))
    model.add(Conv3D(
        nb_filters[2],
        (nb_conv[0], nb_conv[1], nb_conv[1]),
        activation='relu'
    ))
    model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[1], nb_pool[1])))
    model.add(Conv3D(
        nb_filters[3],
        (nb_conv[0], nb_conv[1], nb_conv[1]),
        activation='relu'
    ))
    model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[1], nb_pool[1])))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(numberOfLabels))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.summary()
    return model

'''
Parameters
    model - the model structure to be used for training
    df_x - reshaped dataset
    df_y - one hot encoded labels
Creates a train test split
Trains the dataset using the provided model
Saves the trained model
Gives a sample prediction
'''
def trainSaveModel(model, df_x, df_y):
    # Change into nd array
    arrayX = np.array(df_x)
    arrayY = np.array(df_y)

    # Create test/train split
    x_train, x_test, y_train, y_test = train_test_split(arrayX, arrayY, test_size=0.25, random_state=4)

    # Train the model on x epochs and save the entire model (architecture/weights/biases/optimizer)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=7, verbose=1)
    model.save('mri_modelv5.h5')
    # model.load_model('mri_model.h5')

    # Print Matches to debug
    encodedPrediction = model.predict(x_test[:])
    match = 0
    for index in range(0, len(encodedPrediction)):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        actual = label_encoder.inverse_transform([argmax(y_test[index])])
        prediction = label_encoder.inverse_transform([argmax(encodedPrediction[index])])
        print(encodedPrediction[index])
        print('Highest Predicted Label: ', prediction)
        print('Actual Label: ', actual)
        if actual == prediction:
            match += 1
        print('Total Matches: ', match)
        print(' ')

'''
Parameters
    rootdir - directory to dataset
'''
if __name__ == "__main__":
    rootdir = sys.argv[1]
    listOfData, labels = loadData(rootdir)
    df_x = reshapeData(listOfData)
    df_y = encodeLabels(labels)
    model = createModel()
    trainSaveModel(model, df_x, df_y)