import keras
from keras.engine import Input, Model
from numpy import argmax
import pandas as pd
import numpy as np
import SimpleITK
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from util_3d import *

def predict(tumor_img_filepath):
    labels = ['0-150', '150-300', '300-450', '450-600', '600+']
    label_encoder = LabelEncoder()
    y = np.array(labels)
    label_encoder.fit(y)
    model = load_model('mri_model.h5')
    input_image = SimpleITK.ReadImage(tumor_img_filepath)
    rawArray = SimpleITK.GetArrayFromImage(input_image)
    trimmedArray = trim_array_3d(rawArray)
    inputData = trimmedArray.reshape(1, len(trimmedArray), len(trimmedArray[0]), len(trimmedArray[0][0]), 1)
    encodedPrediction = model.predict(inputData)
    print(encodedPrediction)
    prediction = label_encoder.inverse_transform([argmax(encodedPrediction)])
    print(prediction)
    return prediction