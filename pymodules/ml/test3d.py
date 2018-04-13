import SimpleITK
from keras.models import load_model
from util_3d import *
import sys

'''
Parameters
    encodedPrediction - the one hot encoded label to use to predict 
    labels - labels for the prediction
    
Transform encodedPredictions to a dictionary of labeled data
'''
def getLabelPrediction(encodedPrediction, labels=['Less than 250 Days', '250 to 500 Days', 'More than 500 Days']):
    formattedLabels = []
    numberOfLabels = len(encodedPrediction[0])
    for index in range(0, numberOfLabels):
        formattedLabels.append("%.2f" % round((encodedPrediction[0][index] * 100), 2))

    labelPrediction = {}
    for index in range(0, len(labels)):
        labelPrediction[labels[index]] = formattedLabels[index]

    print(labelPrediction)
    return labelPrediction


'''
Parameters
    model_filepath - location of the trained model 
    tumor_img_filepath - location of the datafile

Trim and reshape tumor img
Get encoded prediction of labels based off model
'''
def getEncodedPrediction(model_filepath, tumor_img_filepath,  dim=(50,80,80)):
    model = load_model(model_filepath)
    input_image = SimpleITK.ReadImage(tumor_img_filepath)
    rawArray = SimpleITK.GetArrayFromImage(input_image)
    trimmedArray = trim_array_3d(rawArray, dim)

    img_depth = len(trimmedArray)
    img_cols = len(trimmedArray[0])
    img_rows = len(trimmedArray[0][0])
    nb_channels = 1
    nb_samples = 1

    inputData = trimmedArray.reshape(nb_samples, img_depth, img_cols, img_rows, nb_channels)
    encodedPrediction = model.predict(inputData)
    print(encodedPrediction)
    return encodedPrediction

'''
Parameters
    model_filepath - location of the trained model
    tumor_img_filepath - location of the datafile 
'''
if __name__ == "__main__":
    model_filepath = sys.argv[1]
    tumor_img_filepath = sys.argv[2]
    encodedPrediction = getEncodedPrediction(model_filepath, tumor_img_filepath)
    labelPrediction = getLabelPrediction(encodedPrediction)