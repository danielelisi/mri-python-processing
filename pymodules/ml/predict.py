import SimpleITK
from keras.models import load_model
from util_3d import *

def labelPrediction(tumor_img_filepath, labels=['0-150', '150-300', '300-450', '450-600', '600+']):
    encoded = encodedPrediction(tumor_img_filepath)
    firstLabel = ("%.2f" % round((encoded[0][0] * 100), 2))
    secondLabel = ("%.2f" % round((encoded[0][1] * 100), 2))
    thirdLabel = ("%.2f" % round((encoded[0][2] * 100), 2))
    fourthLabel = ("%.2f" % round((encoded[0][3] * 100), 2))
    fifthLabel = ("%.2f" % round((encoded[0][4] * 100), 2))
    labelPrediction = {labels[0]: firstLabel, labels[1]: secondLabel, labels[2]: thirdLabel, labels[3]: fourthLabel, labels[4]: fifthLabel}
    print(labelPrediction)
    return labelPrediction

def encodedPrediction(tumor_img_filepath):
    model = load_model('mri_modelv2.h5')
    input_image = SimpleITK.ReadImage(tumor_img_filepath)
    rawArray = SimpleITK.GetArrayFromImage(input_image)
    trimmedArray = trim_array_3d(rawArray)
    inputData = trimmedArray.reshape(1, len(trimmedArray), len(trimmedArray[0]), len(trimmedArray[0][0]), 1)
    encodedPrediction = model.predict(inputData)
    print(encodedPrediction)
    return encodedPrediction
