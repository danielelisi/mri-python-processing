import SimpleITK
from skimage.measure import label, regionprops
from skimage.restoration import denoise_bilateral
from scipy import ndimage as ndi
from skimage.filters import rank, gaussian
from skimage.morphology import watershed, disk
import numpy as np


class BrainData:
    '''
    Class to store the data of MRI image and provide function to easily
    access slices of the MRI image
    '''

    TOP_PROF = 0
    FRONT_PROF = 1
    SIDE_PROF = 2

    def __init__(self, mha_location):

        # try catch this shit
        try:
            input_image = SimpleITK.ReadImage(mha_location)
        except Exception:
            raise Exception(f'Error opening file "{mha_location}"')

        # do some checking here to make sure it is a 3d image

        # get the image data from mha
        self.data = SimpleITK.GetArrayFromImage(input_image)
        self.dimensions = self.data.shape

    def get_slice(self, profile, index):

        '''
        Returns uint8 2d ndarray of a specific slice of the MRI image
        Returns None if error is encountered
        '''

        if index >= self.dimensions[profile]:
            return None

        if profile == self.TOP_PROF:
            return self.data[index, :, :]
        if profile == self.FRONT_PROF:
            return self.data[:, index, :]
        if profile == self.SIDE_PROF:
            return self.data[:, :, index]

    def get_dimensions(self):
        return self.dimensions
        
    
############################################
# HELPER FUNCTIONS
############################################

def segment(brain_img):

    '''
    Segmenting regions of the brain using watershed function
    '''
    # get the low gradient
    markers = rank.gradient(brain_img, disk(5)) < 20

    markers = ndi.label(markers)[0]

    # get the local gradient
    gradient = rank.gradient(brain_img, disk(1))

    # process the watershed
    labels = watershed(gradient, markers)

    return labels


def get_tumor_region(label, image):
    return None


def normalize_255(image):

    input_data = image
   

    min = np.amin(input_data)
    max = np.amax(input_data)
    
    normalized = (input_data-min) / (max-min) * 255

    normalized = np.clip(normalized, a_min=0, a_max=255)

    normalized = normalized.astype("uint8")

    #print(normalized)
    return normalized


def equalize(image):

    # need to disregard 0 values
    h = np.histogram(image, bins=256)[0]
    H = np.cumsum(h) / float(np.sum(h))
 
    e = np.floor(H[image.flatten()] * 255.)
    return e.reshape(image.shape).astype('uint8')

def median(data):

    return rank.median(data, disk(1))

def bilateral(data, win_size=5, multichannel=False):

    return denoise_bilateral(data, win_size=win_size, multichannel=multichannel)



    