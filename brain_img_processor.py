import SimpleITK
from skimage.measure import label, regionprops

from scipy import ndimage as ndi
from skimage.filters import rank, gaussian
from skimage.morphology import watershed, disk

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
        input_image = SimpleITK.ReadImage( mha_location )

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


def isolate_brain(img_array):

    '''
    Lazy way of isolating the brain from the 2d mri image
    '''

    result = {'data': None, 'origin': (0, 0)}

    # binarize the image so we can properly separate the brain region.
    # hardcode the threshold for now. this is just 
    # a fast isolation method
    bin_data = img_array > 10

    # start labelling regions
    label_image = label(bin_data)

    # get the regions
    regions = regionprops(label_image)

    # if the number of regions is zero, there is no brain
    if(len(regions)) == 0:
        return result
    # the number of regions should be 1
    # if its greater than 1, find the largest one
    selected_region = 0
    max_area = 0

    if len(regions) > 1:
        for index, region in enumerate(regions):
            if(region.area > max_area):
                selected_region = index
                max_area = region.area

    coords = regions[selected_region].bbox
    result['origin'] = (coords[2]-coords[0], coords[3] - coords[1])
    result['data'] = img_array[coords[0] : coords[2], coords[1] : coords[3]]

    return result

# TODO need to change values for denoising to get optimum 
# values (we need to decrease the number of regions detected to make it faster)

def segment(brain_img):

    '''
    Segmenting regions of the brain using watershed function
    '''

    # denoise the image
    denoised = gaussian(brain_img)

    # get the low gradient
    markers = rank.gradient(denoised, disk(3)) < 1

    markers = ndi.label(markers)[0]

    # get the local gradient
    gradient = rank.gradient(denoised, disk(1))

    # process the watershed
    labels = watershed(gradient, markers)

    return labels

def check_brain(brain_img):

    return None

def get_tumor_region(label, image):

    return None


