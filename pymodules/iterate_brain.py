from brain_img_processor import BrainData, isolate_brain, segment, denoise, normalize_255, equalize
import numpy as np
import matplotlib.pyplot as plt #library to view images

from scipy import ndimage as ndi




def get_slices(brain_data):
    brain = brain_data
    hide_image_num = 100
    #retrieve all top down images from mri scan
    top_images = []

    for i in range(brain.dimensions[0]):
        slice_top = brain.get_slice(BrainData.TOP_PROF, i)
        if (sum(sum(slice_top)) > hide_image_num):
            top_images.append(slice_top)

    # retrieve all side on images from mri scan
    side_images = []

    for i in range(brain.dimensions[1]):
        slice_side = brain.get_slice(BrainData.SIDE_PROF, i)
        if (sum(sum(slice_side))> hide_image_num):
            side_images.append(slice_side)
    # retrieve all front facing images from mri scan
    front_images = []

    for i in range(brain.dimensions[2]):
        slice_front = brain.get_slice(BrainData.FRONT_PROF, i)
        if (sum(sum(slice_front)) > hide_image_num):
            front_images.append(slice_front)
    
    return [top_images, side_images, front_images]