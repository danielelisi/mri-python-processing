# Authors Gagan Heer, Dylan Chew, Alex Lilley, Alex Mori, Daniele Lisi, Jed Ikin
#
# This program takes a 2D image array and removes the skull.
# Note: This does not work for a 3D image array.
#
# Below are the steps we used to perform the skull stripping. We got this algorithm
# from an article written by Shaswati Roy and Pradipta Maji called "A Simple Skull Stripping
# Algorithm for Brain MRI".

# 1) Apply median filtering with a window of size 3×3 to the input image.
# 2) Compute the initial mean intensity value Ti of the image.
# 3) Identify the top, bottom, left, and right pixel locations, from where brain
# skull starts in the image, considering gray values of the skull are greater than Ti.
# 4) Form a rectangle using the top, bottom, left, and right pixel locations.
# 5) Compute the final mean value Tf of the brain using the pixels located
# within the rectangle.
# 6) Approximate the region of brain membrane or meninges that envelop
# the brain, based on the assumption that the intensity of skull is more
# than Tf and that of membrane is less than Tf.
# 7) Set the average intensity value of membrane as the threshold value T.
# 8) Convert the given input image into binary image using the threshold T.
# 9) Apply a 13×13 opening morphological operation to the binary image in order
# to separate the skull from the brain completely.
# 10) Find the largest connected component and consider it as brain.
# 11) Finally, apply a 21×21 closing morphological operation to fill the
# gaps within and along the periphery of the intracranial region.

# Import statements

import numpy

# Library to view images
import matplotlib.pyplot as plt

# Library for reading image
from scipy import misc

# Library to apply filter to images
from skimage.filters import rank
from skimage.morphology import closing, square, opening
from skimage.segmentation import clear_border
from skimage.measure import regionprops, label
from skimage.morphology import disk


# Step 1
def apply_med_filter(image):

    # Apply median filter
    return rank.median(image, disk(3))


# Step 2
def get_init_mean(image):

    # Get the initial mean intensity of the filtered image
    return image.mean()


# Step 3
def identify_pixel_locations(image, mean_init_intensity):

    # Get the coordinates of the bounding box of the brain, including the skull
    threshold = mean_init_intensity
    bounding = closing(image > threshold, square(13))
    cleared = clear_border(bounding)

    # Get the coordinates of the bounding box
    return label(cleared)


# Step 4
def form_rect(pixel_box):
    rect_coord = []
    max_area = 0

    for region in regionprops(pixel_box):
        if region.filled_area >= max_area:
            max_area = region.filled_area
            rect_coord = region.bbox

    return rect_coord


# Step 5
def get_final_mean(image, pixel_box):
    selected_region = image[pixel_box[0]:pixel_box[2], pixel_box[1]:pixel_box[3]]
    return selected_region.mean(), selected_region


# Step 6
def approx_brain_reg(final_mean_intensity, selected_region):

    # Get the value t which is the the mean intensity of the membrane
    height, width = selected_region.shape
    membrane = []

    for i in range(height):
        for j in range(width):
            if selected_region[i, j] < final_mean_intensity:
                membrane.append(selected_region[i, j])

    return membrane


# Step 7
def get_average_mean(membrane):
    return sum(membrane) / float(len(membrane))


# Step 8
def make_binary_image(image, average_mean_intensity):
    binary_image = image > average_mean_intensity

    # Get the largest region
    max_area = 0
    label_image = label(binary_image)
    s_region = None
    rect_coord = None

    for region in regionprops(label_image):

        if region.area > max_area:
            max_area = region.area
            s_region = region
            rect_coord = region.bbox

    return binary_image, s_region, rect_coord


# Step 9
def apply_open_morph(binary_image):

    # Apply opening morphology to binary image
    return opening(binary_image, square(3))


# Step 10
def find_brain(s_region, med_image, pixel_box):

    min_rect, min_coord, max_rect, max_coord = pixel_box

    binary_mask = numpy.zeros(shape=med_image.shape)
    binary_mask[min_rect:max_rect, min_coord:max_coord] = s_region.image

    return binary_mask


# Step 11
def close_mask(image_in_binary, origin_image):

    # Close the mask
    image_in_binary = closing(image_in_binary, square(21))
    image_in_binary = 1 - image_in_binary

    # Approximate the region of brain membrane or
    # meninges that envelop the brain, based on the assumption
    # that the intensity of skull is more than Tf
    # and that of membrane is less than Tf.
    return numpy.ma.masked_array(origin_image, mask=image_in_binary)


# This is the function to call to run the file
def run(file):

    image = 'brain.mha'

    try:
        image = misc.imread(file, mode="L")
    except FileNotFoundError:
        print('File not found, please try again')

    med_filter_image = apply_med_filter(image)
    init_mean = get_init_mean(med_filter_image)
    pixel_box = identify_pixel_locations(med_filter_image, init_mean)
    rect = form_rect(pixel_box)
    final_mean, select_region = get_final_mean(med_filter_image, rect)
    brain_reg = approx_brain_reg(final_mean, select_region)
    average_mean = get_average_mean(brain_reg)
    binary_image, s_region, pixel_box = make_binary_image(med_filter_image, average_mean)
    binary_image = apply_open_morph(binary_image)
    binary_mask = find_brain(s_region, binary_image, pixel_box)
    closed_mask = close_mask(binary_mask, image)

    # Display the image of just the brain
    plt.tight_layout()
    plt.imshow(select_region, cmap="gray")
    plt.imshow(binary_mask, cmap="gray")
    plt.imshow(closed_mask, cmap="gray")
    plt.show()
