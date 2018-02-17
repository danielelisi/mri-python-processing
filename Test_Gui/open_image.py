from brain_img_processor import BrainData, isolate_brain, segment, denoise, normalize_255, equalize
from skimage.restoration import denoise_bilateral
from skimage.filters import rank
from skimage.morphology import disk
import matplotlib.pyplot as plt     # library to view images

from scipy import ndimage as ndi
from skimage.morphology import watershed
from smooth_curve import smooth_curve


def run(img):
    brain = BrainData(img)

    slice_top = brain.get_slice(BrainData.TOP_PROF, 120)
    slice_side = brain.get_slice(BrainData.SIDE_PROF, 100)

    isolated_brain = isolate_brain(slice_side)
    brain_data = isolated_brain['data']

    # first row sequence: bilateral filter->normalization->equalization
    # do a bilateral filtering using 5 as windows size, standard deviation is image's standard deviation
    b_filtered1 = denoise_bilateral(brain_data, win_size=5, multichannel=False)
    normal1 = normalize_255(b_filtered1)
    equalize1 = equalize(normal1)
    median1 = rank.median(equalize1, disk(1))

    # second row sequence: normalization->equalization->bilateral filter
    normal2 = normalize_255(brain_data)
    equalize2 = equalize(normal2)
    b_filtered2 = denoise_bilateral(equalize2, win_size=5, multichannel=False)
    median2 = rank.median(b_filtered2, disk(1))
    fig3, ax3 = plt.subplots(nrows=2, ncols=2)

    brain_vec1 = normal1.ravel()
    brain_vec1 = brain_vec1[brain_vec1 > 0]
    n, bins, patches = ax3[0, 0].hist(brain_vec1, bins=range(256))

    # taking the values from the histogram and applying a floating point average to them to smooth curve for analysis
    floating_point = 20
    x_data = bins[1:]
    y_data = n
    return smooth_curve(floating_point, y_data, x_data)
