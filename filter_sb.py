from brain_img_processor import BrainData, isolate_brain, segment, denoise, normalize_255, equalize
import numpy as np
import matplotlib.pyplot as plt #library to view images
from skimage.restoration import denoise_bilateral
from skimage.filters import rank
from skimage.morphology import disk

from scipy import ndimage as ndi
from skimage.morphology import watershed
from smooth_curve import smooth_curve
brain = BrainData('data/brain.mha')

slice_top = brain.get_slice(BrainData.TOP_PROF, 120)
slice_side = brain.get_slice(BrainData.SIDE_PROF, 100)

isolated_brain = isolate_brain(slice_side)
brain_data = isolated_brain['data']

#first row sequence: bilateral filter->normalization->equalization
# do a bilateral filtering using 5 as windows size, standard deviation is image's standard deviation
b_filtered1 = denoise_bilateral(brain_data, win_size=5, multichannel=False)
normal1 = normalize_255(b_filtered1)
equalize1 = equalize(normal1)
median1 = rank.median(equalize1, disk(1))

#second row sequence: normalization->equalization->bilateral filter
normal2 = normalize_255(brain_data)
equalize2 = equalize(normal2)
b_filtered2 = denoise_bilateral(equalize2, win_size=5, multichannel=False)
median2 = rank.median(b_filtered2, disk(1))

fig2, ax2 = plt.subplots(nrows=2, ncols=4, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
fig3, ax3 = plt.subplots(nrows=2, ncols=2)

fig.suptitle("Denoising and smoothing")
fig2.suptitle("segmentation")
ax[0,0].imshow(brain_data, cmap="gray")
ax[0,0].set_title("isolated brain")

ax[0,1].imshow(b_filtered1, cmap="gray")
ax[0,1].set_title("bilateral filtered")

ax[0, 2].imshow(normal1, cmap="gray")
ax[0, 2].set_title("normalized")

ax[0, 3].imshow(equalize1, cmap="gray")
ax[0, 3].set_title("histogram equalized")

ax[0, 4].imshow(median1, cmap="gray")
ax[0, 4].set_title("median filtered")

ax[1, 0].imshow(brain_data, cmap="gray")
ax[1, 0].set_title("isolated brain")

ax[1, 1].imshow(normal2, cmap="gray")
ax[1, 1].set_title("normalized")

ax[1, 2].imshow(equalize2, cmap="gray")
ax[1, 2].set_title("histogram equalized")

ax[1, 3].imshow(b_filtered2, cmap="gray")
ax[1, 3].set_title("bilateral filtered")

ax[1, 4].imshow(median2, cmap="gray")
ax[1, 4].set_title("median filtered")


marker1 = rank.gradient(median1, disk(1)) < 10
marker1 = ndi.label(marker1)[0]
gradient1 = rank.gradient(median1, disk(1))
watershed1 = watershed(gradient1, marker1)

marker2 = rank.gradient(median2, disk(1)) < 10
marker2 = ndi.label(marker2)[0]
gradient2 = rank.gradient(median2, disk(1))
watershed2 = watershed(gradient2, marker2)

ax2[0,0].imshow(median1, cmap="gray")
ax2[0,0].set_title("median filtered")

ax2[0,1].imshow(marker1)
ax2[0,1].set_title("marker")

ax2[0,2].imshow(gradient1)
ax2[0,2].set_title("local gradient")

ax2[0,3].imshow(watershed1)
ax2[0,3].set_title("watershed")

ax2[1,0].imshow(median2, cmap="gray")
ax2[1,0].set_title("median filtered")

ax2[1,1].imshow(marker2)
ax2[1,1].set_title("marker")

ax2[1,2].imshow(gradient2)
ax2[1,2].set_title("local gradient")

ax2[1,3].imshow(watershed2)
ax2[1,3].set_title("watershed")

brain_vec1 = normal1.ravel()
brain_vec1 = brain_vec1[brain_vec1 > 0]
n , bins, patches = ax3[0,0].hist(brain_vec1, bins=range(256))

# taking the values from the histogram and applying a floating point average to them to smooth curve for analysis
floating_point= 20
x_data = bins[1:]
y_data = n
new_curve = smooth_curve(20, y_data,x_data)

plt.plot(new_curve[0],new_curve[1])
plt.tight_layout()
plt.show()