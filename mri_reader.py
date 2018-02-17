from brain_img_processor import BrainData, isolate_brain, segment, denoise, normalize_255, equalize
import matplotlib.pyplot as plt #library to view images
from skimage.measure import label, regionprops

brain_data = BrainData('data/brain.mha')

print(brain_data.get_dimensions())

slice_top = brain_data.get_slice(BrainData.TOP_PROF, 120)
slice_front = brain_data.get_slice(BrainData.SIDE_PROF, 100)

brain_image = slice_front;
########################################################
brain = isolate_brain(brain_image)
normalized= normalize_255(brain['data'])

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})

ax[0, 0].imshow(normalized, cmap="gray")
ax[0, 0].set_title("original")

ax[0, 1].imshow(normalized, cmap="gray")
ax[0, 1].set_title("normalized")

equalized = equalize(normalized)

segmented = segment(equalized)

ax[1, 0].imshow(equalized, cmap="gray")
ax[1, 0].set_title("equalized")

print(segmented.shape)
print(segmented.ndim)

labels = label(segmented);

regions = regionprops(labels, normalized)

ax[1, 1].imshow(normalized, cmap="gray")
ax[1, 1].imshow(segmented, cmap=plt.cm.spectral, alpha=.5)
ax[1, 1].set_title("segmented")

denoised = denoise(equalized)

ax[2, 0].imshow(denoised, cmap="gray")
ax[2, 0].set_title("denoise using nl")

segment(denoised)
ax[2, 1].imshow(segment(denoised))
ax[2,1].set_title("denoise segmented")
plt.tight_layout()
plt.show()


# get gray value histogram of the isolated brain

# process the histogram
# how?

# based on the values of histogram
# does tumor exists?

# if yes
# segment the brain
# get the treshold (mean) to be used to test regions
# if the mean of the region is > than the threshold
# that region is a tumor