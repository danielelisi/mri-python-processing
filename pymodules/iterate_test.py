from brain_img_processor import BrainData, isolate_brain, segment, denoise, normalize_255, equalize
import numpy as np
import matplotlib.pyplot as plt #library to view images

from scipy import ndimage as ndi
from iterate_brain import get_slices


brain = BrainData('data/brain.mha')

images = get_slices(brain)

print (len(images[0]),len(images[1]),len(images[2]))