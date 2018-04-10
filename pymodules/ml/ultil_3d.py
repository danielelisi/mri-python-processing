# Author: Jed Iquin
#
# Module that trims a 3d array where a threshold can be specified
# threshold is the minimum number of pixels in a 2d array that can be counted as a 
# non empty plane

import SimpleITK
import numpy as np
import sys

def trim_array_3d(img_array, thresh=50):
    '''
    Trim a 3d array

    default threshold is 50 pixels
    '''

    d, h, w = img_array.shape

    print(d, h, w)
    #iterate through depth

    min_d, max_d = get_min_max(d, img_array, 'depth', thresh)
    min_h, max_h = get_min_max(h, img_array, 'height', thresh)
    min_w, max_w = get_min_max(w, img_array, 'width', thresh)

    print(f'({d},{h},{w}) trimmed to ({max_d-min_d},{max_h-min_h},{max_w-min_w})')
    print(min_d, max_d, min_h, max_h, min_w, max_w)

    return img_array[min_d:max_d, min_h:max_h, min_w:max_w]
 

def get_min_max(size, img_array, axes, thresh):

    found_ub, found_lb = False, False
    min, max = 0, 0

    for i in range(size):

        k = size-i-1
        
        if (found_lb and found_ub) or (i == size//2 and not (found_lb or found_ub) ):
            break

        if not found_lb:

            if(axes == 'depth'):
                check_array = img_array[i].flatten()
            elif(axes == 'height'):
                check_array = img_array[:,i,:].flatten()
            else:
                check_array = img_array[:,:,i].flatten()

            if np.trim_zeros(check_array).shape[0] >= thresh:
                found_lb = True
                min = i

        if not found_ub:
            
            if(axes == 'depth'):
                check_array = img_array[k].flatten()
            elif(axes == 'height'):
                check_array = img_array[:,k,:].flatten()
            else:
                check_array = img_array[:,:,k].flatten()

            if np.trim_zeros(check_array).shape[0] >= thresh:
                found_ub = True
                max = k

    return min, max

if __name__ == '__main__':

    '''
    pass in an argument which is the location of mha file
    '''

    try:
        input_image = SimpleITK.ReadImage(sys.argv[1])
    except:
        raise Exception('file not found')

    data = SimpleITK.GetArrayFromImage(input_image)

    trimmed = trim_array_3d(data)