# Author: Jed Iquin
#
# Module that trims a 3d array where a threshold can be specified
# threshold is the minimum number of pixels in a 2d array that can be counted as a 
# non empty plane

import SimpleITK
import numpy as np
import sys
import cProfile

def trim_array_3d(img_array, dim=(50,50,50), thresh=50):
    '''
    Trim a 3d array

    default threshold is 50 pixels

    set dim to 'unbounded' to remove restrictions
    otherwise its default to 50x50x50

    '''

    d, h, w = img_array.shape

    print(d, h, w)
    #iterate through depth

    min_d, max_d = get_min_max(d, img_array, thresh)
    min_h, max_h = get_min_max(h, np.transpose(img_array, axes=(1,0,2)), thresh)
    min_w, max_w = get_min_max(w, np.transpose(img_array, axes=(2,0,1)), thresh)

    print(f'({d},{h},{w}) trimmed to ({max_d-min_d},{max_h-min_h},{max_w-min_w})')
    print(min_d, max_d, min_h, max_h, min_w, max_w)

    result = img_array[min_d:max_d, min_h:max_h, min_w:max_w]

    if dim != 'unbounded':
        result = reshape_and_pad(result, dim)

    print(f'Result image shape is {result.shape}')

    return result
 
def reshape_and_pad(img_array, dim):

    result = img_array

    img_dim = result.shape
    print(img_dim)

    diff_d = dim[0] - img_dim[0]
    diff_h = dim[1] - img_dim[1]
    diff_w = dim[2] - img_dim[2]

    print(diff_d, diff_h, diff_w)
    # reshape and pad depth
    is_odd = (diff_d % 2 == 1)
    half = abs(diff_d) // 2
    if diff_d < 0:
        result = result[half: -(half+1 if is_odd else half)]
    else:
        result = np.pad(result, ((half, half+1 if is_odd else half ),(0,0),(0,0)), 'constant')

    # reshape and pad height
    is_odd = (diff_h % 2 == 1)
    half = abs(diff_h) // 2
    if diff_h < 0:
        result = result[:, half: -(half+1 if is_odd else half),:]
    else:
        result = np.pad(result, ((0,0),(half, half+1 if is_odd else half ),(0,0)), 'constant')

    print(result.shape)
    # reshape and pad width.
    is_odd = (diff_w % 2 == 1)
    half = abs(diff_w) // 2
    if(diff_w < 0):
        result = result[:, :,half: -(half+1 if is_odd else half)]
    else:
        result = np.pad(result, ((0,0),(0,0),(half, half+1 if is_odd else half )), 'constant')

    return result

def get_min_max(size, img_array, thresh):

    found_ub, found_lb = False, False
    min, max = 0, 0

    for i in range(size):

        k = size-i-1
        
        if (found_lb and found_ub) or (i == size//2 and not (found_lb or found_ub) ):
            break

        if not found_lb:

            check_array = img_array[i].flatten()

            if np.trim_zeros(check_array).shape[0] >= thresh:
                found_lb = True
                min = i

        if not found_ub:
            
            check_array = img_array[k].flatten()

            if np.trim_zeros(check_array).shape[0] >= thresh:
                found_ub = True
                max = k

    return min, max

if __name__ == '__main__':

    test = np.arange(80000).reshape(80, 20, 50)
    test = np.pad(test, ((100,60),(40,80),(60,50)),'constant')

    # run a profiler to check statistics
    print("#######################")
    print(" RUNNING TRIM FUNCTION")
    print("#######################")
    cProfile.run('trim_array_3d(test)')
    result = trim_array_3d(test)
    print(result.shape)

    # trimmed = trim_array_3d(data)