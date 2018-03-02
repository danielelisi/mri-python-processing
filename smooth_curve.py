import numpy

def smooth_curve(moving_num, x_data_arr, y_data_arr):
    smoothed_y = []
    shortened_x = x_data_arr[moving_num:]
    for i in range(y_data_arr.size - moving_num):
        # going through the data points in reverse because we want the end data points more then the start points and this method
        # loses a data point for each increment of moving_num 
        current_point = sum(y_data_arr[-(i+moving_num+1):-(i+1)])/moving_num
        smoothed_y.append(current_point)
    # flip the final points back to the original order
    return [shortened_x[::-1], smoothed_y]

def gradient_check(x_values, y_values):
    peak_count = 0
    pos_grad = False
    peak_values = []
    for i in range(y_values.size - 1):
        # check if the next value - current value is positive if it is then its a pos gradient and there will be a peak
        # using 1.1 as a ratio instead of 1 to make it ignore small bumps
        if (y_values[(i+1)] / y_values[i]) >= 1 :
            # only tallies if not already added one
            if (not pos_grad):
                if (y_values[(i+1)] / y_values[i]) >= 1.05:
                    pos_grad = True
                    peak_count +=1
        else:
            # sets to false when a decline starts so you can add another to peak_count if a new incline is detected
            if (pos_grad):
                # these are not magic numbers they are calculated bullshit numbers that kinda work
                if 0.985 >=(y_values[(i+1)] / y_values[i]) >= 0.95:
                    if(x_values[i+1] > 100):
                        print(y_values[(i+1)] / y_values[i])
                        peak_values.append(x_values[i+1])
                        pos_grad = False
                    else:
                        pos_grad = False
                else:
                    pos_grad = False
    if (peak_count > 1):
        return True , peak_values
    else:
        return False , peak_values  


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

