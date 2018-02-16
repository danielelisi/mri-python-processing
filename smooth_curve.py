def smooth_curve(float_num, x_data_arr, y_data_arr):
    smoothed_y = []
    shortened_x = x_data_arr[:x_data_arr.size - float_num]
    for i in range(y_data_arr.size - float_num):
        current_point = sum(y_data_arr[i:i+float_num])/float_num
        smoothed_y.append(current_point)
    return [shortened_x, smoothed_y]
