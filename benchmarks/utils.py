import numpy as np


def upsample_array(input_array, num_points):
    if num_points <= len(input_array):
        return input_array
    original_points = np.arange(len(input_array))
    upsampled_points = np.linspace(0, len(input_array) - 1, num_points)
    upsampled_array = np.interp(upsampled_points, original_points, input_array)
    return upsampled_array
