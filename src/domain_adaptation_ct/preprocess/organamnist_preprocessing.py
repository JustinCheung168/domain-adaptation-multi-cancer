import numpy as np

def rescale_image_pixel_values(image: np.ndarray, rescaled_min: float, rescaled_max: float):
    """
    Rescale a min-max normalized image's pixel values (assumed to already be in range [0, 1]) to range [output_min, output_max].

    Params:
        `image` - An array which is assumed to already have been min-max normalized to range [0,1].
        `rescaled_min` - The new minimum value of the pixel values' range.
        `rescaled_max` - The new maximum value of the pixel values' range.
    """
    scale_factor = rescaled_max - rescaled_min
    return (image * scale_factor) + rescaled_min

def normalize_image(image: np.ndarray):
    """
    Our normalization logic.

    `image` is assumed to already be a float array in range [0, 1]. 
    """
    return rescale_image_pixel_values(image, rescaled_min = -1.0, rescaled_max = 1.0)
