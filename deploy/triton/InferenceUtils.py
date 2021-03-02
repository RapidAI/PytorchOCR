import numpy as np


def single_image_normalize(_means, _stds, _img):
    assert len(_img.shape) == 3 and _img.shape[2] == 3, 'must be bgr image'
    to_return_array = (_img - _means) / _stds
    to_return_array = np.transpose(to_return_array, (2, 0, 1))[None, ...]
    return to_return_array
