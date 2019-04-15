import numpy as np
import cv2
from keras import backend as K

def center_crop(ndarray, crop_size):
    '''Input ndarray is of rank 3 (height, width, depth).
    Argument crop_size is an integer for square cropping only.
    Performs padding and center cropping to a specified size.
    '''
    h, w, d = ndarray.shape
    if crop_size == 0:
        raise ValueError('argument crop_size must be non-zero integer')
    
    if any([dim < crop_size for dim in (h, w)]):
        # zero pad along each (h, w) dimension before center cropping
        pad_h = (crop_size - h) if (h < crop_size) else 0
        pad_w = (crop_size - w) if (w < crop_size) else 0
        rem_h = pad_h % 2
        rem_w = pad_w % 2
        pad_dim_h = (pad_h/2, pad_h/2 + rem_h)
        pad_dim_w = (pad_w/2, pad_w/2 + rem_w)
        # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
        npad = (pad_dim_h, pad_dim_w, (0,0))
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)
        h, w, d = ndarray.shape
    # center crop
    h_offset = (h - crop_size) / 2
    w_offset = (w - crop_size) / 2
    cropped = ndarray[h_offset:(h_offset+crop_size),
                      w_offset:(w_offset+crop_size), :]

    return cropped


def lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5):
    lrate = base_lr * (1.0 - (curr_iter / float(max_iter)))**power
    K.set_value(model.optimizer.lr, lrate)

    return K.eval(model.optimizer.lr)
