"""
Convenient functions that are helpful throughout the code

Any questions about the code can be addressed to Manu Gopakumar 
at manugopa@stanford.edu.

This code and data is released under the Creative Commons 
Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be 
      obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please
      cite our work.

Technical Paper:
Full-colour 3D holographic augmented-reality displays with metasurface 
waveguides

Citation:
Gopakumar, M. et al. Full-colour 3D holographic augmented-reality displays 
with metasurface waveguides. Nature (2024).

"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_image(field, target_shape, pytorch=True):
    """
    Pads a tensor to target_shape in size.
    Padding is done such that when used with crop_image(), odd and even 
    dimensions are handled correctly to properly undo the padding.
    
    Input
    -----
    :param field: the tensor to be padded. May have as many leading dimensions
         as necessary (e.g., batch or channel dimensions)
    :param target_shape: the 2D target output dimensions. If any dimensions are
         smaller than the tensor dimension, no padding is applied
    :param pytorch: if True, uses torch functions, if False, uses numpy
    Output
    ------
    :return: padded tensor

    """

    size_diff = np.array(target_shape) - np.array(field.shape[-2:])
    odd_dim = np.array(field.shape[-2:]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        if pytorch:
            pad_axes = [int(p)  # convert from np.int64
                        for tple in zip(pad_front[::-1], pad_end[::-1])
                        for p in tple]
            return torch.nn.functional.pad(field, pad_axes, mode='constant',
                                           value=0)
        else:
            leading_dims = field.ndim - 2  # only pad the last two dims
            if leading_dims > 0:
                pad_front = np.concatenate(([0] * leading_dims, pad_front))
                pad_end = np.concatenate(([0] * leading_dims, pad_end))
            return np.pad(field, tuple(zip(pad_front, pad_end)), 'constant',
                          constant_values=0)
    else:
        return field


def crop_image(field, target_shape):
    """
    Crops a tensor to target_shape in size.
    Cropping is done such that when used with pad_image(), odd and even 
    dimensions are handled correctly to properly undo the padding.
    
    Input
    -----
    :param field: the tensor to be cropped. May have as many leading dimensions
         as necessary (e.g., batch or channel dimensions)
    :param target_shape: the 2D target output dimensions. If any dimensions are
         smaller than the tensor dimension, no cropping is applied
    Output
    ------
    :return: cropped tensor

    """

    if target_shape is None:
        return field

    size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
    odd_dim = np.array(field.shape[-2:]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        return field[(..., *crop_slices)]
    else:
        return field


def cond_mkdir(path):
    """
    Creates a folder at specified path if it does not exist
    
    Input
    -----
    :param path: folder path

    """
    if not os.path.exists(path):
        os.makedirs(path)


def im2float(im, dtype=np.float32):
    """
    Convert uint16 or uint8 image to float32, with range scaled to 0-1

    Input
    -----
    :param im: image
    :param dtype: default np.float32
    Output
    ------
    :return: float image
    """
    if issubclass(im.dtype.type, np.floating):
        return im.astype(dtype)
    elif issubclass(im.dtype.type, np.integer):
        return im / dtype(np.iinfo(im.dtype).max)
    else:
        raise ValueError(f'Unsupported data type {im.dtype}')


def srgb_gamma2lin(im_in):
    """
    Convert srgb image in range (0-1) to linear intensity

    Input
    -----
    :param im_in: srgb image
    Output
    ------
    :return: linear image
    """
    thresh = 0.04045
    if torch.is_tensor(im_in):
        im_out = torch.where(im_in <= thresh, im_in / 12.92, 
                             ((im_in + 0.055) / 1.055) ** (12/5))
    else:
        im_out = np.where(im_in <= thresh, im_in / 12.92, 
                          ((im_in + 0.055) / 1.055) ** (12/5))

    return im_out


def phasemap_8bit(phasemap, inverted=True, rotate=True):
    """
    Convert a phasemap tensor into an 8bit phasemap to be saved out

    Input
    -----
    :param phasemap: input phasemap tensor, in the range of [-pi, pi].
    :param inverted: Flag indicating whether the phasemap is inverted.
    :param rotate: Flag indicating whether the phasemap is rotated.
    Output
    ------
    :return: 8bit phasemap
    """

    out_phase = phasemap.cpu().detach().squeeze().numpy()
    out_phase = ((out_phase + np.pi) % (2 * np.pi)) / (2 * np.pi)
    if inverted:
        phase_out_8bit = ((1 - out_phase) * 255).round().astype(np.uint8)
    else:
        phase_out_8bit = ((out_phase) * 255).round().astype(np.uint8)
    if rotate:
        phase_out_8bit = np.swapaxes(phase_out_8bit, -2, -1)
    return phase_out_8bit
    

def complex_interpolate(field, src_res, target_res):
    """ 
    Interpolates a complex field from the source resolution to the target
    resolution if there is a difference in the resolutions
    
    Input
    -----
    :param field: field to be interpolated
    :param src_res: source resolution
    :param target_res: target resolution
    Output
    ------
    :return: interpolated field
    """
    if (src_res[0] != target_res[0]) or (src_res[1] != target_res[1]):
        field_real = F.interpolate(field.real, size=target_res,
            mode='bilinear', antialias=True)
        field_imag = F.interpolate(field.imag, size=target_res,
            mode='bilinear', antialias=True)
        return torch.complex(field_real, field_imag)
    else:
        return field


class PMap(dict):
    # Editible PMap Class for parameters
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
