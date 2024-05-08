"""
Data loader for target scenes

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

import os
import cv2
from skimage.transform import resize
from imageio import imread
import numpy as np
import torch
from torchvision.transforms.functional import resize as resize_tensor

import utils


# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'


def depth_read(filename):
    """
    Read depth data from file
    
    Input
    -----
    :param filename: path to depth .dpt file
    Output
    ------
    :return: depth as a numpy array
    """
    f = open(filename, 'rb')
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file \
        (should be: {0}, is: {1}). Big-endian machine? '.format(
        TAG_FLOAT, check)
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width * height
    assert width > 0 and height > 0 and size > 1 and size < 100000000,  \
        ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(
        width, height)
    depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def get_image_filenames(dir, focuses=None):
    """
    Returns all filenames in the input directory that are images
    
    Input
    -----
    :param dir: path to directory(s)
    Output
    ------
    :return: array of image filenames
    """
    image_types = ('jpg', 'jpeg', 'JPEG', 'tiff', 'tif', 'png', 'bmp', 'gif',
                    'exr', 'dpt', 'hdf5')
    if isinstance(dir, str):
        files = os.listdir(dir)
        exts = (os.path.splitext(f)[1] for f in files)
        if focuses is not None:
            images = [os.path.join(dir, f)
                      for e, f in zip(exts, files)
                      if e[1:] in image_types and 
                      int(os.path.splitext(f)[0].split('_')[-1]) in focuses]
        else:
            images = [os.path.join(dir, f)
                      for e, f in zip(exts, files)
                      if e[1:] in image_types]
        return images
    elif isinstance(dir, list):
        # Suppport multiple directories (randomly shuffle all)
        images = []
        for folder in dir:
            files = os.listdir(folder)
            exts = (os.path.splitext(f)[1] for f in files)
            images_in_folder = [os.path.join(folder, f)
                                for e, f in zip(exts, files)
                                if e[1:] in image_types]
            images = [*images, *images_in_folder]

        return images


def resize_keep_aspect(image, target_res, pad=False, pytorch=False):
    """
    Resizes image to the target_res while keeping aspect ratio by cropping
    
    Input
    -----
    :param image: image to be resized
    :param target_res: desired resolution of resized image
    :param pad: if True, will pad zeros instead of cropping to preserve ratio
    :param pytorch: flag for pad_image
    Output
    ------
    :return: resized image
    """

    im_res = image.shape[-2:]

    if im_res[0] == target_res[0] and im_res[1] == target_res[1]:
        return image

    # finds the resolution needed for either dimension to have the target
    # ratio, when the other is kept constant. If the image doesn't have the
    # target ratio, then one of these two will be larger and the other smaller,
    # than the current image dimensions
    resized_res = (int(np.ceil(im_res[1] * target_res[0] / target_res[1])),
                   int(np.ceil(im_res[0] * target_res[1] / target_res[0])))

    # only pads smaller or crops larger dims, meaning that the resulting image
    # size will be the target aspect ratio after a single pad/crop to the
    # resized_res dimensions
    if pad:
        image = utils.pad_image(image, resized_res, pytorch=False)
    else:
        image = utils.crop_image(image, resized_res)

    # Resize image, now with the correct ratio, to target resolution
    if pytorch:
        image = resize_tensor(image, target_res)
        return image
    else:
        # switch to numpy channel dim convention, resize, switch back
        image = np.transpose(image, axes=(1, 2, 0))
        image = resize(image, target_res, mode='reflect')
        return np.transpose(image, axes=(2, 0, 1))


class TargetLoader(torch.utils.data.IterableDataset):
    """
    Data loader for target scenes
    
    Parameters
    -----
    :param data_path: path to folder with target scenes
    :param target: type of target scenes (2d/3d)
    :param channel: color channel to extract
    :param image_res: resolution outputted by data loader
    :param roi_res: resolution to scale image to for supervision
    :param num_planes: number of planes for 3D targets
    """

    def __init__(self, data_path=None, target='2d', channel=None,
                 image_res=(800, 1280), roi_res=(700, 1190), 
                 num_planes=4, **kwargs):
        self.data_path = data_path
        self.target_type = target.lower()
        self.channel = channel
        self.roi_res = roi_res
        self.image_res = image_res
        self.num_planes = num_planes

        self.im_names = get_image_filenames(self.data_path)
        if self.target_type in ('3d', 'fs', 'focal-stack', 'focal_stack'):
            # extract list of all the target images for the first depth plane
            self.im_names = [f for f in self.im_names if '_0.' in f]
        self.im_names.sort()

        # create list of image IDs
        self.order = list(range(len(self.im_names)))

    def __iter__(self):
        self.ind = 0
        return self

    def __len__(self):
        return len(self.order)

    def __next__(self):
        if self.ind < len(self.order):
            img_idx = self.order[self.ind]

            self.ind += 1
            if self.target_type in ('2d', 'rgb'):
                return self.load_image(self.im_names[img_idx])
            if self.target_type in ('3d', 'fs', 'focal-stack', 'focal_stack'):
                return self.load_focal_stack(self.im_names[img_idx])
        else:
            raise StopIteration

    def load_image(self, file):
        if 'exr' in file:
            im = cv2.imread(file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        else:
            im = imread(file)

        if len(im.shape) < 3:
            # augment channels for gray images
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)  

        # select channel while keeping dims
        im = im[..., self.channel, np.newaxis]

        im = utils.im2float(im, dtype=np.float64)  # convert to double, max 1

        # linearize intensity and convert to amplitude
        if not 'exr' in file:
            im = utils.srgb_gamma2lin(im)
        im = np.sqrt(im)  # to amplitude

        # move channel dim to torch convention
        im = np.transpose(im, axes=(2, 0, 1))

        # normalize resolution
        im = resize_keep_aspect(im, self.roi_res)
        im = utils.crop_image(utils.pad_image(im, self.image_res, 
                                pytorch=False), self.image_res)

        path = os.path.splitext(file)[0]
        return (torch.from_numpy(im).float(),
                os.path.split(path)[1])

    def load_focal_stack(self, file):
        # Load first plane target image
        img, idx = self.load_image(file)

        # Concatenate target images for other planes
        for d in range(self.num_planes-1):
            img_d, _ = self.load_image(file.replace('_0.', f'_{d+1}.'))
            img = torch.cat((img, img_d), dim=0)

        return (img, idx[:-2])
