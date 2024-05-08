"""
ASM implementation for freespace propagation

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

import torch
import torch.nn as nn
import utils
import torch.fft as tfft
import math

class Propagation(nn.Module):
    """
    Propagation module for multiplane ASM propagation
    
    Parameters
    -----
    :param prop_dist: propagation distance(s)
    :param wavelength: wavelength
    :param feature_size: sampling resolution
    :param res: resolution of field to be propagated
    :param dim: The dimension to stack propagated fields
    :param linear_conv: If true, pad input with zeros for linear convolution
    """
    def __init__(self, prop_dist, wavelength, feature_size, 
                 res, dim=1, linear_conv=True):
        super(Propagation, self).__init__()

        if not isinstance(prop_dist, list):
            prop_dist = [prop_dist]
        self.prop_dist = prop_dist
        self.feature_size = feature_size
        self.wvl = wavelength
        self.linear_conv = linear_conv
        self.bl_asm = min(prop_dist) > 0.3
        self.dim = dim  
        Hs = []
        for prop_dist in self.prop_dist:
            h = torch.view_as_real(self.compute_H(torch.ones(1, 1, *res, 
                                        dtype=torch.complex64), 
                                    prop_dist, self.wvl, self.feature_size,
                                    self.linear_conv,
                                    bl_asm=self.bl_asm))
            Hs.append(h)
        self.H = torch.cat(Hs, dim=1)

    def compute_H(self, input_field, prop_dist, wvl, feature_size, 
                  lin_conv=True, bl_asm=False):

        # Determine Frequency domain sampling
        r_mul = 2 if lin_conv else 1
        num_y, num_x = r_mul*input_field.shape[-2], r_mul*input_field.shape[-1]
        dy, dx = feature_size  
        fy = torch.linspace(-1 / (2 * dy), 1 / (2 * dy), num_y)
        fx = torch.linspace(-1 / (2 * dx), 1 / (2 * dx), num_x)
        FX, FY = torch.meshgrid(fx, fy)
        FX = torch.transpose(FX, 0, 1)
        FY = torch.transpose(FY, 0, 1)

        # Compute ASM propagation kernel in frequency domain
        G = 2 * math.pi * (1 / wvl**2 - (FX ** 2 + FY ** 2)).sqrt()
        H_exp = G.reshape((1, 1, *G.shape))

        if bl_asm:
            # Zero-out unwanted frequencies for potential aliasing
            fy_max = 1/math.sqrt((2*prop_dist*(1/(dy*float(num_y))))**2+1)/wvl
            fx_max = 1/math.sqrt((2*prop_dist*(1/(dx*float(num_x))))**2+1)/wvl
            H_filter = ((torch.abs(FX) < fx_max) & (torch.abs(FY) < fy_max)
                        ).type(torch.FloatTensor)
        else:
            H_filter = 1

        # Create complex kernel
        H =  torch.exp(1j * H_exp * prop_dist) * H_filter

        return H

    def forward(self, u_in, plane_idx=None):
        # If the input is a phase convert it to a field before propagation
        if u_in.dtype == torch.float32:
            u_in = torch.exp(1j * u_in)

        if plane_idx is not None: 
            # Propagate to a single plane without collapsing propagator
            H = torch.view_as_complex(self.H[:, plane_idx:plane_idx+1, ...])
        else:
            H = torch.view_as_complex(self.H)

        if self.linear_conv:
            # preprocess with padding for linear conv.
            input_resolution = u_in.size()[-2:]
            conv_size = [i * 2 for i in input_resolution]
            u_in = utils.pad_image(u_in, conv_size)

        # Perform propagation with multiplication in frequency-domain
        U1 = tfft.fftshift(tfft.fftn(u_in, dim=(-2, -1), norm='ortho'), 
                            (-2, -1))
        U2 = U1 * H
        u_out = tfft.ifftn(tfft.ifftshift(U2, (-2, -1)), dim=(-2, -1),
                            norm='ortho')

        if self.linear_conv:
            u_out = utils.crop_image(u_out, input_resolution)
        
        return u_out

    def __len__(self):
        # Return number of planes that are propagated to
        return len(self.prop_dist)

    @property
    def plane_idx(self):
        # Get idx of single plane propagator
        return self._plane_idx

    @plane_idx.setter
    def plane_idx(self, idx):
        if idx is None:
            return
        # Collapse multiplane propagator to single plane propagator
        self._plane_idx = idx
        if len(self.prop_dist) > 1:
            self.prop_dist = [self.prop_dist[idx]]
        if self.H is not None:
            self.H = self.H[:, idx:idx+1, ...]

    def to(self, *args, **kwargs):
        # Move Propagation module to a device
        slf = super().to(*args, **kwargs)
        if slf.H is not None:
            slf.H = slf.H.to(*args, **kwargs)
        try:
            slf.dev = next(slf.parameters()).device
        except StopIteration:  # no parameters
            device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
            if device_arg is not None:
                slf.dev = device_arg
        return slf


