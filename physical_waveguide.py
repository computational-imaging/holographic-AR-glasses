"""
Implementation for physically-modelled phenomena in the waveguide

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
import torch
import numpy as np
import torch.nn as nn
import torch.fft as tfft
import torch.nn.functional as F

import utils

class WaveguideTransferFunction(nn.Module):
    """
    Module for modelling frequency-dependent transfer function
    through the waveguide
    
    Parameters
    -----
    :param feature_size: sampling resolution
    :param wavelength: wavelength
    :param pupil_index: indices of internally reflected copies of the
        entrance pupil that are modelled
    :param wvguide_thickness: thickness of the waveguide
    :param grating_period: grating period of the couplers
    :param refractive_index: refractive index of the waveguide
    :param out_coupler_res: out-coupled resolution after waveguide
    :param in_coupler_res: resolution of in-coupled wavefront
    :param fx_max: Maximum horizontal frequencies in the waveguide that will
        end up inside our ROI
    :param fy_max: Maximum vertical frequencies in the waveguide that will
        end up inside our ROI
    :param out_coupler_shift: distance between center of in-coupler and center
        of out-coupler
    :param grating_direction: direction of diffraction order
    """

    def __init__(self, feature_size, wavelength, pupil_index, 
                wvguide_thickness, grating_period, refractive_index,
                out_coupler_res, in_coupler_res,
                fx_max=None, fy_max=None,
                out_coupler_shift=60e-3,
                grating_direction=1):
        super(WaveguideTransferFunction, self).__init__()

        self.feature_size = feature_size
        self.wvl = wavelength
        self.wvl_wg = wavelength/refractive_index
        self.pupil_index = pupil_index
        self.wvguide_thickness = wvguide_thickness
        self.grating_period = grating_period
        self.grating_direction = grating_direction
        self.refractive_index = refractive_index
        self.fx_max = fx_max
        self.fy_max = fy_max
        self.out_coupler_shift = out_coupler_shift
        # Compute the out_coupler resolution
        self.out_coupler_res = out_coupler_res
        self.H_wg = torch.view_as_real(self.compute_H_wg(in_coupler_res, 
            self.out_coupler_res, self.pupil_index))

    def compute_H_wg(self, in_coupler_res, out_coupler_res, pupil_index):
        # frequency coordinates sampling
        fy = torch.linspace(-1 / (2 * self.feature_size[0]),
                             1 / (2 * self.feature_size[0]),
                             in_coupler_res[0]+out_coupler_res[0])
        fx = torch.linspace(-1 / (2 * self.feature_size[1]), 
                             1 / (2 * self.feature_size[1]), 
                             in_coupler_res[1]+out_coupler_res[1])
        delta_fy = fy[1] - fy[0]
        delta_fx = fx[1] - fx[0]
        FX, FY = torch.meshgrid(fx, fy)
        FX = torch.transpose(FX, 0, 1)
        FY = torch.transpose(FY, 0, 1)
        
        # Tilted frequency axis for diffracted propagation
        FX_prime = FX + self.grating_direction / self.grating_period
        k = 2 * math.pi / self.wvl_wg

        # Compute masks for angles that are not propagted in the waveguide
        not_tir_mask = self.wvl**2 * (FX_prime**2 + FY**2) < 1
        evanescant_mask = (1 - (self.wvl_wg * FX_prime)**2 - 
            (self.wvl_wg * FY)**2) < 0

        # Compute ramp for shifting the coordinate plane to the center of the
        # out coupler
        Shift_phase_F = 2 * math.pi * FX * self.out_coupler_shift

        H_sum = None # Initialize the waveguide transfer function

        # Sum over all internally reflected copies that could reach out-coupler
        for n_pupil in pupil_index:
            freq_kernel = (1 - (self.wvl_wg * FX_prime)**2 - 
                (self.wvl_wg * FY)**2).sqrt()

            # Compute reflection coefficients
            n_ref = self.refractive_index
            Reflect_phase_F = -2 * (2 * (n_pupil - 1) + 1) * torch.atan2((
                    self.wvl**2 * (FX_prime**2 + FY**2) - 1).sqrt(), 
                n_ref * freq_kernel)
            Reflect_phase_F[evanescant_mask] = 0.
            Reflect_phase_F[not_tir_mask] = 0.

            # Compute Propagation phase delay through the waveguide
            Wvguide_prop_phase_F = (2 * n_pupil * self.wvguide_thickness * 
                k * freq_kernel)
            Wvguide_prop_phase_F[evanescant_mask] = 0.

            # Compute kernel by combining the phase delays
            H = torch.exp(1j * (Reflect_phase_F + Wvguide_prop_phase_F + 
                Shift_phase_F))
            H[evanescant_mask] = 0.
            H[not_tir_mask] = 0.

            # Use analytical derivative for phase slope to determine where the
            # propagation kernel would be aliased for band limited propagation
            dfx_phase_F = (2 * math.pi * self.out_coupler_shift - 2 * n_pupil *
                 self.wvguide_thickness * k * (self.wvl_wg**2) *
                 FX_prime / freq_kernel)
            dfx_phase_F[evanescant_mask] = 0.
            dfy_phase_F = (-2 * n_pupil* self.wvguide_thickness * k * 
                (self.wvl_wg**2) * FY / freq_kernel)
            dfy_phase_F[evanescant_mask] = 0.
            # Zero parts of kernel that would be aliased (abs(df) > pi/delta_f)
            H *= ((torch.abs(dfy_phase_F) < math.pi/delta_fy) * 
                (torch.abs(dfx_phase_F) < math.pi/delta_fx))

            H_sum = H if H_sum is None else H_sum + H
            
        # Filter frequencies that will end up outside our ROI
        if self.fx_max is not None:
            H_sum *= (torch.abs(FX) <= self.fx_max).type(torch.FloatTensor)
        if self.fy_max is not None:
            H_sum *= (torch.abs(FY) <= self.fy_max).type(torch.FloatTensor)

        return H_sum

    def forward(self, u_in):
        H_wg = torch.view_as_complex(self.H_wg)

        ## Propagate field through waveguide
        # preprocess with padding for linear conv.
        input_resolution = u_in.size()[-2:]
        conv_size = [p+i for p, i in zip(self.out_coupler_res, input_resolution)]
        u_in = utils.pad_image(u_in, conv_size)

        # Apply waveguide transfer function in the frequency domain
        U1 = tfft.fftshift(tfft.fftn(u_in, dim=(-2, -1), norm='ortho'),
            (-2, -1))
        U2 = U1 * H_wg
        u_out = tfft.ifftn(tfft.ifftshift(U2, (-2, -1)), dim=(-2, -1),
            norm='ortho')

        # Crop to output resolution
        u_out = utils.crop_image(u_out, self.out_coupler_res)    

        return u_out


    def to(self, *args, **kwargs):
        # Move waveguide module to a device
        slf = super().to(*args, **kwargs)
        if slf.H_wg is not None:
            slf.H_wg = slf.H_wg.to(*args, **kwargs)
        try:
            slf.dev = next(slf.parameters()).device
        except StopIteration:  # no parameters
            device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
            if device_arg is not None:
                slf.dev = device_arg
        return slf


class ConvergingIllumination(nn.Module):
    """
    Module for modelling illumination focal power
    
    The focal power is applied before propagation in the waveguide.
    After the waveguide an imaging lens with the opposite focal power
    is used to image far virtual distances to a small imaging volume
    
    Parameters
    -----
    :param feature_size: sampling resolution
    :param wavelength: wavelength
    :param illumination_focal_length: focal length of the illumination
    :param imaging_res: resolution of the field in the imaging volume
    :param illum_res: resolution of the field of the converging illumination
    """

    def __init__(self, feature_size, wavelength,
                illumination_focal_length,
                imaging_res, illum_res):
        super(ConvergingIllumination, self).__init__()
        self.feature_size = feature_size
        self.wvl = wavelength
        self.illumination_focal_length = illumination_focal_length
        self.illum = torch.view_as_real(
            self.compute_lens(imaging_res, illum_res))

    def forward(self, u_in, imaging=False):
        # Crop wavefront to input size
        lens = utils.crop_image(torch.view_as_complex(self.illum), 
                                u_in.shape[-2:])
        if imaging:
            # undo focal power for imaging volume
            u_out = u_in/lens
        else:
            # apply converging illumination focal power
            u_out = u_in*lens
        return u_out

    def compute_lens(self, imaging_res, illum_res):
        # Determine lens sampling
        res = [max(im, il) for im, il in zip(imaging_res, illum_res)]
        y = torch.linspace(-self.feature_size[0]*(res[0]-1)/2, 
                            self.feature_size[0]*(res[0]-1)/2, res[0])
        x = torch.linspace(-self.feature_size[1]*(res[1]-1)/2, 
                            self.feature_size[1]*(res[1]-1)/2, res[1])
        X, Y = torch.meshgrid(x, y)
        X = torch.transpose(X, 0, 1)
        Y = torch.transpose(Y, 0, 1)

        # Compute lens phase function
        k = 2 * math.pi / self.wvl
        R = (X**2+Y**2+self.illumination_focal_length**2).sqrt()
        illum = torch.exp(-1j*k*R)

        return illum

    def to(self, *args, **kwargs):
        # Move illumination module to a device
        slf = super().to(*args, **kwargs)
        if slf.illum is not None:
            slf.illum = slf.illum.to(*args, **kwargs)
        try:
            slf.dev = next(slf.parameters()).device
        except StopIteration:  # no parameters
            device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
            if device_arg is not None:
                slf.dev = device_arg
        return slf
