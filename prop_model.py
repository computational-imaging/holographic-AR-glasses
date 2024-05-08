"""
Implementation for simulated wave propagation models
Conventional freespace model uses the ASM module from freespace.py. Our 
proposed waveguide models are constructed here using the analytically-derived
modules from physical_waveguide.py.

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
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

import utils
from network_modules import UnetGenerator, init_weights, Field2Input, \
    Output2Field
import freespace
from physical_waveguide import WaveguideTransferFunction, \
    ConvergingIllumination


def model(opt):
    """
    Initialize and load propagation model based on parameters in opt
    
    Input
    -----
    :param opt: arguments defining propagation model
    Output
    ------
    :return: propagation model
    """

    if opt.prop_model.lower() == 'freespace':
        # Initialize conventional freespace model
        sim_prop = freespace.Propagation(opt.prop_dists, opt.wavelength, 
                                 opt.feature_size, opt.slm_res,
                                 linear_conv=True, dim=1)
    else:        
        # Initialize proposed waveguide models
        sim_prop = PropWvguide(opt.prop_dists, 
                    opt.wavelength, 
                    opt.feature_size, 
                    slm_res=opt.slm_res,
                    out_coupler_res=opt.out_coupler_res,
                    in_coupler_res=opt.in_coupler_res,
                    illumination_focal_length=opt.illumination_focal_length,
                    pupil_index=opt.pupil_index,
                    wvguide_thickness=opt.wvguide_thickness,
                    grating_period=opt.grating_period,
                    refractive_index=opt.refractive_index,
                    out_coupler_shift=opt.out_coupler_shift,
                    grating_direction=opt.grating_direction,
                    learned_model='learned' in opt.prop_model
                )
        sim_prop.eval()

    if opt.prop_model_path is not None:
        # Load trained model parameters
        checkpoint = torch.load(os.path.join(opt.prop_model_path, f'{opt.chan_str}.ckpt'))
        sim_prop.load_state_dict(checkpoint["state_dict"])
        print(f'  - Model loaded from {opt.prop_model_path}')

    if opt.eval_plane_idx is not None:
        # Set evaluation parameters
        sim_prop.plane_idx = opt.eval_plane_idx

    return sim_prop


class PropWvguide(pl.LightningModule):
    """
    Implementation of proposed waveguide model
    
    Parameters
    -----
    :param prop_dists: propagation distance(s) from the SLM
    :param wavelength: wavelength
    :param feature_size: SLM pixel pitch
    :param slm_res: Resolution of SLM.
    :param out_coupler_res: Out-coupler resolution
    :param in_coupler_res: resolution of in-coupled wavefront
    :param illumination_focal_length: focal length of the illumination
    :param pupil_index: indices of internally reflected copies of the
        entrance pupil that are modelled
    :param wvguide_thickness: thickness of the waveguide
    :param grating_period: grating period of the couplers
    :param refractive_index: refractive index of the waveguide
    :param out_coupler_shift: distance between center of in-coupler and center
        of out-coupler
    :param grating_direction: direction of diffraction order
    :param learned_model: Flag to incorporate AI parameters for
        learned physical waveguide model
    """
    def __init__(self, prop_dists, wavelength, feature_size, 
                slm_res=(1080, 1920), out_coupler_res=(1000, 1000),
                in_coupler_res=(1000, 1000), illumination_focal_length=None,
                pupil_index=[1], wvguide_thickness=5e-3, grating_period=300e-9,
                refractive_index=1.8, out_coupler_shift=60e-3,
                grating_direction=1, learned_model=False):
        super(PropWvguide, self).__init__()

        ##################################
        # Compute Simulation Resolutions #
        ##################################
        self.slm_res = slm_res

        # Upsampling determined by resolution required to avoid aliasing based
        # on frequencies present at each point in the propagation
        illum_upsample_factor = (4, 4)
        wg_upsample_factor = (2, 2)
        imaging_upsample_factor = (3, 3)

        # Compute upsampled resolutions required to model illumination 
        illum_upsampled_slm_res = tuple([u * r for u, r in 
            zip(illum_upsample_factor, slm_res)])
        self.illum_upsampled_in_coupler_res = tuple([u * r for u, r in 
            zip(illum_upsample_factor, in_coupler_res)])
        self.illum_upsampled_out_coupler_res = tuple([u * r for u, r in 
            zip(illum_upsample_factor, out_coupler_res)])
        illum_upsampled_feature_size = tuple([f / u for f, u in 
            zip(feature_size, illum_upsample_factor)])

        # Compute upsampled resolutions required for waveguide propagation 
        self.wg_upsampled_in_coupler_res = tuple([u * r for u, r in 
            zip(wg_upsample_factor, in_coupler_res)])
        self.wg_upsampled_out_coupler_res = tuple([u * r for u, r in 
            zip(wg_upsample_factor, out_coupler_res)])
        wg_upsampled_feature_size = tuple([f / u for f, u in 
            zip(feature_size, wg_upsample_factor)])

        # Compute upsampled resolutions required for imaging volume
        self.imaging_upsampled_out_coupler_res = tuple([u * r for u, r in 
            zip(imaging_upsample_factor, out_coupler_res)])
        self.imaging_upsampled_slm_res = tuple([u * r for u, r in 
            zip(imaging_upsample_factor, slm_res)])
        imaging_upsampled_feature_size = tuple([f / u for f, u in 
            zip(feature_size, imaging_upsample_factor)])

        self.illum_upsample = nn.Upsample(size=illum_upsampled_slm_res,
            mode='nearest')

        #################################
        # Initialize Learned Operations #
        #################################
        self.learned_model = learned_model
        if learned_model:
            # Learned in-coupler efficiency
            self.ic_eff_amp = nn.Parameter(torch.ones(1, 1, 
                *self.wg_upsampled_in_coupler_res, requires_grad=True))
            self.ic_eff_phase = nn.Parameter(torch.zeros(1, 1, 
                *self.wg_upsampled_in_coupler_res, requires_grad=True))

            # Learned out-coupler efficiency
            self.oc_eff = nn.Parameter(torch.view_as_real(torch.ones(1, 1, 
                        *self.wg_upsampled_out_coupler_res, dtype=torch.complex64, 
                        requires_grad=True)))

            # In-Coupler network
            ic_cnn_res = tuple(res if res % 32 == 0 else
                                res + (32 - res % 32)
                                for res in self.wg_upsampled_in_coupler_res)
            ic_input = Field2Input(ic_cnn_res, coord='both',
                                    latent_amp=self.ic_eff_amp, 
                                    latent_phase=self.ic_eff_phase)
            ic_cnn = UnetGenerator(input_nc=4, output_nc=2,
                                    outer_skip=True)
            init_weights(ic_cnn, init_type='normal')
            ic_output = Output2Field(self.wg_upsampled_in_coupler_res, 'both')
            self.ic_cnn = nn.Sequential(ic_input, ic_cnn, ic_output)

            # Target network
            target_cnn_res = tuple(res if res % 32 == 0 else
                                   res + (32 - res % 32) for res in
                                   self.imaging_upsampled_out_coupler_res)
            target_input = Field2Input(target_cnn_res, coord='both_1ch_output',
                                        shared_cnn=True)
            target_cnn = UnetGenerator(input_nc=4, output_nc=1,
                                       outer_skip=True)
            init_weights(target_cnn, init_type='normal')
            target_output = Output2Field(self.imaging_upsampled_out_coupler_res, 
                                        'both_1ch_output',
                                        num_ch_output=len(prop_dists))
            self.target_cnn = nn.Sequential(target_input, target_cnn,
                                            target_output)

        ##################################
        # Initialize Physical Operations #
        ##################################
        # Explicitly model illumination focal power
        self.converging_illumination = ConvergingIllumination(
                                illum_upsampled_feature_size, wavelength,
                                illumination_focal_length,
                                self.illum_upsampled_out_coupler_res,
                                self.illum_upsampled_in_coupler_res)

        # Explicitly model waveguide transfer function
        # Truncate frequencies that do not propagate within our FOV
        fy_eyepiece_max = ((out_coupler_res[0]-50)*feature_size[0]/
            (wavelength * np.sqrt(4*illumination_focal_length**2+
            ((out_coupler_res[0]-50)*feature_size[0])**2)))
        fx_eyepiece_max = ((out_coupler_res[1]-50)*feature_size[1]/
            (wavelength * np.sqrt(4*illumination_focal_length**2+
            ((out_coupler_res[1]-50)*feature_size[1])**2)))
        self.wg_tf = WaveguideTransferFunction(wg_upsampled_feature_size, 
                                wavelength, pupil_index, wvguide_thickness,
                                grating_period, refractive_index,
                                self.wg_upsampled_out_coupler_res,
                                self.wg_upsampled_in_coupler_res,
                                fx_max=fx_eyepiece_max, fy_max=fy_eyepiece_max,
                                out_coupler_shift=out_coupler_shift,
                                grating_direction=grating_direction)

        # Use freespace propagation to simulate views seen at different 
        # distances through the waveguide
        self.prop_imaging = freespace.Propagation(prop_dists, wavelength, 
                                        imaging_upsampled_feature_size,
                                        self.imaging_upsampled_out_coupler_res,
                                        linear_conv=False)

    def forward(self, field, plane_idx=None):
        # If the input is a phase convert it to a field before propagation
        if field.dtype == torch.float32:
            field = torch.exp(1j * field)

        # Upsample the wavefront at the SLM to model converging illumination
        slm_field = self.illum_upsample(field.abs()) * torch.exp(1j * 
                                        self.illum_upsample(field.angle()))

        # Apply truncation of in-coupler and illumination phase delay
        ic_field = utils.crop_image(slm_field, 
                                    self.illum_upsampled_in_coupler_res)
        ic_field = self.converging_illumination(ic_field)
        
        # Resample the in-coupler field, optionally apply in-coupler CNN,
        # and propagate through the waveguide
        ic_field = utils.complex_interpolate(ic_field,
                                        self.illum_upsampled_in_coupler_res, 
                                        self.wg_upsampled_in_coupler_res)
        if self.learned_model:
            ic_field = self.ic_cnn(ic_field)
        oc_field = self.wg_tf(ic_field)

        # Optionally apply learned out-coupler efficiency, resample the field,
        # and image the field with the inverse of the illumination focal power
        if self.learned_model:
            oc_field = oc_field * torch.view_as_complex(self.oc_eff)  
        oc_field = utils.complex_interpolate(oc_field,
            self.wg_upsampled_out_coupler_res, self.illum_upsampled_out_coupler_res)
        imaging_field = self.converging_illumination(oc_field, imaging=True)

        # Resample the field and propagate to target planes seen through the 
        # waveguide.
        imaging_field = utils.complex_interpolate(imaging_field,
            self.illum_upsampled_out_coupler_res, self.imaging_upsampled_out_coupler_res)
        target_field = self.prop_imaging(imaging_field, plane_idx)

        # Optionally apply target CNN
        if self.learned_model:
            # Temporarily set target CNN to output a single plane if a single
            # plane is requested
            if plane_idx is not None:
                num_ch_output = self.target_cnn[-1].num_ch_output
                self.target_cnn[-1].num_ch_output = 1
            target_field = self.target_cnn(target_field).abs()*torch.exp(1j * 
                                                        target_field.angle())
            if plane_idx is not None:
                self.target_cnn[-1].num_ch_output = num_ch_output

        # Resize and downsample target field to the slm resolution
        target_field = utils.pad_image(target_field,
            self.imaging_upsampled_slm_res, pytorch=True)
        target_field = utils.crop_image(target_field,
            self.imaging_upsampled_slm_res)
        intensity = nn.functional.interpolate(target_field.abs()**2, 
                            size=self.slm_res, mode='bilinear', antialias=True)                    
        amp = torch.clamp(intensity, min=1e-8).mean(dim=0,
                            keepdim=True).sqrt() # 1e-8 to avoid NaN gradients
        
        return amp

    @property
    def plane_idx(self):
        # Get idx of single plane propagator
        return self._plane_idx

    @plane_idx.setter
    def plane_idx(self, idx):
        # Collapse multiplane propagator to single plane propagator
        if idx is None:
            return
        self._plane_idx = idx
        if len(self.prop_imaging) > 1:
            self.prop_imaging.plane_idx = idx
        if self.learned_model and self.target_cnn[-1].num_ch_output>1:
            self.target_cnn[-1].num_ch_output = 1

    def to(self, *args, **kwargs):
        # Move waveguide model to a device
        slf = super().to(*args, **kwargs)
        if slf.prop_imaging is not None:
            slf.prop_imaging.to(*args, **kwargs)
        if slf.converging_illumination is not None:
            slf.converging_illumination.to(*args, **kwargs)
        if slf.wg_tf is not None:
            slf.wg_tf.to(*args, **kwargs)
        try:
            slf.dev = next(slf.parameters()).device
        except StopIteration:  # no parameters
            device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
            if device_arg is not None:
                slf.dev = device_arg
        return slf