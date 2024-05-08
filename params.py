"""
Define argument parser and default parameter settings for physical setup

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
import numpy as np
import torch.nn as nn

import utils

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9


def str2bool(v):
    """ Simple query parser for configArgParse
    (which doesn't support native bool from cmd)
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def add_parameters(p):
    """
    Define command line arguments for scripts
    
    Input
    -----
    :param p: configargparse.ArgumentParser to receive arguments
    Output
    ------
    :return: updated parser
    """
    p.add_argument('--channel', type=int, default=1, 
                    help='Red:0, green:1, blue:2')
    p.add_argument('--prop_model', type=str, default='freespace', 
                    help='Type of propagation model, \
                    (freespace/physical/learnedphysical)')
    p.add_argument('--out_path', type=str, default='./results',
                   help='Directory for output')
    p.add_argument('--data_path', type=str, default=None,
                   help='Directory for input')
    p.add_argument('--phase_path', type=str, default=None,
                   help='Path to phase for simulation')
    p.add_argument('--prop_model_path', type=str, default=None,
                    help='Path to saved learned model')
    p.add_argument('--exp', type=str, default='', help='Name of experiment')
    p.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    p.add_argument('--num_iters', type=int, default=1500, 
                    help='Number of iterations to run SGD algorithm')
    p.add_argument('--mem_eff', type=str2bool, default=False,
                   help='If true, run a memory efficient version of sgd for \
                   3d hologram synthesis')
    p.add_argument('--target', type=str, default='rgb',
                   help='Type of target:'
                        '{2d, rgb} or '
                        '{3d, fs, focal-stack, focal_stack}')
    p.add_argument('--eval_plane_idx', type=int, default=None,
                help='specify plane for 2D optimization')

    return p


def set_configs(opt_p):
    """
    initialize hardware setup parameters
    
    Input
    -----
    :param opt_p: configargparse.ArgumentParser to store arguments
    Output
    ------
    :return: PMap opt with full setup parameters
    """
    if not isinstance(opt_p, utils.PMap):
        opt = utils.PMap()
        for k, v in vars(opt_p).items():
            opt[k] = v
    else:
        opt = opt_p

    # initialize parameters from the hardware setup
    laser_config(opt)
    slm_config(opt)
    waveguide_config(opt)

    return opt


def run_id(opt):
    """
    create identification string for experiment
    
    Input
    -----
    :param opt: configargparse.ArgumentParser with arguments
    Output
    ------
    :return: identification string
    """
    opt.chan_str = ('red', 'green', 'blue')[opt.channel]
    id_str = f'{opt.exp}_{opt.chan_str}_{opt.prop_model}'
    if opt.eval_plane_idx is not None:
        id_str = f'p{opt.eval_plane_idx}' + id_str
    return id_str


def slm_config(opt):
    """
    Add SLM specific hardware parameters to opt
    
    Input
    -----
    :param opt: configargparse.ArgumentParser with arguments
    """
    # setting for specific SLM.
    opt.feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
    opt.slm_res = (1920, 1080)  # resolution of SLM
    opt.image_res = opt.slm_res
    

def laser_config(opt):
    """
    Add laser specific hardware parameters to opt
    
    Input
    -----
    :param opt: configargparse.ArgumentParser with arguments
    """
    opt.wavelengths = (638.35 * nm, 521.16 * nm, 443.50 * nm) 
    opt.wavelength = opt.wavelengths[opt.channel]


def waveguide_config(opt):   
    """
    Add waveguide specific hardware parameters to opt
    
    Input
    -----
    :param opt: configargparse.ArgumentParser with arguments
    """     
    # Simulate effect of light that internally reflects up to 5 times back
    # and forth in the waveguide
    opt.pupil_index = list(range(5))

    # Fabricated waveguide parameters
    opt.grating_direction = -1
    opt.refractive_index = [1.79827653, 1.81822298, 1.84356052][opt.channel]
    opt.grating_period = 384*nm
    opt.illumination_focal_length = [85.6*mm, 85.6*mm, 85.6*mm][opt.channel]
    opt.wvguide_thickness = 5.06*mm
    opt.in_coupler_size = (6.5*mm, 6.5*mm)
    opt.out_coupler_size = (6.5*mm, 6.918*mm) 
    opt.out_coupler_shift = opt.grating_direction*(23.943*mm-
        opt.in_coupler_size[1]/2+opt.out_coupler_size[1]/2) 

    # Use focal length to compute distances from the SLM
    # that correspond to specific virtual distances from viewer
    diopters = [0.0, 1.0/3, 2.0/3, 1.0]
    prop_dists_rgb = 3*[[-(1/(d+(1/opt.illumination_focal_length))) 
                            for d in diopters]]
    opt.prop_dists = prop_dists_rgb[opt.channel]
    opt.num_planes = len(diopters)
    
    # Define simulation resolutions based on coupler dimensions
    opt.in_coupler_res = [opt.in_coupler_size[0]/opt.feature_size[0], 
        opt.in_coupler_size[1]/opt.feature_size[1]]
    opt.out_coupler_res = [opt.out_coupler_size[0]/opt.feature_size[0], 
        opt.out_coupler_size[1]/opt.feature_size[1]]
    opt.roi_res = [800, 800]
    if 'learned' in opt.prop_model.lower():
        # Model slightly larger regions for both couplers to model
        # any error in out-coupler shift
        ic_pad_factor = 1.15
        oc_pad_factors = [1.15, 1.35]
        if opt.channel == 0:
            # Add more leeway for red which experiences the most horizontal
            # shift in the waveguide
            oc_pad_factors[1] = 1.65

        opt.out_coupler_res = [o*r for o, r in zip(oc_pad_factors,
            opt.out_coupler_res)]
        opt.in_coupler_res = [ic_pad_factor*r for r in opt.in_coupler_res]
    else:
        # Empirically padding the out-coupler region for the purely physical 
        # model provides limited robustness to error in out-coupler shift
        opt.out_coupler_res = (opt.out_coupler_res[0]+300, 
            opt.out_coupler_res[1]+300)
    # Round up to even value
    opt.out_coupler_res = [2*math.ceil(i/2) for i in opt.out_coupler_res]
    opt.in_coupler_res = [2*math.ceil(i/2) for i in opt.in_coupler_res]


