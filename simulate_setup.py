"""
Script to run phases through simulated waveguide setup

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
import torch
import math
import imageio
import configargparse
import numpy as np

import params
import prop_model    
import utils

def main():
    # Process command line arguments and initialize simulation parameters
    torch.set_default_dtype(torch.float32)
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False,
          is_config_file=True, help='Path to config file.')
    params.add_parameters(p)
    opt = p.parse_args()
    opt.prop_model = 'learnedphysical'
    opt = params.set_configs(opt)
    dev = torch.device('cuda')
    run_id = params.run_id(opt)

    # Read in phase
    if opt.phase_path is None:
        raise ValueError(f'Phase path must be specified')
    else:
        phase_im_enc = imageio.imread(opt.phase_path)
        im = (1 - phase_im_enc / np.iinfo(np.uint8).max) * 2 * np.pi - np.pi
        im = np.swapaxes(im, -2, -1)
        phase = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Initialize simulated propagation model to optimize through
    sim_prop = prop_model.model(opt)
    sim_prop.eval()
    sim_prop.to(dev)

    # Propagate and save out amplitude at each target plane
    for plane_idx in range(opt.num_planes):
        recon_field = sim_prop(phase.to(dev), plane_idx=plane_idx)
        recon_amp = utils.crop_image(recon_field, opt.roi_res).abs()
        recon_amp = torch.clamp(recon_amp.squeeze().cpu().detach(), 0, 1)
        recon_amp = (255*recon_amp).numpy().astype(np.uint8)
        recon_path = opt.phase_path.replace('.png', f'_amp{plane_idx}.png')
        imageio.imwrite(recon_path, recon_amp)

if __name__ == "__main__":
    main()