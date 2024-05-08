"""
Script to run hologram synthesis algorithms

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

import utils
import params
import CGH
import data_loader
import prop_model    

def main():
    # Process command line arguments and initialize simulation parameters
    torch.set_default_dtype(torch.float32)
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False,
          is_config_file=True, help='Path to config file.')
    params.add_parameters(p)
    opt = params.set_configs(p.parse_args())
    dev = torch.device('cuda')

    # Create folder for saving out optimized phases
    run_id = params.run_id(opt)
    out_path = os.path.join(opt.out_path, run_id)
    utils.cond_mkdir(out_path)
    print(f'  - Outputs will be written to {out_path}')
        
    # Initialize simulated propagation model to optimize through
    sim_prop = prop_model.model(opt)
    sim_prop.eval()
    sim_prop.to(dev)

    # Initialize dataloader
    img_loader = data_loader.TargetLoader(**opt)

    # Loop through targets and save optimized phases in output folder
    for i, target in enumerate(img_loader):
        target_amp, target_idx = target
        target_amp = target_amp.to(dev).detach()
        if len(target_amp.shape) < 4:
            target_amp = target_amp.unsqueeze(0)
        print(f'  - Running phase optimization for {target_idx}th image ...')

        # Initialize random phase
        init_phase = (2 * math.pi * (-0.5 + 1.0 * torch.rand(1, 1, 
                      *opt.slm_res))).to(dev)

        # Run phase optimization
        with torch.enable_grad():
            final_phase = CGH.gradient_descent(init_phase, target_amp,
                                forward_prop=sim_prop, 
                                **opt)

        # Save out final phase
        phase_out = utils.phasemap_8bit(final_phase)
        phase_out_path = os.path.join(out_path, f'{target_idx}.png')
        imageio.imwrite(phase_out_path, phase_out)

if __name__ == "__main__":
    main()