"""
Gradient descent CGH algorithm implemented for 2D/3D supervision.

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
import torch.optim as optim

import utils


def compute_scaled_loss(recon_field, target_amp, roi_res, loss_fn):
    """
    Scale reconstructed field brightness by global scale factor
    before computing loss
    
    Input
    -----
    :param recon_field: reconstructed field
    :param target_amp: target scene
    :param roi_res: resolution of region of interest to optimize
    :param loss_fn: loss function to optimize
    Output
    ------
    :return: loss computed on scaled reconstruction

    """

    recon_amp = utils.crop_image(recon_field, roi_res).abs()

    # Compute scale that minimizes MSE btw recon and target
    with torch.no_grad():
        s = (recon_amp * target_amp).mean() / \
            (recon_amp ** 2).mean() 

    # Compute loss on scaled reconstruction
    return loss_fn(s * recon_amp, target_amp)


def gradient_descent(init_phase, target_amp, forward_prop=None, num_iters=1000,
                     roi_res=None, lr=0.01, mem_eff=False, *args, **kwargs):
    """
    Gradient-descent based method for phase optimization.
    
    Input
    -----
    :param init_phase: initial phase for gradient descent iterations
    :param target_amp: target scene
    :param forward_prop: simulated propagation model
    :param num_iters: number of optimization iterations
    :param roi_res: resolution of region of interest to optimize
    :param lr: learning rate for optimization
    :param mem_eff: Option for 3D scenes to trade lower peak memory usage for
                    higher computational cost per iteration
    Output
    ------
    :return: phase pattern optimized to produce the desired scene

    """

    # Initialize optimization variables and optimizer
    slm_phase = init_phase.requires_grad_(True)  
    optvars = [{'params': slm_phase}]
    optimizer = optim.Adam(optvars, lr=lr)
    if roi_res is not None:
        target_amp = utils.crop_image(target_amp, roi_res)
    loss_fn = nn.functional.mse_loss

    # Iteratively update phase to improve simulated reconstruction quality
    for t in range(num_iters):
        print(f'Iter {t}/{num_iters}')
        optimizer.zero_grad()

        # Simulate output of AR system and compute loss against desired output
        if mem_eff: 
            # For 3D, optionally compute the gradient contribution for each
            # depth independently to reduce peak memory usage
            for depth_idx in range(target_amp.shape[1]):
                recon_field = forward_prop(slm_phase, plane_idx=depth_idx)
                loss_val = compute_scaled_loss(recon_field, 
                    target_amp[:,depth_idx:depth_idx+1,...], roi_res, loss_fn)
                loss_val.backward(retain_graph=False)
        else:
            recon_field = forward_prop(slm_phase)
            loss_val = compute_scaled_loss(recon_field, target_amp, 
                    roi_res, loss_fn)
            loss_val.backward()

        # Iteratively update phase based on the loss
        optimizer.step()

    return slm_phase.clone().cpu().detach()


