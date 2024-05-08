"""
Implementation of network modules

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

import functools
import torch
import torch.nn as nn
from torch.nn import init

import utils


class Field2Input(nn.Module):
    """
    Module for converting complex-valued fields to multi-channel input for CNNs
    Optional field modulation can be applied first with latent_amp/latent_phase
    
    Parameters
    -----
    :param input_res: resolution of field
    :param coord: coordinate system to represent the field with
    :param latent_amp: optional amplitude modulation
    :param latent_phase: optional phase modulation
    :param shared_cnn: Flag to stack multiple fields n the batch dimension to
        be processed by a single CNN
    """
    def __init__(self, input_res=(800, 1280), coord='rect', latent_amp=None, 
                 latent_phase=None, shared_cnn=False):
        super(Field2Input, self).__init__()
        self.input_res = input_res
        self.coord = coord.lower()
        self.latent_amp = latent_amp
        self.latent_phase = latent_phase
        self.shared_cnn = shared_cnn

    def forward(self, input_field):
        # If input is an slm phase, convert it to a field
        if input_field.dtype == torch.float32:
            input_field = torch.exp(1j * input_field)

        # 1) Apply optional phase modulation
        if self.latent_phase is not None:
            input_field = input_field * torch.exp(1j * self.latent_phase)

        # 2) Apply optional amplitude modulation
        if self.latent_amp is not None:
            input_field = self.latent_amp * input_field

        # Pad and crop field to the desired resolution
        input_field = utils.pad_image(input_field, self.input_res, 
                                      pytorch=True)
        input_field = utils.crop_image(input_field, self.input_res)

        # To use shared CNN, put everything into batch dimension;
        if self.shared_cnn:
            num_mb, num_dists = input_field.shape[0], input_field.shape[1]
            input_field = input_field.reshape(num_mb*num_dists, 1, 
                                              *input_field.shape[2:])

        # Convert complex field to channels in the desired coordinate system
        if self.coord == 'rect':
            stacked_input = torch.cat((input_field.real, input_field.imag), 1)
        elif self.coord == 'polar':
            stacked_input = torch.cat((input_field.abs(), 
                                       input_field.angle()), 1)
        elif self.coord == 'amp':
            stacked_input = input_field.abs()
        elif 'both' in self.coord:
            stacked_input = torch.cat((input_field.abs(), input_field.angle(), 
                                       input_field.real, input_field.imag), 1)

        return stacked_input


class Output2Field(nn.Module):
    """
    Module for converting multi-channel output of CNNs to complex-valued fields
    Optional field modulation can be applied first with latent_amp/latent_phase
    
    Parameters
    -----
    :param output_res: resolution of field
    :param coord: coordinate system of channels
    :param num_ch_output: number of field channels
    """
    def __init__(self, output_res=(800, 1280), coord='rect', num_ch_output=1):
        super(Output2Field, self).__init__()
        self.output_res = output_res
        self.coord = coord.lower()
        self.num_ch_output = num_ch_output

    def forward(self, stacked_output):
        # Convert channels to field
        if self.coord in ('rect', 'both'):
            real = stacked_output[:,:stacked_output.shape[1]//2,...]
            imag = stacked_output[:,stacked_output.shape[1]//2:,...]
            complex_valued_field = torch.complex(real, imag).contiguous()
        elif self.coord == 'polar':
            amp = stacked_output[:,:stacked_output.shape[1]//2,...]
            phi = stacked_output[:,stacked_output.shape[1]//2:,...]
            complex_valued_field = amp * torch.exp(1j * phi)
        elif self.coord == 'amp' or '1ch_output' in self.coord:
            complex_valued_field = stacked_output * \
                            torch.exp(1j * torch.zeros_like(stacked_output))

        # Pad and crop field to the desired resolution
        output_field = utils.pad_image(complex_valued_field, self.output_res, 
                                       pytorch=True)
        output_field = utils.crop_image(output_field, self.output_res)

        # reshape tensor to desired channel count
        output_field = output_field.reshape(
            output_field.shape[0]*output_field.shape[1] // self.num_ch_output,
            self.num_ch_output, *output_field.shape[2:])

        return output_field


class UnetSkipConnectionBlock(nn.Module):
    """
    Defines the Unet submodule with skip connections.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|

    Parameters
    -----
    :param outer_nc: the number of filters in the outer conv layer
    :param inner_nc: the number of filters in the inner conv layer
    :param input_nc: the number of channels in input images/features
    :param submodule: inner submodules
    :param outermost: flag if this module is the outermost module
    :param innermost: flag if this module is the innermost module
    :param norm_layer: normalization layer
    :param use_dropout: flag to use dropout layers
    :param outer_skip: add an additional outer skip connection
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False,
                 outer_skip=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.outer_skip = outer_skip
        if norm_layer == None:
            use_bias = True
        elif type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=5,
                             stride=2, padding=2, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        if norm_layer is not None:
            if norm_layer == nn.GroupNorm:
                downnorm = norm_layer(8, inner_nc)
            else:
                downnorm = norm_layer(inner_nc)
        else:
            downnorm = None
        uprelu = nn.ReLU(True)
        if norm_layer is not None:
            if norm_layer == nn.GroupNorm:
                upnorm = norm_layer(8, outer_nc)
            else:
                upnorm = norm_layer(outer_nc)
        else:
            upnorm = None

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv, downrelu]
            up = [upconv] 
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            if norm_layer is not None:
                down = [downconv, downnorm, downrelu]
                up = [upconv, upnorm, uprelu]
            else:
                down = [downconv, downrelu]
                up = [upconv, uprelu]

            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            if norm_layer is not None:
                down = [downconv, downnorm, downrelu]
                up = [upconv, upnorm, uprelu]
            else:
                down = [downconv, downrelu]
                up = [upconv, uprelu]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost and not self.outer_skip:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.

    Parameters:
        :param net: network to be initialized
        :param init_type: the name of an initialization method
        :param init_gain: scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or 
                    classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is \
                        not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class UnetGenerator(nn.Module):
    """
    Create a Unet-based generator that constructs the U-Net from the
    innermost layer to the outermost layer recursively.

    Parameters
    -----
    :param input_nc: the number of channels in input images
    :param output_nc: the number of channels in output images
    :param num_downs: the number of downsamplings in UNet. 
    :param nf0: the number of filters in the last conv layer
    :param max_channels: clamp the max filters in any conv layer
    :param norm_layer: normalization layer
    :param use_dropout: flag to use dropout layers
    :param outer_skip: add an additional outer skip connection
    """

    def __init__(self, input_nc=1, output_nc=1, num_downs=5, nf0=32, 
                 max_channels=128, norm_layer=nn.InstanceNorm2d, 
                 use_dropout=False, outer_skip=True):
        super(UnetGenerator, self).__init__()
        self.outer_skip = outer_skip
        self.input_nc = input_nc

        assert num_downs >= 2

        # Add the innermost layer
        unet_block = UnetSkipConnectionBlock(min(2 ** (num_downs - 1) * nf0,
                                                max_channels),
                                             min(2 ** (num_downs - 1) * nf0,
                                                max_channels),
                                             input_nc=None, submodule=None,
                                             norm_layer=norm_layer,
                                             innermost=True)

        for i in list(range(1, num_downs - 1))[::-1]:
            if i == 1:
                norm = None
            else:
                norm = norm_layer

            unet_block = UnetSkipConnectionBlock(min(2 ** i * nf0, 
                                                    max_channels),
                                                 min(2 ** (i + 1) * nf0, 
                                                    max_channels),
                                                 input_nc=None, 
                                                 submodule=unet_block,
                                                 norm_layer=norm,
                                                 use_dropout=use_dropout)

        # Add the outermost layer
        self.model = UnetSkipConnectionBlock(min(nf0, max_channels),
                                             min(2 * nf0, max_channels),
                                             input_nc=input_nc, 
                                             submodule=unet_block, 
                                             outermost=True,
                                             norm_layer=None, 
                                             outer_skip=self.outer_skip)
        if self.outer_skip:
            self.additional_conv = nn.Conv2d(input_nc + min(nf0, max_channels),
                                             output_nc, kernel_size=4, 
                                             stride=1, padding=2, bias=True)
        else:
            self.additional_conv = nn.Conv2d(min(nf0, max_channels), output_nc,
                                             kernel_size=4, stride=1, 
                                             padding=2, bias=True)

    def forward(self, cnn_input):
        """Standard forward"""
        output = self.model(cnn_input)
        output = self.additional_conv(output)
        output = output[:,:,:-1,:-1]
        return output
