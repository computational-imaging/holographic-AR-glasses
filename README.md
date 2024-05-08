# Full-Colour 3D Holographic Augmented-Reality Displays with Metasurface Waveguides <br> Nature 2024
### [Project Page](http://www.computationalimaging.org/publications/holographicAR/) | [Paper](https://www.nature.com/articles/s41586-024-07386-0)
PyTorch implementation of hologram synthesis techniques in <br>
[Full-Colour 3D Holographic Augmented-Reality Displays with Metasurface Waveguides](http://www.computationalimaging.org/publications/holographicAR/)<br>
 [Manu Gopakumar](https://manugopa.github.io)\*,
 [Gun-Yeal Lee](https://gunyeal.github.io/)\*,
 [Suyeon Choi](https://choisuyeon.github.io/),
 [Brian Chao](https://bchao1.github.io/),
 [Yifan Peng](https://www.eee.hku.hk/~evanpeng/),
 [Jonghyun Kim](https://research.nvidia.com/person/jonghyun-kim),
 [Gordon Wetzstein](https://web.stanford.edu/~gordonwz/)<br>
  \*denotes equal contribution  
in Nature 2024

## Getting started
Our code primarily builds on [PyTorch Lightning](https://www.pytorchlightning.ai/) and PyTorch.
A conda environment with all dependencies required to run our code can be set up with the below commands:
```
conda env create -f environment.yml
conda activate holographicAR
```

For many of the commands it will be necessary to have trained instances of our learned physical waveguide models. Red, green, and blue models fit to our prototype are available on [Google Drive](https://drive.google.com/file/d/1lonVIcUy9CGpEPR0ATLHfSwuvWLixIVW/view?usp=sharing). To use these models, unzip this folder and place the .ckpt files in the models directory under the root folder of this codebase. 

## High-Level structure
The code is organized as follows:

`./`
* ```main.py``` is a script that runs the SGD hologram synthesis algorithm with different propagation models.
* ```simulate_setup.py``` is a script that can be used to simulate the focal stack that would be seen through the waveguide for different phase patterns using the learned physical waveguide model.
* ```data_loader.py``` contains modules for loading target scenes.
* ```CGH.py``` contains the gradient descent CGH algorithm implemented for 2D/3D supervision.
* ```freespace.py``` contains an angular spectrum method (ASM) implementation for freespace propagation. 
* ```physical_waveguide.py``` contains implementations for the physically-modelled phenomena in the waveguide. Specifically, analytical expressions are implemented to model converging illumination and the waveguide transfer function.
* ```prop_model.py``` creates instances of the different simulated wave propagation models. The conventional freespace model uses the ASM module from freespace.py. Our proposed waveguide models are constructed here using the analytically-derived modules from physical_waveguide.py.
* ```params.py``` defines the parameters of our physical setup. 
* ```network_modules.py``` contains the modules used to construct the CNNs in our learned physical waveguide model. 
* ```utils.py``` has some helpful utility functions.

`./models/` is the location to put the trained checkpoints for the learned physical waveguide model.

`./data/` contains example 2d and 3d target scenes

## Example commands
Using the gradient descent CGH algorithm, our codebase can optimize phase patterns to produce desired holograms using different wave propagation models.
The commands for synthesizing 2d holograms are as follows:
```
# Generating 2d holograms using the freespace propagation model
python main.py --data_path=./data/2d/ --channel=$c --target=2d --eval_plane_idx=$i --prop_model=freespace

```
```
# Generating 2d holograms using the physical waveguide model
python main.py --data_path=./data/2d/ --channel=$c --target=2d --eval_plane_idx=$i --prop_model=physical

```
```
# Generating 2d holograms using the learned physical waveguide model
python main.py --data_path=./data/2d/ --channel=$c --target=2d --eval_plane_idx=$i --prop_model=learnedphysical --prop_model_path=models/

```
In these commands, holograms can be generated for red, green, and blue illumination wavelengths by setting the argument for --channel=$c to 0, 1, and 2 respectively.
For 2d holograms, the depth of the hologram is specified by passing the index of the depth plane as the argument for --eval_plane_idx=$i.
Setting eval_plane_idx to 0, 1, 2, or 3 results in 2d holograms that appear 0, 1/3, 2/3, or 1 diopter away from the user respectively.

The codebase can also be used to generate multiplane 3d holograms:
```
# Generating 3d holograms using the freespace propagation model
python main.py --data_path=./data/3d/ --channel=$c --target=3d --prop_model=freespace

```
```
# Generating 3d holograms using the physical waveguide model
python main.py --data_path=./data/3d/ --channel=$c --target=3d --prop_model=physical

```
For generating 3d holograms with the learned physical waveguide model, a memory-efficient variant of the gradient descent algorithm has been implemented to limit the peak GPU memory usage. This variant can be enabled by setting the argument --mem_eff=True.
```
# Generating 3d holograms using the learned physical waveguide model
python main.py --data_path=./data/3d/ --channel=$c --target=3d --prop_model=learnedphysical --prop_model_path=models/ --mem_eff=True

```

Since the learned physical waveguide model can accurately predict the view through our AR glasses, it can be used as a proxy to simulate the focal stack that would be seen through our setup for phase patterns generated with different CGH implementations.
To simulate the output of our setup for a specific phase pattern, the script 'simulate_setup.py' can be called with the argument --phase_path set to the location of the phase pattern.

```
# Simulate the focal stack seen through our AR glasses when the phase pattern at 'results/p0_green_freespace/lion.png' is displayed on the SLM.
python simulate_setup.py --channel=1 --prop_model_path=models/ --phase_path=results/p0_green_freespace/lion.png

```
The script will output the predicted virtual content focal stack in the same folder as the corresponding phase pattern.

### Citation
If you find our work useful in your research, please cite:
```
@article{Gopakumar:2024:HolographicAR,
         author = {Manu Gopakumar
                   and Gun-Yeal Lee
                   and Suyeon Choi
                   and Brian Chao
                   and Yifan Peng
                   and Jonghyun Kim
                   and Gordon Wetzstein},
         title = {{Full-colour 3D holographic augmented-reality displays with metasurface waveguides}},
         journal = {Nature},
         year = {2024},
}
```

### Contact
If you have any questions, please email Manu Gopakumar at manugopa@stanford.edu.
