# Attempted Reimplementation of the iVGAE model from the following paper:
[Relating Graph Neural Networks to Structural Causal Models](https://arxiv.org/abs/2109.04173)

# Packages
Pytorch 1.12.0 
- conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

Pytorch Geometric 
- conda install pyg -c pyg

Base start from PytorchGeometric tutorial: 
https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial6/Tutorial6.ipynb

# Base model
The base model can be ran on the Citeseer dataset by running:
`python main.py`