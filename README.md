# Attempted Reimplementation of the iVGAE model from the following paper:
[Relating Graph Neural Networks to Structural Causal Models](https://arxiv.org/abs/2109.04173)

# Packages

Pytorch 1.12.0  
`conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge`  
  
Pytorch Geometric  
`conda install pyg -c pyg`  
  
Matplotlib  
`conda install matplotlib`  
  
Base start from PytorchGeometric tutorial:  
https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial6/Tutorial6.ipynb

# iVGAE Test Run
The base model can be tested on the Citeseer dataset by running:  
`python main.py`

# Training without interventions
The base model can be trained on either the ASIA (asia) or the Earthquake (earthquake) datasets:  
`python train.py --dset asia --epochs 200`  

# Training with interventions
To train with an intervention, provide a list of variables to intervene on and their probabilities:  
`python train.py --dset asia --epochs 200 --i_vars T --i_probs 0.50`  

# Testing without interventions
TBA  

# Testing with interventions
TBA  
