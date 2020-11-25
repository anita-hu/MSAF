## MSAF: Multimodal Split Attention Fusion

Code for the paper [MSAF: Multimodal Split Attention Fusion](). This is our implementation of the MSAF module and the three MSAF-powered multimodal networks. 

If you use this code, please cite our paper:
```

```

## Installation
Clone this repo along with submodules
```
git clone --recurse-submodules https://github.com/anita-hu/MSAF.git
```
Install dependencies

**Method 1**: Using environment.yml (installs dependencies for all three datasets)

With this method, you can skip dependency installation steps from the dataset specific README files
```
conda env create -f environment.yml
```

**Method 2**: Without environment.yml

This code was developed with Python 3.6, PyTorch 1.5.1 in Ubuntu 20.04
- Basic dependencies (needed for all datasets): [Pytorch](https://pytorch.org/get-started/previous-versions/), Tensorboard
- Dataset specific dependencies: see README file in each dataset folder
    - [RAVDESS](ravdess/README.md)
    - [CMU-MOSEI](cmu_mosei/README.md)
    - [NTU RGB+D](ntu_rgb_d/README.md)

## Usage
- The MSAF module is implemented in MSAF.py
- The README file in each dataset folder has details on data preprocessing, training and evaluation
