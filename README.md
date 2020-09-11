## MSAF: Multimodal Split Attention Fusion

If you use this code, please cite our paper:
```

```

## Installation
**Method 1**: Using requirements.txt (installs dependencies for all three datasets)
```
conda create --name msaf --file requirements.txt
```

**Method 2**: Without requirements.txt

This code was developed with Python 3.6, PyTorch 1.5.1 in Ubuntu 20.04
- Basic dependencies (needed for all datasets): [Pytorch](https://pytorch.org/get-started/previous-versions/), Tensorboard
- Dataset specific dependencies: see README file in each dataset folder
    - [RAVDESS](ravdess/README.md)
    - [CMU-MOSEI](cmu_mosei/README.md)
    - [NTU RGB+D](ntu_rgb_d/README.md)

## Usage
- The MSAF module is available in MSAF.py
- For training and evaluation on the datasets, see README in each folder for more details
