# Robustness Evaluation against Corruptions for ODT-based Classifiers

This repository contains the Pytorch implementation for our paper.


## Overview
We propose the first comprehensive robustness evaluation framework for deep learning classifiers based on Optical Diffraction Tomography (ODT). Despite ODT's powerful label-free 3D imaging capabilities, the sensitivity of AI models to corruptions in ODT inputs has not been systematically addressed.

Our work presents:
- A corruption benchmark customized for ODT images
- A novel metric, Calibrated Corrupted Error (CCE), to fairly assess performance degradation
- Extensive robustness testing across four model architectures and diverse corruption types
- A new data augmentation method, \cutpix, which improves robustness by blending shape and texture features

## Making corrupted datasets

The `utils/corruption.py` file provides a comprehensive set of functions to create corrupted versions of ODT datasets. These corruptions simulate various real-world imaging artifacts and noise types that can affect ODT systems.


```python
from utils.corruption import make_noise, load_mat

# Load your ODT volume
volume = load_mat("path/to/your/volume.mat")

# Apply corruption
corrupted_volume = make_noise(
    volume=volume,
    noise_type="gaussian_noise",  # Choose from available corruption types
    severity=3  # Severity level (1-5)
)
```

### Severity Levels

Each corruption can be applied with different severity levels (typically 1-5):
- Level 1: Mild corruption
- Level 2: Moderate corruption
- Level 3: Strong corruption
- Level 4: Severe corruption
- Level 5: Extreme corruption