# ORBIT Project Information

## Project Name

**ORBIT** - Optimized Reasoning for Brain Inference & Timelines

## Overview

ORBIT is a method for enhancing spatiotemporal disease progression models via latent diffusion and prior knowledge. It enables individual-based spatiotemporal disease progression on 3D brain MRIs using latent diffusion models.

## Key Features

- **Latent Diffusion Models**: Uses autoencoders and diffusion models in latent space
- **ControlNet Integration**: Incorporates prior knowledge for better progression modeling
- **Individual-based Progression**: Predicts disease progression for individual patients
- **3D Brain MRI Support**: Works with full 3D brain MRI volumes

## Acronym

**ORBIT** = **O**ptimized **R**easoning for **B**rain **I**nference & **T**imelines

## Installation

```bash
cd ORBIT
pip install -e .
```

This installs the `orbit` package and creates the `orbit` CLI command.

## Usage

### CLI Command

```bash
orbit --help
orbit --input input.csv --output output_dir --confs config.yaml --target_age 80 --target_diagnosis 2 --steps 10
```

### Training

ORBIT training consists of 3 main phases:
1. **Autoencoder Training**: Train the latent space encoder/decoder
2. **UNet Training**: Train the diffusion model in latent space
3. **ControlNet Training**: Train the control network with prior knowledge

## Author

**Rohan Tennety**

## Resources

- **Paper (MedIA)**: [Medical Image Analysis](https://www.sciencedirect.com/science/article/pii/S1361841525002816)
- **Paper (MICCAI)**: [MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/0511_paper.pdf)
- **Video**: [YouTube](https://youtu.be/6YKz2MNM4jg?si=nkG21K4lIgLrH-pK)

## Note

The internal package structure still uses `brlp` folder names for compatibility, but the user-facing package name and CLI command are `orbit`.

