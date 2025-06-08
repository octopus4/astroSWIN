# AstroSWIN
![astroswin_tiny](https://github.com/user-attachments/assets/b0771b0d-2189-4a60-9c5f-a404aa7dadb8)

**Swin Transformer for Astrophotography Image Enhancement**  

## Overview

This repository adapts the Swin Transformer architecture for astrophotography image enhancement for joint deconvolution and denoise

## Examples

1. NGC4565 (Needle Galaxy)
<img src="https://github.com/user-attachments/assets/b9f7f166-0305-4a1a-8a15-86f6b7cc583a" title="NGC4565 Needle Galaxy" width="500px">

2. NGC3031 Bode's Galaxy
<img src="https://github.com/user-attachments/assets/8d4643bc-8717-4458-9769-a4899c6ad134" title="NGC3031 Bode's Galaxy" width="500px">

## Contents

The repository includes five core notebooks:
1. Data Collection
    - `collect_astrobin_data.ipynb`: Fetches "Image of the Day" archive (IOTD's) from [AstroBin](https://www.astrobin.com/) in jpeg format for efficiency
    - `collect_hubble.ipynb`: Downloads Hubble Space Telescope images from [ESA/Hubble](https://esahubble.org/) archives (partial dataset)
2. Preprocessing
    - `preprocessing.ipynb`: Pipeline to generate 256×256 patches from source images and split them into train/test sets.
    _Note: Is not necessary for current training pipeline_
3. Training
    - `train_aswin.ipynb`: Custom training loop with:
        - Loss functions (Brightness Loss, Gradient Loss, Histogram Loss, Adaptive Background Loss)
        - Blur augmentations
        - Mixed-precision training optimizations
4. Inference
    - `infer_astroswin.ipynb`: Production-ready Colab notebook for model inference.

## Technical limitations

### First iteration (version 0.7)

- Hardware: Trained on NVIDIA GTX 1660Ti Mobile (6GB)
- Optimizations:
    - 256×256 pre-made patches (tradeoff between VRAM limits and detail preservation)
    - Mixed-precision training (fp16)
    - Gradient checkpointing
    - Gradient accumulation steps = 2

### Second iteration (version 1.0)

- Hardware: Trained on Google Colaboratory Tesla T4
- Optimizations:
    - Mixed-precision training (fp16)
    - Gradient checkpointing
    - Gradient accumulation steps = 4

## Training process

The training pipeline consists of two major phases:
1. **Domain Adaptaion** on noisy and heterogeneous-quality data
2. **Fine-Tuning** on curated high-quality data
3. **Further tuning** on _bigger_ amount of data with proper loss functions

### Stage 1: Domain Adaptation

**Objective:** Learn astrophotography-specific features
**Implementation Details:**
0. Checkpoint:
    - `caidas/swin2SR-lightweight-x2-64`: [huggingface](https://huggingface.co/caidas/swin2SR-lightweight-x2-64/tree/main)
1. Dataset:
    - Astrobin IotD JPEG thumbnails, randomly cropped to 256x256 size during batch generation
2. Initial setup:
    - Gaussian Blur ($\sigma \in [1.0, 3.0]$) via torchvision.transforms.GaussianBlur as augmentation
    - Loss: Combined MSE (L2) + MAE (L1) with equal weights
3. Iterative Refinement:
    - Epoch 1: Baseline training revealed accurate dust lane reconstruction but poor stellar core handling (artifact-prone star shapes).
    - Epoch 2: Introduced Gradient Loss (L1 on image gradients) to emphasize edge preservation, which helped to improve star detection but introduced chromatic aberrations
    - Epoch 3: Added Histogram Loss as an absolute error between the target image histogram and the output image histogram
    - Epoch 4-5:
        - Added Brightness Mask Loss as an L1/L2 applied to sigmoid-generated masks for bright/dark regions
        - Tuned loss components weights in order to get admissible results
        - Created distortions pool to sample from (Gaussian Blur, Motion Blur, Bokeh Blur, Anisotropic Gaussian Blur)

### Stage 2: Fine-Tuning

**Objective:** Balance sharpness and naturalness while mitigating overprocessing.
**Key Adjustments:**
- Dataset Curation:
    - Remove low-quality images from astrobin
    - Integrated ESAHubble "Large" images (20% of dataset)
    - Limitations: due to VRAM constraints I pre-generated possibly overlapping patches from every image in the dataset
- Architectural Changes:
    - Adaptive Homogeneous Background Loss (AHBG): Penalizes high-frequency noise in low-variance regions
    - 256×256 patches → Train/Test split (70/30)
    - Tradeoff: Risk of overfitting to local patterns
- Training Schedule:
    - Epoch 6: AHBG-dominated weighting ($\lambda = 1.5$), outcome: smoother backgrounds but undersaturated colors (astroswin_v06)
    - Epoch 7: Histogram-dominated weighting, outcome: natural tones with preserved details (astroswin_v07)

### Stage 3: Error correction

**Objective:** Deal with noisy & crispy background
**Key Adjustments:**
- Dataset Extension:
    - Add more than 2GB of images from ESAHubble library
    - Add nebulae images from ESAHubble
    - Tiny split proportion to preserve more data for train
- Architectural Changes:
    - Remove masked L1/L2 losses, instead added sharpness-aware loss based on _mix_ of discrete Laplace operator and L1 loss
    - Add random noise to image during the training process
    - Add random downsampling for big images so the model could process different scales
    - Create random combinations of blur functions
    - Set weights as follows in the training notebook: 2.0 for mixed loss of L1 and laplacian-based, 1.0 for gradient loss, 1.0 for histogram loss

## Future work

As a follow-up, I am going to parameterize model with deconvolution strength parameter.

## Sources

- [Dataset](https://drive.google.com/file/d/1v1HvfuoQPMprDa5FgRvjRMlwIwt5mp84/view?usp=sharing)
- [Trained model checkpoint](https://drive.google.com/file/d/1WHKDZhAOKSegPRgYi2iFizky_14mbBht/view?usp=sharing)

## Acknowledgments

- AI-assisted development with [Deepseek-R1](https://www.deepseek.com/)
- Initial code concepts inspired by [SwinIR](https://github.com/JingyunLiang/SwinIR)
- Dataset sources: [AstroBin](https://www.astrobin.com/), [ESA/Hubble](https://esahubble.org/)

-------------------------------------------------
License: [MIT](https://choosealicense.com/licenses/mit/) | Contact: laokrit@gmail.com

*Model weights available under CC BY-NC-SA 4.0 for non-commercial use*
