# Generative VQ-VAE Model for HipMRI Study on Prostate Cancer

## Contents
[Overview](#overview)\
[How It Works](#how-it-works)\
[File Structure](#file-structure)\
[Pre-processing](#pre-processing)\
[Data Splitting and Justification](#data-splitting-and-justification)\
[Results](#results)\
[Dependencies](#dependencies)\
[How To Run](#how-to-run)

---

## Overview
This project implements a **Vector Quantized Variational Autoencoder (VQ-VAE)** to model **2D prostate MRI slices** from the HipMRI Study on Prostate Cancer

The goal is to create a model capable of reconstructing realistic MRI images, achieving a **Structural Similarity Index (SSIM)** greater than **0.6**, with “reasonably clear” reconstructed images.

This model learns a **discrete latent representation** of MRI structures, a codebook of visual patterns, that can later be used for both **reconstruction** and **generation** of new realistic MRI-like images.

---

## How It Works
The implemented VQ-VAE follows three main components:

1. **Encoder:** Compresses input MRI slices into a lower-dimensional latent representation.  
2. **Vector Quantizer:** Maps continuous encoder outputs into discrete latent “codebook” entries.  
3. **Decoder:** Reconstructs images from quantized embeddings.

During training, the model minimizes:
- **Reconstruction loss (MSE)**: ensures pixel-level accuracy.
- **VQ loss:** ensures discrete latent codes stay close to encoder outputs.

After training, the decoder can generate *new* MRI-like samples by sampling from the learned codebook — making this a **generative** model.

---

## File Structure
├── modules.py # Model architecture (Encoder, Decoder, VectorQuantizer, VQVAE)\
├── dataset.py # Data loader and preprocessing for MRI slices\
├── train.py # Training, validation, plotting of metrics\
├── predict.py # Reconstruction and SSIM evaluation\
├── checkpoints/ # Saved model weights and plots\
└── README.md # Documentation (this file)\


---

## Pre-processing
All `.nii.gz` MRI volumes are loaded using **NiBabel** and split into individual 2D slices.  
Preprocessing includes:
- Intensity normalization to `[0, 1]`
- Percentile clipping (1st–99th percentile)
- Resizing to `(128 × 128)` pixels using bilinear interpolation (This resizing is technically done when the slice is requested from the HipMRISlicesDataset datatype as defined in its \_\_getitem__)
- Conversion to single-channel tensors

These steps stabilize training and reduce sensitivity to MRI contrast differences across scans.


---

## Data Splitting and Justification
The dataset includes  `train`, `test` and `validate` folders\
So, each folder is used for its repective purpose in training (`train` and `validate`) or testing (`test`)

- **Training set:** `keras_slices_train`  
- **Validation set:** `keras_slices_validate`  
- **Testing set:** `keras_slices_test`


---

## Results

---
## Dependencies
| Library | Version (tested) | Purpose |
|----------|------------------|----------|
| `torch` | ≥2.0 | Deep learning framework |
| `torchvision` | ≥0.15 | Image utilities |
| `nibabel` | ≥5.1 | MRI data loading |
| `scikit-image` | ≥0.22 | SSIM computation |
| `numpy` | ≥1.24 | Array operations |
| `matplotlib` | ≥3.7 | Plotting losses |
| `tqdm` | ≥4.66 | Progress bars |

## How To Run
To run this, you need to first make sure all the above libraries are installed as some are not standard. This can be done via the following in your terminal
```bash
pip install torch torchvision nibabel scikit-image matplotlib tqdm
```
Then you must run train.py
```bash
python train.py
```
This will train a model with a max of 50 epochs, stopping only when either 50 epochs is reached or the set target SSIM is reached. Which is currently set at 0.9. 

And then the predict.py model
