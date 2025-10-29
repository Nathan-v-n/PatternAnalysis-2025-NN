
# === FILE: predict.py ===
"""
Example usage of trained model. Loads checkpoint, runs reconstruction on some
validation images and visualizes original vs reconstruction and prints SSIM.
"""

from modules import VQVAE
from dataset import make_dataloaders
from dataset import HipMRISlicesDataset


import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim 
from torch.utils.data import DataLoader
import os


def show_reconstructions(checkpoint='checkpoints/vqvae_best.pth',
                        root='HipMRI_Study_open',
                        n=6,
                        device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    model = VQVAE(in_channels=1, z_channels=64,
                  num_embeddings=512, hidden=128).to(device)
    
    state = torch.load(checkpoint, map_location=device)
    if 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)
    model.eval()

    ds = HipMRISlicesDataset(root, split='test')
    loader = DataLoader(ds, batch_size=n, shuffle=True)
    xb = next(iter(loader)).to(device)
    with torch.no_grad():
        xr, _, _ = model(xb)
    xb_np = xb.cpu().numpy()
    xr_np = xr.cpu().numpy()

    os.makedirs("predict_output", exist_ok=True)

    for i in range(min(n, xb_np.shape[0])):
        orig = xb_np[i,0]
        recon = xr_np[i,0]
        sc = ssim(orig, recon, data_range=1.0)
        print(f'Example {i} SSIM: {sc:.4f}')
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.title('Original')
        plt.imshow(orig, cmap='gray')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.title('Reconstruction')
        plt.imshow(recon, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join("predict_output", f'example_{i}_SSIM_{sc:.6f}.png'))
        plt.show()


if __name__ == '__main__':
    show_reconstructions()
