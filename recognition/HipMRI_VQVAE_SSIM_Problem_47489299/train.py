
# === FILE: train.py ===
"""
Train script: trains VQ-VAE to reconstruct 2D slices.
Saves checkpoints and plots losses & SSIM metric.
"""

import os
import torch
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm 
from skimage.metrics import structural_similarity as ssim 
import numpy as np

from modules import VQVAE
from dataset import make_dataloaders


def compute_batch_ssim(x, x_recon):
    # x and x_recon: torch tensors Bx1xHxW in [0,1]
    x_np =        x.detach().cpu().numpy()
    xr_np = x_recon.detach().cpu().numpy()
    scores = []
    for i in range(x_np.shape[0]):
        a = x_np[i,0]
        b = xr_np[i,0]
        try:
            sc = ssim(a, b, data_range=1.0)
        except Exception:
            sc = 0.0
        scores.append(sc)
    return float(np.mean(scores))


def train(root='HipMRI_Study_open/keras_slices_data', epochs=50, batch_size=16, lr=2e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    train_loader, val_loader, _ = make_dataloaders(root, batch_size=batch_size)

    model = VQVAE(in_channels=1, z_channels=64, num_embeddings=512, hidden=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    recon_loss_fn = nn.MSELoss()

    history = {'train_loss':[], 'val_loss':[], 'val_ssim':[]}

    best_val_ssim = 0.0
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for xb in tqdm(train_loader, desc=f'Epoch {epoch} train'):
            xb = xb.to(device)
            optimizer.zero_grad()
            xr, vq_loss, _ = model(xb)
            recon_loss = recon_loss_fn(xr, xb)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        val_ssim = 0.0
        with torch.no_grad():
            for xb in val_loader:
                xb = xb.to(device)
                xr, vq_loss, _ = model(xb)
                loss = recon_loss_fn(xr, xb) + vq_loss
                val_loss += loss.item() * xb.size(0)
                val_ssim += compute_batch_ssim(xb, xr) * xb.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        val_ssim = val_ssim / len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_ssim'].append(val_ssim)

        print(f'Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, val_ssim={val_ssim:.4f}')

        # save best
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(save_dir, 'vqvae_best.pth'))

        # early stopping target SSIM reached
        if val_ssim >= 0.60:
            print(f'Target SSIM reached ({val_ssim:.4f}) at epoch {epoch}, saving and stopping.')
            break

    # Save training history and final model
    torch.save({'history': history}, os.path.join(save_dir, 'history.pth'))
    torch.save(model.state_dict(), os.path.join(save_dir, 'vqvae_final.pth'))

    # Plot training history
    plt.figure(figsize=(10, 4))

    # --- Plot 1: Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # --- Plot 2: SSIM ---
    plt.subplot(1, 2, 2)
    plt.plot(history['val_ssim'], label='Validation SSIM', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM (Structural Similarity Index)')
    plt.title('Validation Structural Similarity Over Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_plots.png'))
    plt.show()

    print('Training complete. Best val SSIM:', best_val_ssim)
    return model, history


if __name__ == '__main__':
    train(epochs=50, batch_size=16)
