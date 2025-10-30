# AI References:

Only used chatGPT once to give me a base implementation of the problem to then
iterate over. It did really well on the very first go.

## Prompt:
Createa  base implementation of the following problem/description to then 
iterate and work over

Create a generative model of the HipMRI Study on Prostate Cancer using the 
processed 2D slices (2D images) with the using a VQVAE [12] or VQVAE2 [13] 
that has a “reasonably clear image” and a Structured Similarity (SSIM) of 
over 0.6

Your implementation must include the following files
1. “modules.py" containing the source code of the components of your model. 
Each component must be implementated as a class or a function
2. “dataset.py" containing the data loader for loading and preprocessing
 your data
3. “train.py" containing the source code for training, validating, testing and
 saving your model. The model should be imported from “modules.py” and 
 the data loader should be imported from “dataset.py”. Make sure to plot 
 the losses and metrics during training
4. “predict.py" showing example usage of your trained model. Print out any
 results and / or provide visualisations where applicable


Assume there is a folder with the data with the following file structure:

HipMRI_Study_open/keras_slices_data\
which has these folders: \
keras_slices_train\
keras_slices_test\
keras_slices_validate

which all have files of type .nii.gz


## Response:
```python
# === FILE: modules.py ===
"""
VQ-VAE implementation (simple VQ-VAE) using PyTorch.
Components implemented as classes/functions:
 - Encoder
 - Decoder
 - VectorQuantizer
 - VQVAE (wrapper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden=128, z_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(hidden//2, hidden, 4, 2, 1)
        self.conv3 = nn.Conv2d(hidden, z_channels, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels=1, hidden=128, z_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(z_channels, hidden, 3, 1, 1)
        self.deconv1 = nn.ConvTranspose2d(hidden, hidden//2, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(hidden//2, out_channels, 4, 2, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.relu(self.conv1(z))
        x = self.relu(self.deconv1(x))
        x = self.deconv2(x)
        # assume normalized images in [0,1]
        x = torch.sigmoid(x)
        return x


class VectorQuantizer(nn.Module):
    """A simple VQ layer (EMA NOT used)."""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs):
        # inputs: (B, C, H, W)
        input_shape = inputs.shape
        flat_input = inputs.permute(0,2,3,1).contiguous()
        flat_input = flat_input.view(-1, self.embedding_dim)  # (B*H*W, C)

        # Calculate distances
        distances = (flat_input.pow(2).sum(1, keepdim=True)
                     - 2 * flat_input @ self.embedding.weight.t()
                     + self.embedding.weight.pow(2).sum(1, keepdim=True).t())

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = encodings @ self.embedding.weight  # (B*H*W, D)
        quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], self.embedding_dim)
        quantized = quantized.permute(0,3,1,2).contiguous()

        # Losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        encoding_indices = encoding_indices.view(input_shape[0], input_shape[2], input_shape[3])

        return quantized, loss, perplexity, encoding_indices


class VQVAE(nn.Module):
    def __init__(self, in_channels=1, z_channels=64, num_embeddings=512, hidden=128):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, hidden=hidden, z_channels=z_channels)
        self.pre_vq_conv = nn.Conv2d(z_channels, z_channels, 1, 1)
        self.vq = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=z_channels)
        self.decoder = Decoder(out_channels=in_channels, hidden=hidden, z_channels=z_channels)

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_vq_conv(z_e)
        quantized, vq_loss, perplexity, enc_inds = self.vq(z_e)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, perplexity


# === END modules.py ===


# === FILE: dataset.py ===
"""
Data loader for HipMRI 2D slices stored as .nii.gz volumes. We extract slices,
resize them to a fixed size and normalize to [0,1].

Expect the dataset root to look like:
HipMRI_Study_open/keras_slices_data/keras_slices_train/ ... .nii.gz
Additional directories provided (semantic labels etc) are not required for this
proof-of-concept loader but basic hooks are added.
"""

import os
from glob import glob
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms.functional as TF
from skimage.transform import resize


def load_nii_slices(path):
    img = nib.load(path).get_fdata()
    # assume shape (H, W, Slices) or (X,Y,Z)
    if img.ndim == 4:
        img = img[...,0]
    return img


class HipMRISlicesDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_size=(128,128), max_slices_per_volume=9999):
        self.root = os.path.join(root_dir, 'keras_slices_data')
        self.split = split
        self.dir = os.path.join(self.root, f'keras_slices_{split}')
        self.files = sorted(glob(os.path.join(self.dir, '*.nii.gz')))
        self.transform = transform
        self.target_size = target_size
        self.max_slices_per_volume = max_slices_per_volume
        self.samples = []  # list of (volume_path, slice_idx)
        for f in self.files:
            vol = load_nii_slices(f)
            n_slices = vol.shape[2]
            for s in range(min(n_slices, self.max_slices_per_volume)):
                self.samples.append((f, s))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f, s = self.samples[idx]
        vol = load_nii_slices(f)
        slice_img = vol[:,:,s]
        # normalize to 0-1
        slice_img = slice_img.astype(np.float32)
        # simple percentile clip
        p1, p99 = np.percentile(slice_img, (1, 99))
        slice_img = np.clip(slice_img, p1, p99)
        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
        # resize
        if slice_img.shape != self.target_size:
            slice_img = resize(slice_img, self.target_size, preserve_range=True, anti_aliasing=True)
        # add channel
        slice_img = np.expand_dims(slice_img, 0)
        tensor = torch.from_numpy(slice_img).float()
        if self.transform:
            tensor = self.transform(tensor)
        return tensor


def make_dataloaders(root, batch_size=16, target_size=(128,128), num_workers=4):
    train_ds = HipMRISlicesDataset(root, split='train', target_size=target_size)
    val_ds = HipMRISlicesDataset(root, split='validate', target_size=target_size)
    test_ds = HipMRISlicesDataset(root, split='test', target_size=target_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


# === END dataset.py ===


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
    x_np = x.detach().cpu().numpy()
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


def train(root='HipMRI_Study_open', epochs=50, batch_size=16, lr=2e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    train_loader, val_loader, test_loader = make_dataloaders(root, batch_size=batch_size)

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

    # Plot
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(history['val_ssim'], label='val_ssim')
    plt.legend()
    plt.title('Validation SSIM')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_plots.png'))
    plt.close()

    print('Training complete. Best val SSIM:', best_val_ssim)
    return model, history


if __name__ == '__main__':
    train(epochs=50, batch_size=16)

# === END train.py ===


# === FILE: predict.py ===
"""
Example usage of trained model. Loads checkpoint, runs reconstruction on some
validation images and visualizes original vs reconstruction and prints SSIM.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from modules import VQVAE
from dataset import HipMRISlicesDataset
from torch.utils.data import DataLoader


def show_reconstructions(checkpoint='checkpoints/vqvae_best.pth', root='HipMRI_Study_open', n=6, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = VQVAE(in_channels=1, z_channels=64, num_embeddings=512, hidden=128).to(device)
    state = torch.load(checkpoint, map_location=device)
    if 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)
    model.eval()

    ds = HipMRISlicesDataset(root, split='validate', target_size=(128,128))
    loader = DataLoader(ds, batch_size=n, shuffle=True)
    xb = next(iter(loader)).to(device)
    with torch.no_grad():
        xr, _, _ = model(xb)
    xb_np = xb.cpu().numpy()
    xr_np = xr.cpu().numpy()

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
        plt.show()


if __name__ == '__main__':
    show_reconstructions()

# === END predict.py ===
```



I've created the project files (modules.py, dataset.py, train.py, predict.py) in a canvas titled Hipmri Vqvae Project. Open the canvas to view, copy, and run the code.

Quick notes to run:

Install required packages: pip install torch torchvision nibabel scikit-image matplotlib tqdm scikit-image

Put the HipMRI_Study_open folder beside the scripts (or change paths).

Run training: python train.py (adjust epochs, batch_size in the file or call train() with args).

View reconstructions: python predict.py (it will load checkpoints/vqvae_best.pth or vqvae_final.pth).

If you'd like:

I can adapt the model to VQVAE-2 style hierarchical architecture.

Add augmentation, stronger perceptual losses (LPIPS), or EMA VQ updates to improve image clarity and SSIM.

Convert to a Jupyter notebook or provide a Dockerfile for reproducible runs.

Open the canvas and tell me if you want changes (bigger images, different target size, or VQVAE2).