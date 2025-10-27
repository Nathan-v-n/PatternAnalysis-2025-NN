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

