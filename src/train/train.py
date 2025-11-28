import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import auraloss
import torchaudio

from src.model.VAE import VAE
from src.train.dataset import VocalDataset


# ============================================
# LOSS FUNCTION
# ============================================
def compute_loss(x_recon, x_original, mu, logvar, stft_loss_fn):
    """
    STFT reconstruction loss + KL divergence
    """
    recon_loss = stft_loss_fn(x_recon, x_original)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + 1e-4 * kl_loss
    return recon_loss, kl_loss, total_loss

if __name__ == '__main__':
    # ============================================
    # HYPERPARAMETERS
    # ============================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Training hyperparameters
    batch_size = 2          # How many audio clips to process at once
    learning_rate = 1e-4    # AdamW learning rate
    num_epochs = 5000       # How many times to go through the entire dataset
    print_every = 5         # Print loss every N steps

    # Data hyperparameters
    target_length = 40960   # ~0.93s
    target_sr = 44100       # Sample rate

    # Model hyperparameters (matching your VAE architecture)
    in_channels = 2
    base_channels = 128
    latent_dim = 64
    c_mults = [(1, 1), (1, 2), (2, 4), (4, 8), (8, 16)]
    strides = [2, 4, 4, 8, 8]

    # ============================================
    # LOAD DATASET
    # ============================================
    print("Loading dataset...")
    dataset = VocalDataset(
        "data/vocal_dataset",
        target_length=target_length,
        target_sr=target_sr
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"Dataset loaded: {len(dataset)} audio files")
    print(f"Steps per epoch: {len(dataloader)}")

    # ============================================
    # INITIALIZE MODEL
    # ============================================
    print("Initializing VAE model...")
    model = VAE(
        in_channels, 
        base_channels, 
        latent_dim, 
        c_mults, 
        strides
        ).to(device)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ============================================
    # INITIALIZE LOSS FUNCTIONS
    # ============================================
    stft_loss = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[2048, 1024, 512, 256, 128, 64, 32],
        hop_sizes=[512, 256, 128, 64, 32, 16, 8],
        win_lengths=[2048, 1024, 512, 256, 128, 64, 32],
        perceptual_weighting=True,
        sample_rate=target_sr,
    ).to(device)

    print("STFT loss initialized with 7 resolutions")

    # ============================================
    # INITIALIZE OPTIMIZER
    # ============================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.8, 0.99), # From the paper
        weight_decay=0.001 # L2 regularization
    )
    print(f"Optimizer: AdamW with lr={learning_rate}")

    # ============================================
    # TRAINING LOOP
    # ============================================
    print("\nStarting training...")
    print("=" * 50)

    global_step = 0

    for epoch in range(num_epochs):
        model.train()

        for batch_idx, audio_batch in enumerate(dataloader):
            print(f"Processing batch {batch_idx+1}/{len(dataloader)}", end='\r')
            # Move batch to device
            audio_batch = audio_batch.to(device)

            # Forward to model
            mu, logvar, z, x_recon = model(audio_batch)

            # Compute loss
            recon_loss, kl_loss, total_loss = compute_loss(x_recon, audio_batch, mu, logvar, stft_loss)

            # Backwards pass
            optimizer.zero_grad()
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if global_step % print_every == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Step [{global_step}] "
                        f"Loss: {total_loss.item():.4f} | "
                        f"Recon: {recon_loss.item():.4f} | "
                        f"KL: {kl_loss.item():.4f}")
                
            # Safe original and reconstructed audio after every 20 steps
            # Create sample outputs every 20 steps
            if global_step % 20 == 0:
                # Simply save all the reconstructions from the current batch
                sample_dir = Path("samples")
                sample_dir.mkdir(exist_ok=True)
                for i, recon in enumerate(x_recon):
                    sample_path = sample_dir / f"recon_epoch{epoch+1}_step{global_step}_sample{i}.wav"
                    original_path = sample_dir / f"original_epoch{epoch+1}_step{global_step}_sample{i}.wav"
                    # Save original
                    torchaudio.save(original_path, audio_batch[i].detach().cpu(), target_sr)
                    # Save reconstruction
                    torchaudio.save(sample_path, recon.detach().cpu(), target_sr)
                    print(f"Saved sample reconstruction to {sample_path}")
                
            global_step += 1

    print("Training complete")