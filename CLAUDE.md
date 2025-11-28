## Project Overview

**LyraCodec** is a deep learning audio compression system using a Variational Autoencoder (VAE) architecture. It compresses stereo audio from 44.1kHz down to a compact latent representation at ~21.5Hz, achieving a **2048x compression ratio**.

### Key Technologies

- **Framework**: PyTorch 2.6.0+ with CUDA 12.4 support
- **Language**: Python 3.13+
- **Package Manager**: uv (modern Python package manager)
- **Experiment Tracking**: Weights & Biases (wandb)
- **Audio Processing**: torchaudio, soundfile

## Repository Structure

```
LyraCodec/
├── src/
│   ├── model/                  # Neural network architecture
│   │   ├── VAE.py             # Main VAE model with encoder/decoder
│   │   ├── EncoderBlock.py    # Downsampling encoder block
│   │   ├── DecoderBlock.py    # Upsampling decoder block
│   │   ├── ResidualUnit.py    # Dilated conv residual unit
│   │   ├── SnakeActivation.py # Learnable periodic activation
│   │   └── __init__.py
│   └── train/                  # Training infrastructure
│       ├── train.py           # Main training script with VAETrainer class
│       ├── dataset.py         # VocalDataset for loading audio
│       ├── config.yaml        # Training hyperparameters
│       └── __init__.py
├── data/                       # Training data (gitignored)
│   └── vocal_dataset/
├── checkpoints/                # Model checkpoints (gitignored)
├── wandb/                      # Experiment logs (gitignored)
├── pyproject.toml             # Project dependencies
├── uv.lock                    # Locked dependency versions
├── .env                       # Environment variables (contains WANDB_API_KEY)
├── .gitignore
└── README.md                  # Architecture documentation
```

## Architecture Deep Dive

### Model Components

#### 1. **SnakeActivation** (`src/model/SnakeActivation.py`)

A learnable periodic activation function optimized for audio signals:

```python
Snake(x) = x + (1/β) × sin²(α × x)
```

- **α (alpha)**: Controls frequency of periodic component (learned per channel)
- **β (beta)**: Controls strength of the effect (learned per channel)
- Parameters initialized in log-space (as zeros) for stability
- Prevents vanishing gradients unlike ReLU

**Key Implementation Detail**: Parameters are stored in log-space and converted using `exp()` during forward pass to ensure positivity.

#### 2. **ResidualUnit** (`src/model/ResidualUnit.py`)

Multi-scale temporal processing using dilated convolutions:

```
Input → [Snake → Conv1d(k=7, dilation=d) → Snake → Conv1d(k=1)] + Input → Output
```

- Three dilations per block: [1, 3, 9] for local, mid, and long-range patterns
- Skip connections for gradient flow
- Preserves spatial dimensions (no downsampling)

#### 3. **EncoderBlock** (`src/model/EncoderBlock.py`)

Combines feature extraction with downsampling:

```
Input → 3x ResidualUnit → SnakeActivation → Conv1d(stride=s) → Output
```

- Refines features at current resolution before compression
- Kernel size: `2 * stride`, padding: `ceil(stride / 2)`

#### 4. **DecoderBlock** (`src/model/DecoderBlock.py`)

Mirror of EncoderBlock using transposed convolutions:

```
Input → 3x ResidualUnit → SnakeActivation → ConvTranspose1d(stride=s) → Output
```

- Same structure as encoder for symmetric reconstruction

#### 5. **VAE** (`src/model/VAE.py`)

Main model orchestrating encoder and decoder:

**Encoder Pipeline**:

```
[B, 2, 88200] → Conv1d → [B, 128, 88200]
  → EncoderBlock(stride=2)  → [B, 128, 44100]
  → EncoderBlock(stride=4)  → [B, 256, 11025]
  → EncoderBlock(stride=4)  → [B, 512, 2756]
  → EncoderBlock(stride=8)  → [B, 1024, 344]
  → EncoderBlock(stride=8)  → [B, 2048, 43]
  → Snake + Conv1d → [B, 128, 43]
  → Split → μ [B, 64, 43] and log(σ²) [B, 64, 43]
```

**Decoder Pipeline**:

```
[B, 64, 43] → Conv1d → [B, 2048, 43]
  → DecoderBlock(stride=8) → [B, 1024, 352] (padded/cropped to match encoder)
  → DecoderBlock(stride=8) → [B, 512, 2816]
  → DecoderBlock(stride=4) → [B, 256, 11264]
  → DecoderBlock(stride=4) → [B, 128, 45056]
  → DecoderBlock(stride=2) → [B, 128, 90112]
  → Snake + Conv1d → [B, 2, 90112]
```

**Critical Implementation Details**:

- Encoder stores intermediate shapes for decoder to match exactly
- Decoder uses padding/cropping to handle dimension mismatches from upsampling
- `logvar` is clamped to [-10, 10] to prevent extreme variances

### Training Infrastructure

#### **VocalDataset** (`src/train/dataset.py`)

- Loads audio files (.wav, .mp3) recursively from data directory
- Resamples to target sample rate (44.1kHz)
- Random cropping for data augmentation
- **Energy filtering**: Rejects silent samples (threshold: -40dB RMS)
- **Peak normalization**: Scales to target_peak=0.95

**Important**: Silent sample detection prevents training on silence, which can destabilize VAE training.

#### **VAETrainer** (`src/train/train.py`)

Comprehensive training loop with:

**Loss Function**:

```python
total_loss = recon_loss + kl_weight * kl_loss

recon_loss = stft_loss_fn(x_recon, x_original)
kl_loss = -0.5 * mean(1 + logvar - mu² - exp(logvar))
```

**KL Annealing**: Later on gradually increases KL weight over `kl_anneal_epochs` to prevent posterior collapse.

## Key Conventions for AI Assistants

### Code Style

1. **Type hints**: Not extensively used, but encouraged for new functions
2. **Docstrings**: Use for complex functions (see `reparameterization` method)
3. **Comments**: Inline comments for shape transformations (see VAE.encode)
4. **Naming**:
   - Snake_case for files and variables
   - PascalCase for classes
   - Descriptive names (e.g., `encoder_final_activation`)

### Model Architecture Conventions

1. **Channel ordering**: Always `[batch, channels, time]` for 1D convolutions
2. **Shape tracking**: Store intermediate shapes in encoder for decoder matching
3. **Padding formulas**: Document padding calculations (see ResidualUnit)
4. **Activation placement**: Snake activation before convolutions in ResidualUnits

### Training Conventions

1. **Device handling**: Always check `torch.cuda.is_available()` and move to device
2. **Mixed precision**: Wrap forward pass in `torch.cuda.amp.autocast()`
3. **Logging**: Use wandb for metrics and audio samples
4. **Checkpointing**: Save both regular and best model checkpoints

### File Organization Rules

1. **Model code**: All neural network modules in `src/model/`
2. **Training code**: Training logic, dataset, config in `src/train/`
3. **Data**: Place in `data/` directory (gitignored)

## Common Tasks

### Adding a New Model Component

1. Create new file in `src/model/`
2. Inherit from `nn.Module`
3. Import and use in parent model (e.g., VAE.py)
4. Update `src/model/__init__.py` if needed

## Important Notes

### GPU Memory Considerations

- Model has 255,858,050 parameters
- Batch size of 2-4 with mixed precision fits on ~8GB GPU
- Reduce batch size if OOM errors occur
- `target_length` directly impacts memory usage

## When Making Changes

### Before Modifying Model Architecture

1. Always think DEEPLY about shape transformations
2. Understand compression ratios from strides
3. Calculate output shapes for new layers
4. Update comments with shape transformations

## Quick Reference

### Key File Locations

- Main model: `src/model/VAE.py`
- Training entry point: `src/train/train.py`
- Dataset class: `src/train/dataset.py`

### Key Classes

- `VAE`: Complete autoencoder
- `EncoderBlock`: Downsampling block
- `DecoderBlock`: Upsampling block
- `ResidualUnit`: Dilated convolution unit
- `SnakeActivation`: Learnable activation
- `VAETrainer`: Training orchestrator
- `VocalDataset`: Audio data loader

### Key Parameters

- `in_channels=2`: Stereo audio
- `base_channels=128`: Initial feature dimension
- `latent_dim=64`: Compressed representation size
- `c_mults=[1, 2, 4, 8, 16]`: Channel multipliers
- `strides=[2, 4, 4, 8, 8]`: Downsampling factors
- `target_length=40960`: Audio samples per training example
- `batch_size=4`: Samples per batch
