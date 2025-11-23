# CLAUDE.md - LyraCodec Development Guide

## Project Overview

**LyraCodec** is a deep learning audio compression system using a Variational Autoencoder (VAE) architecture. It compresses stereo audio from 44.1kHz (88,200 samples for 2 seconds) down to a compact latent representation at ~21.5Hz (43 latent vectors), achieving a **2048x compression ratio**.

### Key Technologies
- **Framework**: PyTorch 2.6.0+ with CUDA 12.4 support
- **Language**: Python 3.13+
- **Package Manager**: uv (modern Python package manager)
- **Experiment Tracking**: Weights & Biases (wandb)
- **Audio Processing**: torchaudio, soundfile

### Project Status
- ✅ Model architecture implemented
- ✅ Training pipeline implemented
- ⚠️ Training script currently not working (as noted in git history)
- 🔄 Active development phase

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
- `output_padding=stride % 2` for proper upsampling

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
- Reparameterization trick: `z = μ + ε * exp(0.5 * logvar)` where ε ~ N(0,1)

### Training Infrastructure

#### **VocalDataset** (`src/train/dataset.py`)
- Loads audio files (.wav, .mp3) recursively from data directory
- Resamples to target sample rate (44.1kHz)
- Converts mono to stereo by duplication
- Random cropping for data augmentation
- **Energy filtering**: Rejects silent samples (threshold: -40dB RMS)
- **Peak normalization**: Scales to target_peak=0.95

**Important**: Silent sample detection prevents training on silence, which can destabilize VAE training.

#### **VAETrainer** (`src/train/train.py`)
Comprehensive training loop with:

**Loss Function**:
```python
total_loss = recon_loss + kl_weight * kl_loss

recon_loss = MSE(x_recon, x)
kl_loss = -0.5 * mean(1 + logvar - mu² - exp(logvar))
```

**KL Annealing**: Gradually increases KL weight over `kl_anneal_epochs` to prevent posterior collapse.

**Features**:
- Mixed precision training (AMP) for faster training on GPUs
- Gradient clipping (max_norm=1.0)
- Train/validation split (default 90/10)
- Automatic checkpoint management (keeps last N checkpoints)
- Audio sample logging to wandb every N steps
- Best model tracking based on validation loss

**Configuration** (`src/train/config.yaml`):
- Model hyperparameters (channels, latent_dim, strides)
- Training settings (batch_size, learning_rate, epochs)
- Data pipeline (paths, sample rate, target_length)
- Checkpointing (frequency, retention)
- Logging (wandb project, sample frequency)

## Development Workflows

### Setup Environment

```bash
# Install uv package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install PyTorch with CUDA 12.4 support
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# Set up wandb API key
echo "WANDB_API_KEY=your_key_here" > .env
```

### Running Training

```bash
# Basic training
python src/train/train.py

# Resume from checkpoint
# Modify train.py line 385: trainer.train(resume_from="./checkpoints/checkpoint_epoch_50.pt")
```

### Testing Dataset

```bash
# Visualize dataset samples
python src/train/dataset.py
```

### Model Testing

```python
from src.model.VAE import VAE
import torch

# Initialize model
model = VAE()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
x = torch.randn(1, 2, 88200)  # Batch of stereo audio
mu, logvar, z, x_recon = model(x)
print(f"Input: {x.shape}, Latent: {z.shape}, Output: {x_recon.shape}")
```

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

### Git Conventions
- **Branch naming**: Use descriptive feature branches
- **Commit messages**: Clear, concise descriptions (see git log)
- **Recent commits**:
  - "Added training script (not working)"
  - "Added dataset.py class and created proper uv project setup toml"
  - "Small improvements to computational efficiency and removed print statements"

### File Organization Rules
1. **Model code**: All neural network modules in `src/model/`
2. **Training code**: Training logic, dataset, config in `src/train/`
3. **Data**: Place in `data/` directory (gitignored)
4. **Checkpoints**: Saved to `checkpoints/` (gitignored)
5. **Environment**: Sensitive keys in `.env` (gitignored)

## Common Tasks

### Adding a New Model Component
1. Create new file in `src/model/`
2. Inherit from `nn.Module`
3. Import and use in parent model (e.g., VAE.py)
4. Update `src/model/__init__.py` if needed

### Modifying Hyperparameters
1. Edit `src/train/config.yaml` for training changes
2. Edit `VAE.__init__()` signature for architecture changes
3. Keep config.yaml and model signature in sync

### Debugging Training Issues
1. Check print statements in `train.py` lines 139-142 for value ranges
2. Monitor wandb dashboard for loss curves
3. Listen to reconstructed audio samples to evaluate quality
4. Check for NaN/Inf in `mu`, `logvar`, `x_recon`

### Adding New Loss Components
1. Modify `VAETrainer.compute_loss()` method
2. Add corresponding config entries in `config.yaml`
3. Update wandb logging in `train_epoch()` and `validate()`

## Known Issues & TODOs

### Current Issues
1. **Training script not working** (as per latest commit)
   - Check dataset loading paths
   - Verify CUDA/GPU availability
   - Inspect loss values for NaN/explosion

### Debug Checklist When Training Fails
- [ ] Dataset found audio files (`Found N audio files` in output)
- [ ] GPU detected (`Using device: cuda`)
- [ ] No silent samples causing infinite loops
- [ ] Loss values are finite (not NaN/Inf)
- [ ] Gradient norms reasonable (< 100)
- [ ] Audio samples load without errors

### Potential Improvements
- [ ] Add learning rate scheduling
- [ ] Implement spectral loss for better perceptual quality
- [ ] Add data augmentation (pitch shift, time stretch)
- [ ] Multi-GPU training support
- [ ] TensorBoard support alongside wandb
- [ ] Unit tests for model components

## Important Notes

### Environment Variables
- **WANDB_API_KEY**: Required for experiment tracking
- Stored in `.env` file (DO NOT COMMIT)
- Get key from: https://wandb.ai/authorize

### Data Requirements
- Audio files in `.wav` or `.mp3` format
- Minimum length: `target_length` samples (40,960 = ~0.93s at 44.1kHz)
- Organized in `data/vocal_dataset/` (can be nested)
- Should contain varied, non-silent audio

### GPU Memory Considerations
- Model has ~10M+ parameters
- Batch size of 4 with mixed precision fits on ~8GB GPU
- Reduce batch size if OOM errors occur
- `target_length` directly impacts memory usage

### Debugging Print Statements
The training script currently has debug prints (lines 139-142):
```python
print(f"mu range: [{mu.min():.2f}, {mu.max():.2f}]")
print(f"logvar range: [{logvar.min():.2f}, {logvar.max():.2f}]")
print(f"x_recon range: [{x_recon.min():.2f}, {x_recon.max():.2f}]")
```
These should be removed or converted to logging once training is stable.

## When Making Changes

### Before Modifying Model Architecture
1. Read corresponding documentation in README.md
2. Understand compression ratios from strides
3. Calculate output shapes for new layers
4. Update comments with shape transformations

### Before Modifying Training
1. Check current config.yaml values
2. Consider impact on convergence (especially KL weight)
3. Test on small dataset first
4. Save checkpoint before major changes

### Before Committing
1. Remove debug print statements
2. Update comments if logic changed
3. Test code runs without errors
4. Update CLAUDE.md if workflow changed

## Quick Reference

### Key File Locations
- Main model: `src/model/VAE.py`
- Training entry point: `src/train/train.py`
- Configuration: `src/train/config.yaml`
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

---

**Last Updated**: Based on commit `1a54f20 - Added training script (not working)`

For questions about architecture details, see `README.md`.
For training configuration, see `src/train/config.yaml`.
