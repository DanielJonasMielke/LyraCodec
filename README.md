uv pip install torch --index-url https://download.pytorch.org/whl/cu124

---

# Audio VAE Encoder Architecture

## Overview

The encoder compresses stereo audio from 44.1kHz (88,200 samples for 2 seconds) down to a compact latent representation at ~21.5Hz (43 latent vectors), achieving a **2048x compression ratio**.

## Architecture Components

### 1. Snake Activation Function

A learnable periodic activation function designed for audio signals:

```
Snake(x) = x + (1/β) × sin²(α × x)
```

- **α (alpha)**: Controls frequency of periodic component
- **β (beta)**: Controls strength of the effect
- **Benefits**:
  - Captures periodic patterns in audio naturally
  - Smooth gradients (no "dying neuron" problem like ReLU)
  - Each channel learns its own periodicity parameters

### 2. Residual Unit

Processes features at multiple temporal scales using dilated convolutions with skip connections:

```
Structure:
  Input → [Snake → Conv(kernel=7, dilation=d) → Snake → Conv(kernel=1)] → + Input → Output
            └────────────── transformations ──────────────┘              ↑
                                                                    skip connection
```

**Key Features:**

- Three units per block with dilations [1, 3, 9] capture local to long-range patterns
- Skip connections enable gradient flow through deep networks
- Preserves spatial dimensions (no compression)

### 3. Encoder Block

Combines feature processing with downsampling:

```
Input
  ↓
ResidualUnit(dilation=1)  ← Fine-grained patterns
  ↓
ResidualUnit(dilation=3)  ← Medium-range patterns
  ↓
ResidualUnit(dilation=9)  ← Long-range patterns
  ↓
Snake Activation
  ↓
Downsampling Conv(stride=s)  ← Compression step
  ↓
Output
```

## Complete Encoder Pipeline

### Layer-by-Layer Transformation

| Layer            | Input Shape       | Output Shape      | Operation                       | Compression |
| ---------------- | ----------------- | ----------------- | ------------------------------- | ----------- |
| **Initial Conv** | `[B, 2, 88200]`   | `[B, 128, 88200]` | Conv1d(kernel=7)                | 1x          |
| **Block 1**      | `[B, 128, 88200]` | `[B, 256, 44100]` | 3×ResidualUnit + Conv(stride=2) | 2x          |
| **Block 2**      | `[B, 256, 44100]` | `[B, 512, 11025]` | 3×ResidualUnit + Conv(stride=4) | 4x          |
| **Block 3**      | `[B, 512, 11025]` | `[B, 1024, 2756]` | 3×ResidualUnit + Conv(stride=4) | 4x          |
| **Block 4**      | `[B, 1024, 2756]` | `[B, 2048, 344]`  | 3×ResidualUnit + Conv(stride=8) | 8x          |
| **Block 5**      | `[B, 2048, 344]`  | `[B, 2048, 43]`   | 3×ResidualUnit + Conv(stride=8) | 8x          |
| **Final Conv**   | `[B, 2048, 43]`   | `[B, 128, 43]`    | Snake + Conv1d(kernel=3)        | -           |
| **Split**        | `[B, 128, 43]`    | `[B, 64, 43]` × 2 | Split into μ and log(σ²)        | -           |

**Total Compression:** 88,200 → 43 samples = **2048x** reduction

### Hyperparameters

```python
in_channels = 2              # Stereo audio
base_channels = 128          # Starting feature dimension
latent_dim = 64              # Final latent dimensions
c_mults = [1, 2, 4, 8, 16]  # Channel multipliers per block
strides = [2, 4, 4, 8, 8]    # Downsampling factors per block
```

## Design Rationale

### Multi-Scale Processing (Dilated Convolutions)

- **Dilation 1**: Captures immediate temporal relationships (e.g., pitch, timbre)
- **Dilation 3**: Captures short-term patterns (e.g., note transitions)
- **Dilation 9**: Captures long-term structure (e.g., rhythm, phrasing)

### Progressive Downsampling

Each encoder block:

1. **Refines** features at current resolution (ResidualUnits)
2. **Compresses** to next level (downsampling convolution)

This allows the network to extract meaningful features before aggressive compression.

### Skip Connections

With 5 encoder blocks × 13 layers/block ≈ **65 total layers**, skip connections are essential for:

- Preventing vanishing gradients
- Enabling stable training of very deep networks
- Preserving information flow

## Output

The encoder outputs two tensors for VAE reparameterization:

- **μ (mu)**: `[batch, 64, 43]` - Mean of latent distribution
- **log(σ²) (logvar)**: `[batch, 64, 43]` - Log-variance of latent distribution

These represent a compressed, probabilistic encoding of the input audio at ~21.5Hz latent rate.
