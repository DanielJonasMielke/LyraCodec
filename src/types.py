# import typeddict
from typing import TypedDict

class StftLossConfig(TypedDict):
    fft_sizes: list[int]
    hop_sizes: list[int]
    win_lengths: list[int]
    perceptual_weighting: bool

class TrainingConfig(TypedDict):
    learning_rate: float
    batch_size: int
    num_epochs: int
    weight_decay: float
    lr_scheduler_step_size: int
    lr_scheduler_gamma: float
    betas: list[float]
    stft_loss_params: StftLossConfig


class DataConfig(TypedDict):
    path: str
    sample_rate: int
    target_length: int
    num_workers: int


class ModelConfig(TypedDict):
    in_channels: int
    base_channels: int
    latent_dim: int
    c_mults: list[tuple[int, int]]
    strides: list[int]


class HyperparametersConfig(TypedDict):
    training: TrainingConfig
    data: DataConfig
    model: ModelConfig


class DeviceConfig(TypedDict):
    use_cuda: bool


class Config(TypedDict):
    hyperparameters: HyperparametersConfig
    device: DeviceConfig

