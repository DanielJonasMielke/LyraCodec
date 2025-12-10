# import typeddict
from typing import TypedDict

class StftLossConfig(TypedDict):
    fft_sizes: list[int]
    hop_sizes: list[int]
    win_lengths: list[int]
    perceptual_weighting: bool

class LRSchedulerConfig(TypedDict):
    warmup_steps: int
    inv_gamma: float
    power: float

class TrainingConfig(TypedDict):
    learning_rate: float
    lr_scheduler: LRSchedulerConfig
    batch_size: int
    num_epochs: int
    weight_decay: float
    kl_weight: float
    kl_warmup_steps: int
    log_interval: int
    validate_and_save_every: int
    generate_sample_interval: int
    checkpoint_interval: int
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

class WandbConfig(TypedDict):
    project: str
    run_name: str | None


class Config(TypedDict):
    hyperparameters: HyperparametersConfig
    device: DeviceConfig
    wandb: WandbConfig

