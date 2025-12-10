import wandb
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import auraloss

from src.train.dataset import VocalDataset
from src.model.VAE import VAE
from src.types import Config, LRSchedulerConfig

def get_lr_lambda(config: LRSchedulerConfig):
    """
    Create a learning rate lambda function for a scheduler with warmup and inverse decay.
    1. Warmup phase: linear increase from 0 to 1 over warmup_steps
    2. Inverse decay phase: (1 + step / inv_gamma) ** (-power)
    """
    def lr_lambda(step: int) -> float:
        # Phase 1: Warmup
        if step < config['warmup_steps']:
            return step / config['warmup_steps']
        
        # Phase 2: Inverse decay
        return (1 + step / config['inv_gamma']) ** (-config['power'])
    
    return lr_lambda

def load_config(config_path) -> Config:
    """Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Config: Configuration parameters loaded from the YAML file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_device(config: Config) -> str:
    """Determine the device to use based on configuration.

    Args:
        config (Config): Configuration parameters.

    Returns:
        str: 'cuda' if CUDA is to be used, otherwise 'cpu'.
    """
    if config.get('device', {}).get('use_cuda', False):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cpu'

def init_dataloader(config: Config) -> DataLoader:
    """Initialize the DataLoader for the dataset.

    Args:
        config (Config): Configuration parameters.

    Returns:
        DataLoader: Initialized DataLoader for the dataset.
    """
    dataset_params = config['hyperparameters']['data']
    dataset = VocalDataset(
        data_dir=dataset_params['path'],
        target_sr=dataset_params['sample_rate'],
        target_length=dataset_params['target_length']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['hyperparameters']['training']['batch_size'],
        shuffle=True,
        num_workers=dataset_params['num_workers']
    )

    print(f"Dataset loaded with {len(dataset)} samples.")
    print(f"Steps per epoch: {len(dataloader)}")
    return dataloader

def init_model(config: Config, device: str):
    """Initialize the VAE model based on configuration.
    Args:
        config (Config): Configuration parameters.
        device (str): Device to load the model onto.
    Returns:
        VAE: Initialized VAE model.
    """
    model_params = config['hyperparameters']['model']
    model = VAE(
        in_channels=model_params['in_channels'],
        base_channels=model_params['base_channels'],
        latent_dim=model_params['latent_dim'],
        c_mults=model_params['c_mults'],
        strides=model_params['strides']
    ).to(device)

    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model

def init_loss_function(config: Config, device: str):
    """Initialize the STFT loss function based on configuration.

    Args:
        config (Config): Configuration parameters.
        device (str): Device to load the loss function onto.

    Returns:
        auraloss.freq.MultiResolutionSTFTLoss: Initialized STFT loss function.
    """
    stft_params = config['hyperparameters']['training']['stft_loss_params']
    stft_loss = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=stft_params['fft_sizes'],
        hop_sizes=stft_params['hop_sizes'],
        win_lengths=stft_params['win_lengths'],
        perceptual_weighting=stft_params['perceptual_weighting'],
        sample_rate=config['hyperparameters']['data']['sample_rate'],
    ).to(device)
    print(f"STFT Loss function initialized with resolutions: {stft_params['fft_sizes']}")

    return stft_loss

def init_optimizer(model, config: Config):
    """Initialize the optimizer for the model based on configuration.

    Args:
        model (VAE): The model to optimize.
        config (Config): Configuration parameters.
    Returns:
        torch.optim.Optimizer: Initialized optimizer.
    """
    training_params = config['hyperparameters']['training']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_params['learning_rate'],
        betas=training_params['betas'],
        weight_decay=training_params['weight_decay']
    )
    print(f"Optimizer initialized: Adam with learning rate {training_params['learning_rate']}")
    return optimizer

def init_scheduler(optimizer, config: Config):
    scheduler_params = config['hyperparameters']['training']['lr_scheduler']
    lr_lambda = get_lr_lambda(scheduler_params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    print(f"Scheduler initialized: InverseLR with warmup={scheduler_params['warmup_steps']}")
    return scheduler

def compute_loss(x_recon, x_original, mu, logvar, stft_loss_fn, config: Config):
    """
    STFT reconstruction loss + KL divergence
    """
    kl_weight = config['hyperparameters']['training']['kl_weight']

    recon_loss = stft_loss_fn(x_recon, x_original)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_weight * kl_loss
    return recon_loss, kl_loss, total_loss

def init_wandb(config: Config):
    """
    Initialize Weights & Biases logging.
    """
    wandb_config = config['wandb']
    wandb.init(
        project=wandb_config['project'],
        config=config,
        name=wandb_config['run_name']
    )
    print("Weights & Biases logging initialized.")

def train(model: torch.nn.Module, 
          dataloader: DataLoader, 
          optimizer: torch.optim.Adam, 
          scheduler: torch.optim.lr_scheduler.LambdaLR,
          loss_fn: auraloss.freq.MultiResolutionSTFTLoss,
          config: Config,
          device: str):
    print("\nStarting training...")

    train_conf = config['hyperparameters']['training']

    global_step = 0

    for num_epoch in range(train_conf['num_epochs']):
        print("=" * 50)
        print(f"Starting epoch number: {num_epoch}")
        model.train()
    
        for batch_idx, batch in enumerate(dataloader):
            print(f"Processing batch {batch_idx+1}/{len(dataloader)}", end='\r')
            # Move batch to device
            audio_batch = batch.to(device)
            # forward pass
            mu, logvar, _z, x_recon = model(audio_batch)
            # compute loss
            recon_loss, kl_loss, total_loss = compute_loss(
                x_recon, audio_batch, mu, logvar, loss_fn, config
            )
            # backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # Logging
            if global_step % train_conf['loss_log_interval'] == 0:
                current_lr = scheduler.get_last_lr()[0]

                wandb.log({
                    'train/total_loss': total_loss.item(),
                    'train/recon_loss': recon_loss.item(),
                    'train/kl_loss': kl_loss.item(),
                    'train/weighted_kl_loss': (train_conf['kl_weight'] * kl_loss).item(),
                    'train/learning_rate': current_lr,
                    'train/epoch': num_epoch,
                    'train/step': global_step
                }, step=global_step)


            # Generate sample audio at intervals
            if global_step % train_conf['generate_sample_interval'] == 0:
                pass

            # Save model checkpoint at intervals
            if global_step % train_conf['checkpoint_interval'] == 0:
                pass

            global_step += 1


if __name__ == "__main__":
    # ============================================
    # LOAD CONFIGURATION
    # ============================================
    config_path = Path(__file__).parent / 'config.yaml'
    config = load_config(config_path)

    device = get_device(config)
    print(f"Using device: {device}")

    # ============================================
    # INITIALIZE DATASET
    # ============================================
    dataloader = init_dataloader(config)

    # ============================================
    # INITIALIZE MODEL
    # ============================================
    model = init_model(config, device)

    # ============================================
    # INITIALIZE LOSS FUNCTION
    # ============================================
    stft_loss = init_loss_function(config, device)

    # ============================================
    # INITIALIZE OPTIMIZER
    # ============================================
    optimizer = init_optimizer(model, config)

    # ============================================
    # INITIALIZE SCHEDULER
    # ============================================
    scheduler = init_scheduler(optimizer, config)

    # ============================================
    # INITIALIZE WANDB
    # ============================================
    init_wandb(config)

    # ============================================
    # Training 
    # ============================================
    train(model, dataloader, optimizer, scheduler, stft_loss, config, device)