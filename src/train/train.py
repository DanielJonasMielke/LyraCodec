import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import wandb
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dataset import VocalDataset
from model.VAE import VAE

class VAETrainer:
    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device(self.config['system']['device'] 
                                   if torch.cuda.is_available() 
                                   else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = VAE(
            in_channels=self.config['model']['in_channels'],
            base_channels=self.config['model']['base_channels'],
            latent_dim=self.config['model']['latent_dim'],
            c_mults=self.config['model']['c_mults'],
            strides=self.config['model']['strides']
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=self.config['training']['adam_betas'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Setup mixed precision
        self.use_amp = self.config['system']['mixed_precision']
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Setup data
        self._setup_data()
        
        # Setup checkpointing
        self.checkpoint_dir = Path(self.config['checkpointing']['save_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def _setup_data(self):
        """Setup datasets and dataloaders"""
        # Load full dataset
        full_dataset = VocalDataset(
            data_dir=self.config['data']['train_dir'],
            target_length=self.config['data']['target_length'],
            target_sr=self.config['data']['target_sr']
        )
        
        # Split into train/val
        train_size = int(self.config['data']['train_split'] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        print(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )
    
    def compute_loss(self, x, mu, logvar, x_recon, epoch):
        """
        Compute VAE loss = Reconstruction Loss + KL Divergence
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence loss
        # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # KL annealing: gradually increase KL weight
        kl_anneal_epochs = self.config['training']['kl_anneal_epochs']
        if epoch < kl_anneal_epochs:
            kl_weight = self.config['training']['kl_weight'] * (epoch / kl_anneal_epochs)
        else:
            kl_weight = self.config['training']['kl_weight']
        
        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss, kl_weight
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
        
        for batch_idx, x in enumerate(pbar):
            x = x.to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                mu, logvar, z, x_recon = self.model(x)
                # print the max and min of mu and logvar
                print(f"mu range: [{mu.min():.2f}, {mu.max():.2f}]")
                print(f"logvar range: [{logvar.min():.2f}, {logvar.max():.2f}]")
                # print the max and min of x_recon
                print(f"x_recon range: [{x_recon.min():.2f}, {x_recon.max():.2f}]")
                loss, recon_loss, kl_loss, kl_weight = self.compute_loss(
                    x, mu, logvar, x_recon, epoch
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
                self.optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })
            
            # Log to wandb
            if self.global_step % self.config['logging']['log_every_n_steps'] == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/recon_loss': recon_loss.item(),
                    'train/kl_loss': kl_loss.item(),
                    'train/kl_weight': kl_weight,
                    'train/epoch': epoch,
                    'train/step': self.global_step
                })
            
            # Save audio samples every 50 steps
            if self.global_step % 10 == 0:
                num_samples = min(self.config['logging']['num_audio_samples'], x.shape[0])
                original_samples = x[:num_samples].cpu()
                reconstructed_samples = x_recon[:num_samples].detach().cpu()
                self._log_audio_samples(original_samples, reconstructed_samples, epoch, self.global_step)
            
            self.global_step += 1
        
        # Return average losses
        num_batches = len(self.train_loader)
        return (epoch_loss / num_batches, 
                epoch_recon_loss / num_batches, 
                epoch_kl_loss / num_batches)
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        
        # Store samples for audio logging
        original_samples = []
        reconstructed_samples = []
        
        for batch_idx, x in enumerate(tqdm(self.val_loader, desc="Validating")):
            x = x.to(self.device)
            
            mu, logvar, z, x_recon = self.model(x)
            loss, recon_loss, kl_loss, _ = self.compute_loss(
                x, mu, logvar, x_recon, epoch
            )
            
            val_loss += loss.item()
            val_recon_loss += recon_loss.item()
            val_kl_loss += kl_loss.item()
            
            # Save first batch samples for logging
            if batch_idx == 0:
                num_samples = min(self.config['logging']['num_audio_samples'], x.shape[0])
                original_samples = x[:num_samples].cpu()
                reconstructed_samples = x_recon[:num_samples].cpu()
        
        # Average losses
        num_batches = len(self.val_loader)
        avg_val_loss = val_loss / num_batches
        avg_recon_loss = val_recon_loss / num_batches
        avg_kl_loss = val_kl_loss / num_batches
        
        # Log to wandb
        wandb.log({
            'val/loss': avg_val_loss,
            'val/recon_loss': avg_recon_loss,
            'val/kl_loss': avg_kl_loss,
            'val/epoch': epoch
        })
        
        # Log audio samples (keeping epoch-based validation logging as well)
        if epoch % self.config['logging']['save_audio_every_n_epochs'] == 0:
            self._log_audio_samples(original_samples, reconstructed_samples, epoch, step=None)
        
        return avg_val_loss
    
    def _log_audio_samples(self, original, reconstructed, epoch, step=None):
        """Log audio samples to wandb"""
        sample_rate = self.config['data']['target_sr']
        
        step_info = f"step {step}" if step is not None else f"epoch {epoch}"
        
        for i in range(len(original)):
            # Convert from (channels, samples) to (samples, channels) for wandb
            # Also convert to float32 (soundfile doesn't support float16)
            orig_audio = original[i].float().numpy().T  # Transpose to (samples, channels)
            recon_audio = reconstructed[i].float().numpy().T  # Transpose to (samples, channels)
            
            # Log original
            wandb.log({
                f'audio/sample_{i}_original': wandb.Audio(
                    orig_audio, 
                    sample_rate=sample_rate,
                    caption=f"Original ({step_info})"
                )
            })
            
            # Log reconstruction
            wandb.log({
                f'audio/sample_{i}_reconstructed': wandb.Audio(
                    recon_audio,
                    sample_rate=sample_rate,
                    caption=f"Reconstructed ({step_info})"
                )
            })
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
        
        # Clean old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Keep only the last N checkpoints"""
        keep_n = self.config['checkpointing']['keep_last_n']
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if len(checkpoints) > keep_n:
            for old_ckpt in checkpoints[:-keep_n]:
                old_ckpt.unlink()
                print(f"Removed old checkpoint: {old_ckpt}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self, resume_from=None):
        """Main training loop"""
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Initialize wandb
        wandb.init(
            project=self.config['logging']['wandb_project'],
            config=self.config,
            resume='allow' if resume_from else False
        )
        
        print(f"\nStarting training for {self.config['training']['num_epochs']} epochs")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            # Train
            train_loss, train_recon, train_kl = self.train_epoch(epoch)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
            
            # Validate
            val_loss = self.validate(epoch)
            print(f"  Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  New best validation loss!")
            
            if (epoch + 1) % self.config['checkpointing']['save_every_n_epochs'] == 0:
                self.save_checkpoint(epoch + 1, val_loss, is_best)
            
            self.current_epoch = epoch + 1
        
        print("\nTraining complete!")
        wandb.finish()


if __name__ == "__main__":
    # Create trainer
    trainer = VAETrainer("src/train/config.yaml")
    
    # Start training (or resume from checkpoint)
    # trainer.train(resume_from="./checkpoints/checkpoint_epoch_50.pt")
    trainer.train()