import torch
import torch.nn as nn

class STFTDiscriminator(nn.Module):
    def __init__(self, window_size=2048, hop_length=512):
        super().__init__()
        self.window_size = window_size
        self.hop_length = hop_length

        self.register_buffer('window', torch.hann_window(window_size))
        
        ch = 32
        
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1, ch, (3, 9), stride=(1, 1), padding=(1, 4))),
            nn.utils.weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            nn.utils.weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            nn.utils.weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))),
            nn.utils.weight_norm(nn.Conv2d(ch, ch, (3, 3), stride=(1, 1), padding=(1, 1))),
        ])
        
        self.conv_post = nn.utils.weight_norm(nn.Conv2d(ch, 1, (3, 3), stride=(1, 1), padding=(1, 1)))
        
        self.lrelu = nn.LeakyReLU(0.2)
        
    def compute_stft(self, audio):
        # audio: [batch, channels, time_samples]
        batch_size, channels, _ = audio.shape
        
        # Compute STFT for each channel
        stfts = []
        for ch in range(channels):
            stft = torch.stft(
                audio[:, ch, :],
                n_fft=self.window_size,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True
            )
            magnitude = torch.abs(stft)
            stfts.append(magnitude)
        
        # Average across channels
        # Stack: [channels, batch, freq_bins, time_frames]
        # Mean over dim=0 to get: [batch, freq_bins, time_frames]
        magnitude = torch.stack(stfts, dim=0).mean(dim=0)
        
        # Add channel dimension for CNN: [batch, 1, freq_bins, time_frames]
        magnitude = magnitude.unsqueeze(1)
        
        return magnitude

        
    def forward(self, audio):
        # audio: [batch, channels, time_samples]
        x = self.compute_stft(audio)  # [batch, 1, freq_bins, time_frames]
        
        feature_maps = []
        for conv in self.convs:
            x = conv(x)
            x = self.lrelu(x)
            feature_maps.append(x)
        
        x = self.conv_post(x)
        feature_maps.append(x)
        
        return x, feature_maps