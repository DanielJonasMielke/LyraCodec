import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path

class VocalDataset(Dataset):
    def __init__(self, data_dir, target_length=90112, target_sr=44100):
        self.data_dir = Path(data_dir)
        self.target_length = target_length
        self.target_sr = target_sr
        
        # Find all audio files
        self.audio_files = list(self.data_dir.rglob("*.wav")) + \
                          list(self.data_dir.rglob("*.mp3"))
        
        print(f"Found {len(self.audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio
        waveform, _sr = torchaudio.load(self.audio_files[idx])
        
        # Resample if needed
        if _sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=_sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
        
        # If mono, convert to stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        
        # Ensure minimum length
        if waveform.shape[1] < self.target_length:
            padding = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Randomly crop to target length
        max_start = waveform.shape[1] - self.target_length
        if max_start > 0:
            rand_start = torch.randint(0, max_start + 1, (1,)).item()
        else:
            rand_start = 0
        
        wav_snippet = waveform[:, rand_start:rand_start + self.target_length]
        # Check energy and normalize
        if not self.check_audio_energy(wav_snippet, threshold_db=-40):
            # If silent, resample until non-silent
            while not self.check_audio_energy(wav_snippet, threshold_db=-40):
                print(f"[SILENT SAMPLE] Resampling index {idx}")
                rand_start = torch.randint(0, max_start + 1, (1,)).item()
                wav_snippet = waveform[:, rand_start:rand_start + self.target_length]
        wav_snippet = self.normalize_audio(wav_snippet)
        return wav_snippet
    
    def check_audio_energy(self, waveform, threshold_db=-40):
        """Return True if audio has sufficient energy"""
        rms = torch.sqrt(torch.mean(waveform ** 2))
        rms_db = 20 * torch.log10(rms + 1e-8)
        return rms_db > threshold_db
    
    def normalize_audio(self, waveform, target_peak=0.95):
        """Peak normalize audio to target_peak level"""
        # Find the maximum absolute value across all channels
        max_val = waveform.abs().max()
        
        # Avoid division by zero
        if max_val > 0:
            waveform = waveform * (target_peak / max_val)
        
        return waveform
    

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = VocalDataset(data_dir="./data/vocal_dataset")
    sample = dataset[50] # 50 silent sample
    print(f"Sample shape: {sample.shape}")  # Should be (2, target
    print(sample)
    # Plot waveform
    plt.figure(figsize=(10, 4))
    plt.plot(sample.t().numpy())
    plt.title("Waveform")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()