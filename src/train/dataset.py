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
        
        return wav_snippet
    
if __name__ == "__main__":
    dataset = VocalDataset("./data/vocal_dataset")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")