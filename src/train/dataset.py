import torch
import torchaudio
import json
from torch.utils.data import Dataset

class VocalDataset(Dataset):
    def __init__(self, dictionary_path, sample_rate=44100, samples_per_chunk=71680, max_padding_percentage=0.2):
        """
        Args:
            dictionary_path: Path to the JSON dictionary created by create_chunks_dictionary.py
            sample_rate: Sample rate of audio (44100 for your case)
        """
        self.sample_rate = sample_rate
        self.samples_per_chunk = samples_per_chunk
        
        # Load the dictionary
        with open(dictionary_path, 'r') as f:
            self.chunks_dict = json.load(f)
        
        # Create a flat list of all chunks with their file paths
        self.chunk_list = []
        for file_path, chunks in self.chunks_dict.items():
            for chunk_info in chunks:
                sample_start = chunk_info['start_sample']
                sample_end = chunk_info['end_sample']
                sample_length = sample_end - sample_start
                padding_needed = self.samples_per_chunk - sample_length
                if padding_needed > 0:
                    padding_percentage = padding_needed / self.samples_per_chunk
                    if padding_percentage > max_padding_percentage:
                        # Skip this chunk
                        continue

                self.chunk_list.append({
                    'file_path': file_path,
                    'start_sample': chunk_info['start_sample'],
                    'end_sample': chunk_info['end_sample']
                })
        
        print(f"Loaded {len(self.chunk_list)} total chunks from {len(self.chunks_dict)} files")

    def normalize_audio(self, waveform, target_peak=0.95):
        """Peak normalize audio to target_peak level"""
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform * (target_peak / max_val)
        return waveform
    
    def __len__(self):
        """Return total number of chunks"""
        return len(self.chunk_list)
    
    def __getitem__(self, idx):
        """
        Load and return a single chunk of audio
        
        Args:
            idx: Index of chunk to load
            
        Returns:
            audio: Tensor of shape (1, num_samples) for mono at 44.1kHz
        """
        chunk_info = self.chunk_list[idx]
        file_path = chunk_info['file_path']
        start_sample = chunk_info['start_sample']
        end_sample = chunk_info['end_sample']
        
        try:
            # Load the entire audio file
            audio, sr = torchaudio.load(file_path)
            
            # Verify sample rate
            if sr != self.sample_rate:
                raise ValueError(f"Sample rate mismatch: {sr} vs {self.sample_rate}")
            
            # Convert stereo to mono by averaging channels
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            
            # Extract the specific chunk
            chunk_audio = audio[:, start_sample:end_sample]
            
            # Check if chunk is smaller than expected
            num_samples = chunk_audio.shape[1]
            
            if num_samples < self.samples_per_chunk:
                # Pad with zeros
                padding = self.samples_per_chunk - num_samples
                chunk_audio = torch.nn.functional.pad(
                    chunk_audio, 
                    (0, padding),
                    mode='constant',
                    value=0
                )

            # Normalize audio
            chunk_audio = self.normalize_audio(chunk_audio)
            
            return chunk_audio
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return silence on error (mono audio)
            return torch.zeros((1, self.samples_per_chunk))
