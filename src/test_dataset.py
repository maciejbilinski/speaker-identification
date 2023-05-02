import os
import pandas as pd
import torchaudio
import torch
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset

class AudioTestDataset(Dataset):
    def __init__(self, tsv_file, audio_dir):
        df = pd.read_csv(tsv_file, delimiter='\t')
        self.samples = pd.DataFrame({
            'client_id': df['client_id'],
            'path': df['path'].apply(lambda x: os.path.join(audio_dir, x)),
        })
        self.audio_transforms = MelSpectrogram(n_mels=64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_client_id, anchor_path = self.samples.iloc[idx]

        positive_df = self.samples[self.samples['client_id'] == anchor_client_id]
        positive_idx = torch.randint(high=len(positive_df), size=(1,))
        positive_path = positive_df.iloc[positive_idx]['path'].item()


        negative_df = self.samples[self.samples['client_id'] != anchor_client_id]
        negative_idx = torch.randint(high=len(negative_df), size=(1,))
        negative_path = negative_df.iloc[negative_idx]['path'].item()

        anchor_waveform, _ = torchaudio.load(anchor_path)
        positive_waveform, _ = torchaudio.load(positive_path)
        negative_waveform, _ = torchaudio.load(negative_path)

        anchor_spectrogram = self.audio_transforms(anchor_waveform)
        positive_spectrogram = self.audio_transforms(positive_waveform)
        negative_spectrogram = self.audio_transforms(negative_waveform)

        return (anchor_client_id, anchor_spectrogram), (anchor_client_id, positive_spectrogram), (negative_df.iloc[negative_idx]['client_id'].item(), negative_spectrogram)
