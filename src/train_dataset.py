import os
import pandas as pd
import torchaudio
import torch
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset

class AudioTrainDataset(Dataset):
    def __init__(self, csv_file, audio_dir):
        df = pd.read_csv(csv_file)
        self.samples = pd.DataFrame({
            'client_id': df['client_id'],
            'path': df['path'].apply(lambda x: os.path.join(audio_dir, x)),
        })

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

        anchor_spectrogram = torch.load(anchor_path)
        positive_spectrogram = torch.load(positive_path)
        negative_spectrogram = torch.load(negative_path)

        return anchor_spectrogram, positive_spectrogram, negative_spectrogram
